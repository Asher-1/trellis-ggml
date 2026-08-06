#!/usr/bin/env python3
"""PyTorch CUDA benchmark for TRELLIS.2 on RTX 3060.

Runs E2E inference with per-stage timing using the upstream TRELLIS.2 pipeline
loaded from local model files. Uses the same input image (T.png) as the ggml
benchmarks for fair comparison.

Usage:
    python scripts/bench_pytorch_cuda.py --input assets/example_image/T.png
"""

import os
import sys
import time
import json
import argparse

# Setup upstream TRELLIS.2 Python path
UPSTREAM = os.environ.get(
    "TRELLIS2_UPSTREAM",
    "/home/ludahai/develop/code/github/dl/TRELLIS.2-upstream"
)
sys.path.insert(0, UPSTREAM)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["ATTN_BACKEND"] = "sdpa"
os.environ["SPARSE_CONV_BACKEND"] = "none"

import torch
import numpy as np
from PIL import Image


def stub_o_voxel():
    """Stub out o_voxel (CUDA hashmap mesher) with pure-PyTorch CPU fallback."""
    import types
    if "o_voxel" not in sys.modules:
        ovx = types.ModuleType("o_voxel")
        ovx_convert = types.ModuleType("o_voxel.convert")

        def flexible_dual_grid_to_mesh(
            coords, dual_vertices, intersected_flag, split_weight,
            aabb, voxel_size=None, grid_size=None, train=False,
        ):
            """Pure-PyTorch/CPU port of flexible_dual_grid_to_mesh.
            Uses Python dict instead of CUDA hashmap."""
            dev = coords.device
            # Resolve grid_size / voxel_size / aabb
            if isinstance(aabb, (list, tuple)):
                aabb_t = torch.tensor(aabb, dtype=torch.float32, device=dev)
            else:
                aabb_t = aabb.to(dev)
            if isinstance(grid_size, int):
                gs = torch.tensor([grid_size]*3, dtype=torch.int32, device=dev)
            elif isinstance(grid_size, (list, tuple)):
                gs = torch.tensor(grid_size, dtype=torch.int32, device=dev)
            else:
                gs = grid_size.to(dev)
            vs = (aabb_t[1] - aabb_t[0]) / gs.float()

            N = coords.shape[0]
            # Mesh vertices: (coords + dual_vertices) * voxel_size + aabb[0]
            mesh_verts = (coords.float() + dual_vertices) * vs.unsqueeze(0) + aabb_t[0].unsqueeze(0)

            # Build coord->index hashmap (Python dict on CPU)
            coords_cpu = coords.cpu()
            coord_to_idx = {}
            for i in range(N):
                c = (coords_cpu[i, 0].item(), coords_cpu[i, 1].item(), coords_cpu[i, 2].item())
                coord_to_idx[c] = i

            EDGE_OFF = [
                [[0,0,0],[0,0,1],[0,1,1],[0,1,0]],  # x-axis
                [[0,0,0],[1,0,0],[1,0,1],[0,0,1]],  # y-axis
                [[0,0,0],[0,1,0],[1,1,0],[1,0,0]],  # z-axis
            ]
            SPLIT1 = [0, 1, 2, 0, 2, 3]
            SPLIT2 = [0, 1, 3, 3, 1, 2]

            tri_list = []
            interf_cpu = intersected_flag.cpu()
            for v in range(N):
                cx, cy, cz = coords_cpu[v, 0].item(), coords_cpu[v, 1].item(), coords_cpu[v, 2].item()
                for axis in range(3):
                    if interf_cpu[v, axis].item() <= 0:
                        continue
                    q = []
                    ok = True
                    for off in EDGE_OFF[axis]:
                        key = (cx + off[0], cy + off[1], cz + off[2])
                        if key not in coord_to_idx:
                            ok = False
                            break
                        q.append(coord_to_idx[key])
                    if not ok:
                        continue
                    if split_weight is None:
                        # Use normal alignment to choose split
                        mv = mesh_verts
                        t0 = [mv[q[i]] for i in SPLIT1[:3]]
                        n0 = torch.cross(t0[1]-t0[0], t0[2]-t0[0])
                        t1 = [mv[q[i]] for i in SPLIT1[3:]]
                        n1 = torch.cross(t1[1]-t1[0], t1[2]-t1[0])
                        a0 = (n0 * n1).sum().abs()
                        t0b = [mv[q[i]] for i in SPLIT2[:3]]
                        n0b = torch.cross(t0b[1]-t0b[0], t0b[2]-t0b[0])
                        t1b = [mv[q[i]] for i in SPLIT2[3:]]
                        n1b = torch.cross(t1b[1]-t1b[0], t1b[2]-t1b[0])
                        a1 = (n0b * n1b).sum().abs()
                        if a0 > a1:
                            tri_list.extend([q[i] for i in SPLIT1])
                        else:
                            tri_list.extend([q[i] for i in SPLIT2])
                    else:
                        sw = split_weight.cpu()
                        sw02 = sw[q[0]] * sw[q[2]]
                        sw13 = sw[q[1]] * sw[q[3]]
                        if sw02 > sw13:
                            tri_list.extend([q[i] for i in SPLIT1])
                        else:
                            tri_list.extend([q[i] for i in SPLIT2])

            if len(tri_list) == 0:
                faces = torch.zeros((0, 3), dtype=torch.int32, device=dev)
            else:
                faces = torch.tensor(tri_list, dtype=torch.int32, device=dev).reshape(-1, 3)
            return mesh_verts, faces

        ovx_convert.flexible_dual_grid_to_mesh = flexible_dual_grid_to_mesh
        ovx.convert = ovx_convert
        sys.modules["o_voxel"] = ovx
        sys.modules["o_voxel.convert"] = ovx_convert


def stub_cumesh():
    """Stub out cumesh."""
    import types
    if "cumesh" not in sys.modules:
        stub = types.ModuleType("cumesh")
        class _CuMeshStub:
            def __init__(self, *a, **k):
                raise RuntimeError("cumesh stubbed")
        stub.CuMesh = _CuMeshStub
        sys.modules["cumesh"] = stub


def stub_flex_gemm():
    """Stub out flex_gemm."""
    import types
    if "flex_gemm" not in sys.modules:
        fg = types.ModuleType("flex_gemm")
        fg_ops = types.ModuleType("flex_gemm.ops")
        fg_gs = types.ModuleType("flex_gemm.ops.grid_sample")
        fg_sp = types.ModuleType("flex_gemm.ops.spconv")
        def _unavail(*a, **k):
            raise RuntimeError("flex_gemm stubbed")
        fg_gs.grid_sample_3d = _unavail
        fg_sp.sparse_submanifold_conv3d = _unavail
        fg_ops.grid_sample = fg_gs
        fg_ops.spconv = fg_sp
        fg.ops = fg_ops
        sys.modules["flex_gemm"] = fg
        sys.modules["flex_gemm.ops"] = fg_ops
        sys.modules["flex_gemm.ops.grid_sample"] = fg_gs
        sys.modules["flex_gemm.ops.spconv"] = fg_sp


def install_sdpa_attention():
    """Replace sparse attention with SDPA for stock PyTorch."""
    from trellis2.modules.sparse.attention import full_attn
    from trellis2.modules.sparse import VarLenTensor
    import torch.nn.functional as F

    def sdpa_varlen(q, k, v, q_seqlen, kv_seqlen):
        CHUNK = 2048
        out = torch.empty_like(q)
        qo = ko = 0
        for ql, kl in zip(q_seqlen, kv_seqlen):
            ks = k[ko:ko + kl].transpose(0, 1).unsqueeze(0)
            vs = v[ko:ko + kl].transpose(0, 1).unsqueeze(0)
            for s in range(0, ql, CHUNK):
                e = min(s + CHUNK, ql)
                qs = q[qo + s:qo + e].transpose(0, 1).unsqueeze(0)
                o = F.scaled_dot_product_attention(qs, ks, vs)
                out[qo + s:qo + e] = o.squeeze(0).transpose(0, 1)
            qo += ql
            ko += kl
        return out

    def sparse_sdpa(*args, **kwargs):
        num = len(args) + len(kwargs)
        if num == 1:
            qkv = args[0] if args else kwargs["qkv"]
            q_seqlen = [qkv.layout[i].stop - qkv.layout[i].start for i in range(qkv.shape[0])]
            q, k, v = qkv.feats.unbind(dim=1)
            out = sdpa_varlen(q, k, v, q_seqlen, q_seqlen)
            return qkv.replace(out)
        if num == 2:
            q = args[0] if args else kwargs["q"]
            kv = args[1] if len(args) > 1 else kwargs["kv"]
            s = q if isinstance(q, VarLenTensor) else None
            if isinstance(q, VarLenTensor):
                q_seqlen = [q.layout[i].stop - q.layout[i].start for i in range(q.shape[0])]
                qf = q.feats
            else:
                N, L = q.shape[:2]
                q_seqlen = [L] * N
                qf = q.reshape(N * L, *q.shape[2:])
            if isinstance(kv, VarLenTensor):
                kv_seqlen = [kv.layout[i].stop - kv.layout[i].start for i in range(kv.shape[0])]
                kvf = kv.feats
            else:
                N, L = kv.shape[:2]
                kv_seqlen = [L] * N
                kvf = kv.reshape(N * L, *kv.shape[2:])
            k, v = kvf.unbind(dim=1)
            out = sdpa_varlen(qf, k, v, q_seqlen, kv_seqlen)
            if s is not None:
                return s.replace(out)
            N = len(q_seqlen)
            return out.reshape(N, q_seqlen[0], *out.shape[1:])
        if num == 3:
            q = args[0] if args else kwargs["q"]
            k = args[1] if len(args) > 1 else kwargs["k"]
            v = args[2] if len(args) > 2 else kwargs["v"]
            s = q if isinstance(q, VarLenTensor) else None
            if isinstance(q, VarLenTensor):
                q_seqlen = [q.layout[i].stop - q.layout[i].start for i in range(q.shape[0])]
                qf = q.feats
            else:
                N, L = q.shape[:2]
                q_seqlen = [L] * N
                qf = q.reshape(N * L, *q.shape[2:])
            if isinstance(k, VarLenTensor):
                kv_seqlen = [kv.layout[i].stop - kv.layout[i].start for i in range(kv.shape[0])]
                kf, vf = k.feats, v.feats
            else:
                N, L = k.shape[:2]
                kv_seqlen = [L] * N
                kf = k.reshape(N * L, *k.shape[2:])
                vf = v.reshape(N * L, *v.shape[2:])
            out = sdpa_varlen(qf, kf, vf, q_seqlen, kv_seqlen)
            if s is not None:
                return s.replace(out)
            N = len(q_seqlen)
            return out.reshape(N, q_seqlen[0], *out.shape[1:])
        raise AssertionError("bad arg count")

    full_attn.sparse_scaled_dot_product_attention = sparse_sdpa
    from trellis2.modules.sparse.attention import modules as attn_modules
    attn_modules.sparse_scaled_dot_product_attention = sparse_sdpa


def install_torch_sparse_conv():
    """Register pure-PyTorch sparse conv backend."""
    import math
    import types
    import torch
    import torch.nn as nn
    from trellis2.modules.sparse.conv import conv as conv_dispatch

    mod = types.ModuleType("trellis2.modules.sparse.conv.conv_none")

    def sparse_conv3d_init(self, in_channels, out_channels, kernel_size,
                           stride=1, dilation=1, padding=None, bias=True, indice_key=None):
        assert stride == 1 and padding is None
        self.in_channels = in_channels
        self.out_channels = out_channels
        ks = tuple(kernel_size) if isinstance(kernel_size, (list, tuple)) else (kernel_size,) * 3
        self.kernel_size = ks
        self.stride = (1, 1, 1)
        self.dilation = tuple(dilation) if isinstance(dilation, (list, tuple)) else (dilation,) * 3
        self.weight = nn.Parameter(torch.empty(out_channels, *ks, in_channels))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def _coord_key(coords, spatial_shape):
        b, x, y, z = coords.unbind(-1)
        sx, sy, sz = spatial_shape
        return ((b.long() * sx + x.long()) * sy + y.long()) * sz + z.long()

    def sparse_conv3d_forward(self, x):
        import numpy as np
        coords = x.coords
        feats = x.feats
        n = feats.shape[0]
        spatial = tuple(x.spatial_shape)
        Co, Kd, Kh, Kw, Ci = self.weight.shape
        w = self.weight
        out = feats.new_zeros(n, Co)
        if self.bias is not None:
            out += self.bias.to(out.dtype)
        rd, rh, rw = Kd // 2, Kh // 2, Kw // 2
        dd, dh, dw = self.dilation

        # Use numpy for fast vectorized neighbor lookup
        coords_np = coords.cpu().numpy().astype(np.int64)
        # Hash coords to unique keys for fast lookup
        sx, sy, sz = spatial
        keys_np = (coords_np[:, 0] * sx + coords_np[:, 1]) * sy + coords_np[:, 2]
        keys_full = keys_np * sz + coords_np[:, 3]
        sort_idx = np.argsort(keys_full)
        keys_sorted = keys_full[sort_idx]

        # Build offsets list
        offsets = []
        for kd in range(Kd):
            for kh in range(Kh):
                for kw in range(Kw):
                    offsets.append((kd, kh, kw,
                                    (kd - rd) * dd, (kh - rh) * dh, (kw - rw) * dw))

        dev = feats.device
        CHUNK = 8192
        for kd, kh, kw, od, oh, ow in offsets:
            # Vectorized neighbor coord computation
            nc = coords_np.copy()
            nc[:, 1] += od
            nc[:, 2] += oh
            nc[:, 3] += ow
            # Boundary check (vectorized)
            inb = ((nc[:, 1] >= 0) & (nc[:, 1] < spatial[0]) &
                   (nc[:, 2] >= 0) & (nc[:, 2] < spatial[1]) &
                   (nc[:, 3] >= 0) & (nc[:, 3] < spatial[2]))
            if not np.any(inb):
                continue
            # Hash neighbor coords
            nc_inb = nc[inb]
            nk = ((nc_inb[:, 0] * sx + nc_inb[:, 1]) * sy + nc_inb[:, 2]) * sz + nc_inb[:, 3]
            # searchsorted for neighbor lookup
            pos = np.searchsorted(keys_sorted, nk)
            pos_c = np.clip(pos, 0, n - 1)
            hit = (keys_sorted[pos_c] == nk)
            if not np.any(hit):
                continue
            # Build source index array
            src_local = np.where(hit)[0]
            src_orig = np.where(inb)[0][src_local]
            src_idx = sort_idx[pos_c[src_local]]
            # Gather+matmul in chunks on device
            src_t = torch.from_numpy(src_idx).long().to(dev)
            idx_t = torch.from_numpy(src_orig).long().to(dev)
            w_sl = w[:, kd, kh, kw, :].to(feats.dtype).t()
            for s in range(0, len(src_t), CHUNK):
                e = min(s + CHUNK, len(src_t))
                out[idx_t[s:e]] += feats[src_t[s:e]] @ w_sl
        return x.replace(out)

    def sparse_inverse_conv3d_init(self, *a, **k):
        raise NotImplementedError
    def sparse_inverse_conv3d_forward(self, x):
        raise NotImplementedError

    mod.sparse_conv3d_init = sparse_conv3d_init
    mod.sparse_conv3d_forward = sparse_conv3d_forward
    mod.sparse_inverse_conv3d_init = sparse_inverse_conv3d_init
    mod.sparse_inverse_conv3d_forward = sparse_inverse_conv3d_forward
    conv_dispatch._backends["none"] = mod
    from trellis2.modules import sparse as sp
    sp.config.CONV = "none"


def install_chunked_mlp():
    """Monkey-patch SparseConvNeXtBlock3d to process MLP in point-chunks.
    This reduces peak VRAM from ~1.6GB to ~128MB per intermediate tensor."""
    from trellis2.models.sc_vaes.sparse_unet_vae import SparseConvNeXtBlock3d
    import torch.utils.checkpoint

    MLP_CHUNK = 8192  # points per chunk for MLP

    _orig_forward = SparseConvNeXtBlock3d._forward

    def _chunked_forward(self, x):
        h = self.conv(x)
        h = h.replace(self.norm(h.feats))
        # Chunked MLP to limit peak VRAM
        feats = h.feats
        n = feats.shape[0]
        if n <= MLP_CHUNK:
            h = h.replace(self.mlp(feats))
        else:
            outs = []
            for s in range(0, n, MLP_CHUNK):
                outs.append(self.mlp(feats[s:s+MLP_CHUNK]))
            h = h.replace(torch.cat(outs, dim=0))
        return h + x

    SparseConvNeXtBlock3d._forward = _chunked_forward

    # Also patch the ResBlock if it has similar issues
    from trellis2.models.sc_vaes.sparse_unet_vae import SparseResBlock3d
    _orig_res_forward = SparseResBlock3d._forward

    def _chunked_res_forward(self, x):
        h = self.conv(x)
        h = h.replace(self.norm(h.feats))
        h = h.replace(self.mlp(h.feats))
        return h + x

    # Keep original for now (ResBlock MLP is smaller, may not need chunking)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="assets/example_image/T.png")
    ap.add_argument("--models", default="models/TRELLIS.2-4B")
    ap.add_argument("--pipeline-type", default="512", choices=["512", "1024", "1024_cascade"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu-only", action="store_true",
                    help="Only run GPU stages (skip CPU shape decode)")
    args = ap.parse_args()

    REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(REPO, args.input) if not os.path.isabs(args.input) else args.input
    models_path = os.path.join(REPO, args.models) if not os.path.isabs(args.models) else args.models

    print(f"=== PyTorch CUDA Benchmark on {torch.cuda.get_device_name(0)} ===")
    print(f"torch={torch.__version__}, CUDA={torch.version.cuda}")
    print(f"Input: {input_path}")
    print(f"Models: {models_path}")
    print(f"Pipeline type: {args.pipeline_type}")
    print()

    # Setup stubs
    stub_o_voxel()
    stub_cumesh()
    stub_flex_gemm()

    # Import after path setup
    # Add NoRembg stub (T.png has alpha, no need for background removal)
    from trellis2.pipelines import rembg as rembg_mod
    class NoRembg:
        def __init__(self, *a, **k): pass
        def __call__(self, img): return img.convert('RGBA')
        def to(self, device): pass
        def cuda(self): pass
        def cpu(self): pass
    rembg_mod.NoRembg = NoRembg

    from trellis2.pipelines import Trellis2ImageTo3DPipeline

    # Install SDPA attention + sparse conv + chunked MLP
    install_sdpa_attention()
    install_torch_sparse_conv()
    install_chunked_mlp()

    # Force true fp32 for fair comparison (no TF32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # Load pipeline from local models
    print("Loading pipeline...")
    t0 = time.time()
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(models_path)
    # Fix DINO model attribute access (transformers wraps encoder in .model)
    dino = pipeline.image_cond_model.model
    if not hasattr(dino, 'layer') and hasattr(dino, 'model') and hasattr(dino.model, 'layer'):
        dino.layer = dino.model.layer
        print("Patched DINO model.layer -> model.model.layer")
    # Use low_vram mode: keep models on CPU, move to GPU per-stage
    # This avoids OOM on 12GB RTX 3060
    dev = torch.device('cuda')
    pipeline._device = 'cuda'
    # Only move DINO to GPU (small, needed for conditioning)
    pipeline.image_cond_model.to(dev)
    print(f"Using low_vram mode on {torch.cuda.get_device_name(0)}")
    t_load = time.time() - t0
    print(f"Pipeline loaded in {t_load:.1f}s")

    # Load image
    image = Image.open(input_path)
    print(f"Input image: {image.size} {image.mode}")

    # Run E2E inference with timing
    print("\n--- Running E2E inference ---")
    torch.cuda.synchronize()
    t0 = time.time()

    # Preprocess
    t1 = time.time()
    processed = pipeline.preprocess_image(image)
    t_preprocess = time.time() - t1
    print(f"  Preprocess: {t_preprocess:.2f}s")

    # DINO conditioning
    torch.manual_seed(args.seed)
    t1 = time.time()
    cond_512 = pipeline.get_cond([processed], 512)
    t_dino = time.time() - t1
    print(f"  DINO conditioning (512): {t_dino:.2f}s")

    # SS-flow sampling
    t1 = time.time()
    coords = pipeline.sample_sparse_structure(
        cond_512, 32, 1,
        pipeline.sparse_structure_sampler_params
    )
    t_ss = time.time() - t1
    n_coords = coords.shape[0]
    print(f"  SS-flow sampling: {t_ss:.2f}s ({n_coords} active voxels)")
    # Move SS flow model back to CPU
    pipeline.models['sparse_structure_flow_model'].cpu()
    pipeline.models['sparse_structure_decoder'].cpu()
    torch.cuda.empty_cache()

    # Shape SLAT sampling
    t1 = time.time()
    if args.pipeline_type == "512":
        shape_slat = pipeline.sample_shape_slat(
            cond_512, pipeline.models['shape_slat_flow_model_512'],
            coords, pipeline.shape_slat_sampler_params
        )
    elif args.pipeline_type == "1024":
        cond_1024 = pipeline.get_cond([processed], 1024)
        shape_slat = pipeline.sample_shape_slat(
            cond_1024, pipeline.models['shape_slat_flow_model_1024'],
            coords, pipeline.shape_slat_sampler_params
        )
    t_shape_slat = time.time() - t1
    print(f"  Shape SLAT sampling: {t_shape_slat:.2f}s")
    # Move flow model back to CPU to free VRAM for decode
    if args.pipeline_type == "512":
        pipeline.models['shape_slat_flow_model_512'].cpu()
    else:
        pipeline.models['shape_slat_flow_model_1024'].cpu()
    torch.cuda.empty_cache()

    # Shape decode: run on CPU (12GB VRAM insufficient for "none" sparse conv backend)
    if args.gpu_only:
        print("  Shape decode: SKIPPED (--gpu-only)")
        t_shape_dec = 0
        n_verts = n_faces = 0
    else:
        t1 = time.time()
        pipeline.models['shape_slat_decoder'].set_resolution(512 if args.pipeline_type == "512" else 1024)
        pipeline.models['shape_slat_decoder'].cpu()
        shape_slat_cpu = shape_slat.cpu()
        torch.cuda.empty_cache()
        print(f"  Shape decode on CPU ({shape_slat_cpu.feats.shape[0]} points)...", flush=True)
        meshes, subs = pipeline.models['shape_slat_decoder'](shape_slat_cpu, return_subs=True)
        t_shape_dec = time.time() - t1
        n_verts = sum(m.vertices.shape[0] for m in meshes)
        n_faces = sum(m.faces.shape[0] for m in meshes)
        print(f"  Shape decode: {t_shape_dec:.2f}s ({n_verts} verts, {n_faces} faces)")

    torch.cuda.synchronize()
    t_total = time.time() - t0

    print(f"\n=== Results ===")
    print(f"Total inference: {t_total:.1f}s")
    print(f"  DINO:          {t_dino:.2f}s ({100*t_dino/t_total:.1f}%)")
    print(f"  SS-flow:       {t_ss:.2f}s ({100*t_ss/t_total:.1f}%)")
    print(f"  Shape SLAT:    {t_shape_slat:.2f}s ({100*t_shape_slat/t_total:.1f}%)")
    print(f"  Shape decode:  {t_shape_dec:.2f}s ({100*t_shape_dec/t_total:.1f}%)")
    print(f"Vertices: {n_verts:,}")
    print(f"Faces:    {n_faces:,}")

    # VRAM usage
    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated() / 1e9
        print(f"Peak VRAM: {peak_vram:.2f} GB")

    # Save results
    results = {
        "backend": "pytorch_cuda",
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "pipeline_type": args.pipeline_type,
        "total_s": round(t_total, 1),
        "dino_s": round(t_dino, 2),
        "ss_flow_s": round(t_ss, 2),
        "shape_slat_s": round(t_shape_slat, 2),
        "shape_dec_s": round(t_shape_dec, 2),
        "vertices": int(n_verts),
        "faces": int(n_faces),
        "seed": args.seed,
    }
    out_path = "/tmp/bench_pytorch_cuda.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
