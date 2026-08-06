#!/usr/bin/env python3
"""Export a UV-atlas PBR GLB using upstream o_voxel.postprocess.to_glb.

Matches the official TRELLIS.2 example.py export path (CuMesh simplify/unwrap,
nvdiffrast UV bake, flex_gemm grid_sample). Requires the full CUDA stack:
  cumesh, nvdiffrast, flex_gemm, o_voxel

Typical usage on trellis-ggml artifacts (re-runs the texture NN to recover the
sparse PBR volume, then calls upstream to_glb):

  python scripts/glb_upstream_bridge.py \\
      --mesh outputs/pbr_e2e/pbr_official_T.t2mesh \\
      --image assets/example_image/T.png \\
      --resolution 512 \\
      --out outputs/pbr_e2e/pbr_official_T_upstream.glb

Set TRELLIS2_PY to the upstream repo if not at /trellis2.
"""
from __future__ import annotations

import argparse
import os
import struct
import sys

import numpy as np
import torch
import trimesh
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

TRELLIS2_PY = os.environ.get("TRELLIS2_PY", "/trellis2")
if TRELLIS2_PY not in sys.path:
    sys.path.insert(0, TRELLIS2_PY)

import ref_common  # noqa: E402  # sparse/attn monkeypatches only; keep real o_voxel

import o_voxel  # noqa: E402  # must import before ref_common stubs o_voxel


def load_t2mesh(path: str):
    b = open(path, "rb").read()
    magic = b[:8]
    if magic not in (b"T2MESH01", b"T2MESH03"):
        raise SystemExit(f"unsupported mesh magic {magic!r}")
    nv, nt = struct.unpack("<II", b[8:16])
    o = 16
    V = np.frombuffer(b, "<f4", 3 * nv, o).reshape(-1, 3).copy()
    o += 12 * nv
    o += 12 * nv  # normals
    pbr = None
    if magic == b"T2MESH03":
        pbr = np.frombuffer(b, "<f4", 6 * nv, o).reshape(-1, 6).copy()
        o += 24 * nv
    F = np.frombuffer(b, "<i4", 3 * nt, o).reshape(-1, 3).copy()
    return V, F, pbr


def run_texture_volume(mesh: trimesh.Trimesh, image: Image.Image, resolution: int, seed: int, device: str):
    """Return (vertices, faces, attr_volume [L,C], coords [L,3], grid_size)."""
    import json
    from safetensors.torch import load_file
    from trellis2.models.sc_vaes.fdg_vae import FlexiDualGridVaeEncoder
    from trellis2.models.sc_vaes.sparse_unet_vae import SparseUnetVaeDecoder
    from trellis2.models.structured_latent_flow import SLatFlowModel
    from trellis2.pipelines import Trellis2TexturingPipeline
    from trellis2.pipelines.samplers import FlowEulerGuidanceIntervalSampler
    from trellis2.modules.image_feature_extractor import DinoV3FeatureExtractor

    ref_common.setup()
    CK = os.path.join(ref_common.MODELS, "TRELLIS.2-4B", "ckpts")
    tpa = json.load(open(os.path.join(ref_common.MODELS, "TRELLIS.2-4B", "texturing_pipeline.json")))["args"]

    def load_model(cls, stem, **extra):
        cfg = json.load(open(os.path.join(CK, stem + ".json")))["args"]
        m = cls(**{**cfg, **extra})
        sd = {k: v.float() for k, v in load_file(os.path.join(CK, stem + ".safetensors")).items()}
        m.load_state_dict(sd)
        return m.eval().to(device)

    shape_enc = load_model(FlexiDualGridVaeEncoder, "shape_enc_next_dc_f16c32_fp16")
    tex_dec = load_model(SparseUnetVaeDecoder, "tex_dec_next_dc_f16c32_fp16")
    tex_flow = load_model(SLatFlowModel, f"slat_flow_imgshape2tex_dit_1_3B_{resolution}_bf16", dtype="float32")
    dino = DinoV3FeatureExtractor(os.path.join(ref_common.MODELS, "dinov3-vitl16"), image_size=resolution)
    dino.model = dino.model.to(device).eval()

    sampler = FlowEulerGuidanceIntervalSampler(**tpa["tex_slat_sampler"]["args"])
    pipe = Trellis2TexturingPipeline(
        models={
            "shape_slat_encoder": shape_enc,
            "tex_slat_decoder": tex_dec,
            f"tex_slat_flow_model_{resolution}": tex_flow,
        },
        tex_slat_sampler=sampler,
        tex_slat_sampler_params=tpa["tex_slat_sampler"]["params"],
        shape_slat_normalization=tpa["shape_slat_normalization"],
        tex_slat_normalization=tpa["tex_slat_normalization"],
        image_cond_model=dino,
        rembg_model=None,
        low_vram=False,
    )
    pipe._device = device

    torch.manual_seed(seed)
    with torch.no_grad():
        img = pipe.preprocess_image(image)
        m = pipe.preprocess_mesh(mesh)
        cond = pipe.get_cond([img], resolution)
        shape_slat = pipe.encode_shape_slat(m, resolution)
        tex_slat = pipe.sample_tex_slat(cond, tex_flow, shape_slat)
        pbr = pipe.decode_tex_slat(tex_slat)
    verts = torch.from_numpy(m.vertices).float().to(device)
    faces = torch.from_numpy(m.faces).int().to(device)
    attr_volume = pbr.feats.float()
    coords = pbr.coords[:, 1:].int().to(device)
    grid_size = torch.tensor([resolution, resolution, resolution], device=device, dtype=torch.int32)
    return verts, faces, attr_volume, coords, grid_size


def main():
    ap = argparse.ArgumentParser(description="Upstream UV-atlas GLB export bridge")
    ap.add_argument("--mesh", required=True, help="T2MESH01/03 from trellis-ggml")
    ap.add_argument("--image", required=True, help="conditioning RGBA image")
    ap.add_argument("--out", required=True, help="output .glb path")
    ap.add_argument("--resolution", type=int, default=512, choices=[512, 1024])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--texture-size", type=int, default=2048)
    ap.add_argument("--decimation", type=int, default=1_000_000)
    ap.add_argument("--remesh", action="store_true", help="upstream remesh branch (example.py default)")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA required for upstream to_glb (cumesh/nvdiffrast)")

    V, F, _ = load_t2mesh(args.mesh)
    mesh = trimesh.Trimesh(vertices=V, faces=F, process=False)
    image = Image.open(args.image).convert("RGBA")
    print(f"mesh {len(V):,} verts / {len(F):,} tris; image {image.size}", flush=True)

    verts, faces, attr_volume, coords, grid_size = run_texture_volume(
        mesh, image, args.resolution, args.seed, args.device)

    attr_layout = {
        "base_color": slice(0, 3),
        "metallic": slice(3, 4),
        "roughness": slice(4, 5),
        "alpha": slice(5, 6),
    }
    print("calling o_voxel.postprocess.to_glb ...", flush=True)
    glb_mesh = o_voxel.postprocess.to_glb(
        vertices=verts,
        faces=faces,
        attr_volume=attr_volume,
        coords=coords,
        attr_layout=attr_layout,
        grid_size=grid_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=args.decimation,
        texture_size=args.texture_size,
        remesh=args.remesh,
        remesh_band=1,
        remesh_project=0.0,
        verbose=True,
        use_tqdm=True,
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    glb_mesh.export(args.out, extension_webp=True)
    print(f"wrote {args.out} ({os.path.getsize(args.out)/1024/1024:.2f} MB)", flush=True)


if __name__ == "__main__":
    main()
