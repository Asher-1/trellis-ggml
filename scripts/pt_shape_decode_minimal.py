#!/usr/bin/env python3
"""Minimal PyTorch shape decode: load shape decoder only, skip full pipeline.

Usage:
  python scripts/pt_shape_decode_minimal.py
"""

import os, sys, time, json
import numpy as np

UPSTREAM = os.environ.get(
    "TRELLIS2_UPSTREAM",
    "/home/ludahai/develop/code/github/dl/TRELLIS.2-upstream"
)
sys.path.insert(0, UPSTREAM)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["ATTN_BACKEND"] = "sdpa"
os.environ["SPARSE_CONV_BACKEND"] = "none"

import torch
from safetensors.torch import load_file

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INTERMEDIATE_PATH = "/tmp/pt_shape_slat.pt"
MESH_OUTPUT = "/tmp/pt_mesh_output.glb"
SHAPE_DEC_CKPT = os.path.join(REPO, "models/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16.safetensors")
SHAPE_DEC_CONFIG = os.path.join(REPO, "models/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16.json")


# ── stubs & patches ──────────────────────────────────────────────────

def stub_o_voxel():
    import types
    if "o_voxel" not in sys.modules:
        ovx = types.ModuleType("o_voxel")
        ovx_convert = types.ModuleType("o_voxel.convert")
        def flexible_dual_grid_to_mesh(coords, dual_vertices, intersected_flag, split_weight,
                                        aabb, voxel_size=None, grid_size=None, train=False):
            dev = coords.device
            if isinstance(aabb, (list, tuple)):
                aabb_t = torch.tensor(aabb, dtype=torch.float32, device=dev)
            else:
                aabb_t = aabb.to(dev)
            if isinstance(grid_size, int):
                gs = torch.tensor([grid_size]*3, dtype=torch.int32, device=dev)
            else:
                gs = grid_size.to(dev)
            vs = (aabb_t[1] - aabb_t[0]) / gs.float()
            N = coords.shape[0]
            mesh_verts = (coords.float() + dual_vertices) * vs.unsqueeze(0) + aabb_t[0].unsqueeze(0)
            coords_cpu = coords.cpu()
            coord_to_idx = {}
            for i in range(N):
                c = (coords_cpu[i,0].item(), coords_cpu[i,1].item(), coords_cpu[i,2].item())
                coord_to_idx[c] = i
            EDGE_OFF = [[[0,0,0],[0,0,1],[0,1,1],[0,1,0]],
                        [[0,0,0],[1,0,0],[1,0,1],[0,0,1]],
                        [[0,0,0],[0,1,0],[1,1,0],[1,0,0]]]
            SPLIT1 = [0,1,2,0,2,3]; SPLIT2 = [0,1,3,3,1,2]
            tri_list = []
            interf_cpu = intersected_flag.cpu()
            for v in range(N):
                cx,cy,cz = coords_cpu[v,0].item(),coords_cpu[v,1].item(),coords_cpu[v,2].item()
                for axis in range(3):
                    if interf_cpu[v,axis].item() <= 0: continue
                    q = []; ok = True
                    for off in EDGE_OFF[axis]:
                        key = (cx+off[0], cy+off[1], cz+off[2])
                        if key not in coord_to_idx: ok = False; break
                        q.append(coord_to_idx[key])
                    if not ok: continue
                    sw = split_weight.cpu() if split_weight is not None else None
                    if sw is not None:
                        sw02 = sw[q[0]]*sw[q[2]]; sw13 = sw[q[1]]*sw[q[3]]
                        if sw02 > sw13: tri_list.extend([q[i] for i in SPLIT1])
                        else: tri_list.extend([q[i] for i in SPLIT2])
                    else:
                        tri_list.extend([q[i] for i in SPLIT1])
            if not tri_list:
                faces = torch.zeros((0,3), dtype=torch.int32, device=dev)
            else:
                faces = torch.tensor(tri_list, dtype=torch.int32, device=dev).reshape(-1,3)
            return mesh_verts, faces
        ovx_convert.flexible_dual_grid_to_mesh = flexible_dual_grid_to_mesh
        ovx.convert = ovx_convert
        sys.modules["o_voxel"] = ovx
        sys.modules["o_voxel.convert"] = ovx_convert

def stub_cumesh():
    import types
    if "cumesh" not in sys.modules:
        stub = types.ModuleType("cumesh")
        class _S: 
            def __init__(self, *a, **k): raise RuntimeError("cumesh stubbed")
        stub.CuMesh = _S; sys.modules["cumesh"] = stub

def stub_flex_gemm():
    import types
    if "flex_gemm" not in sys.modules:
        fg = types.ModuleType("flex_gemm")
        fg_ops = types.ModuleType("flex_gemm.ops")
        fg_gs = types.ModuleType("flex_gemm.ops.grid_sample")
        fg_sp = types.ModuleType("flex_gemm.ops.spconv")
        def _u(*a,**k): raise RuntimeError("flex_gemm stubbed")
        fg_gs.grid_sample_3d = _u; fg_sp.sparse_submanifold_conv3d = _u
        fg_ops.grid_sample = fg_gs; fg_ops.spconv = fg_sp
        fg.ops = fg_ops
        sys.modules["flex_gemm"] = fg
        sys.modules["flex_gemm.ops"] = fg_ops
        sys.modules["flex_gemm.ops.grid_sample"] = fg_gs
        sys.modules["flex_gemm.ops.spconv"] = fg_sp

def install_sdpa_attention():
    from trellis2.modules.sparse.attention import full_attn
    from trellis2.modules.sparse import VarLenTensor
    import torch.nn.functional as F
    def sdpa_varlen(q, k, v, q_seqlen, kv_seqlen):
        CHUNK = 2048; out = torch.empty_like(q); qo = ko = 0
        for ql, kl in zip(q_seqlen, kv_seqlen):
            ks = k[ko:ko+kl].transpose(0,1).unsqueeze(0)
            vs = v[ko:ko+kl].transpose(0,1).unsqueeze(0)
            for s in range(0, ql, CHUNK):
                e = min(s+CHUNK, ql)
                qs = q[qo+s:qo+e].transpose(0,1).unsqueeze(0)
                o = F.scaled_dot_product_attention(qs, ks, vs)
                out[qo+s:qo+e] = o.squeeze(0).transpose(0,1)
            qo += ql; ko += kl
        return out
    def sparse_sdpa(*args, **kwargs):
        num = len(args)+len(kwargs)
        if num == 1:
            qkv = args[0] if args else kwargs["qkv"]
            q_seqlen = [qkv.layout[i].stop-qkv.layout[i].start for i in range(qkv.shape[0])]
            q,k,v = qkv.feats.unbind(dim=1)
            out = sdpa_varlen(q,k,v,q_seqlen,q_seqlen)
            return qkv.replace(out)
        if num == 2:
            q = args[0] if args else kwargs["q"]
            kv = args[1] if len(args)>1 else kwargs["kv"]
            s = q if isinstance(q, VarLenTensor) else None
            if isinstance(q, VarLenTensor):
                q_seqlen = [q.layout[i].stop-q.layout[i].start for i in range(q.shape[0])]; qf = q.feats
            else:
                N,L = q.shape[:2]; q_seqlen=[L]*N; qf=q.reshape(N*L,*q.shape[2:])
            if isinstance(kv, VarLenTensor):
                kv_seqlen = [kv.layout[i].stop-kv.layout[i].start for i in range(kv.shape[0])]; kvf = kv.feats
            else:
                N,L = kv.shape[:2]; kv_seqlen=[L]*N; kvf=kv.reshape(N*L,*kv.shape[2:])
            k,v = kvf.unbind(dim=1)
            out = sdpa_varlen(qf,k,v,q_seqlen,kv_seqlen)
            if s is not None: return s.replace(out)
            N=len(q_seqlen); return out.reshape(N,q_seqlen[0],*out.shape[1:])
        if num == 3:
            q=args[0] if args else kwargs["q"]; k=args[1] if len(args)>1 else kwargs["k"]; v=args[2] if len(args)>2 else kwargs["v"]
            s = q if isinstance(q, VarLenTensor) else None
            if isinstance(q, VarLenTensor):
                q_seqlen=[q.layout[i].stop-q.layout[i].start for i in range(q.shape[0])]; qf=q.feats
            else:
                N,L=q.shape[:2]; q_seqlen=[L]*N; qf=q.reshape(N*L,*q.shape[2:])
            if isinstance(k, VarLenTensor):
                kv_seqlen=[k.layout[i].stop-k.layout[i].start for i in range(k.shape[0])]; kf,vf=k.feats,v.feats
            else:
                N,L=k.shape[:2]; kv_seqlen=[L]*N; kf=k.reshape(N*L,*k.shape[2:]); vf=v.reshape(N*L,*v.shape[2:])
            out = sdpa_varlen(qf,kf,vf,q_seqlen,kv_seqlen)
            if s is not None: return s.replace(out)
            N=len(q_seqlen); return out.reshape(N,q_seqlen[0],*out.shape[1:])
        raise AssertionError("bad arg count")
    full_attn.sparse_scaled_dot_product_attention = sparse_sdpa
    from trellis2.modules.sparse.attention import modules as attn_modules
    attn_modules.sparse_scaled_dot_product_attention = sparse_sdpa

def install_torch_sparse_conv():
    import math, types, torch, torch.nn as nn
    from trellis2.modules.sparse.conv import conv as conv_dispatch
    mod = types.ModuleType("trellis2.modules.sparse.conv.conv_none")
    def sparse_conv3d_init(self, in_channels, out_channels, kernel_size,
                           stride=1, dilation=1, padding=None, bias=True, indice_key=None):
        assert stride==1 and padding is None
        self.in_channels=in_channels; self.out_channels=out_channels
        ks=tuple(kernel_size) if isinstance(kernel_size,(list,tuple)) else (kernel_size,)*3
        self.kernel_size=ks; self.stride=(1,1,1)
        self.dilation=tuple(dilation) if isinstance(dilation,(list,tuple)) else (dilation,)*3
        self.weight=nn.Parameter(torch.empty(out_channels,*ks,in_channels))
        if bias: self.bias=nn.Parameter(torch.zeros(out_channels))
        else: self.register_parameter("bias",None)
        nn.init.kaiming_uniform_(self.weight,a=math.sqrt(5))
    def _coord_key(coords, spatial_shape):
        b,x,y,z = coords.unbind(-1); sx,sy,sz = spatial_shape
        return ((b.long()*sx+x.long())*sy+y.long())*sz+z.long()
    def sparse_conv3d_forward(self, x):
        coords=x.coords; feats=x.feats; n=feats.shape[0]; spatial=tuple(x.spatial_shape)
        keys=_coord_key(coords,spatial); order=torch.argsort(keys); keys_sorted=keys[order]
        Co,Kd,Kh,Kw,Ci = self.weight.shape; w=self.weight
        out=feats.new_zeros(n,Co)
        if self.bias is not None: out += self.bias.to(out.dtype)
        rd,rh,rw = Kd//2,Kh//2,Kw//2; dd,dh,dw = self.dilation
        offsets=[]
        for kd in range(Kd):
            for kh in range(Kh):
                for kw in range(Kw):
                    offsets.append((kd,kh,kw,(kd-rd)*dd,(kh-rh)*dh,(kw-rw)*dw))
        dev=feats.device; CHUNK=8192
        for kd,kh,kw,od,oh,ow in offsets:
            nc=coords.clone(); nc[:,1]+=od; nc[:,2]+=oh; nc[:,3]+=ow
            inb=((nc[:,1]>=0)&(nc[:,1]<spatial[0])&(nc[:,2]>=0)&(nc[:,2]<spatial[1])&(nc[:,3]>=0)&(nc[:,3]<spatial[2]))
            if not inb.any(): continue
            nk=_coord_key(nc,spatial)
            pos=torch.searchsorted(keys_sorted,nk).clamp(max=n-1)
            hit=inb&(keys_sorted[pos]==nk)
            if not hit.any(): continue
            src=order[pos[hit]]
            hit_global_idx = torch.where(hit)[0]
            w_sl=w[:,kd,kh,kw,:].to(feats.dtype).t()
            for s in range(0,n,CHUNK):
                e=min(s+CHUNK,n)
                mask_in_chunk = (hit_global_idx >= s) & (hit_global_idx < e)
                if not mask_in_chunk.any(): continue
                src_positions = torch.where(mask_in_chunk)[0]
                local_idx = hit_global_idx[mask_in_chunk] - s
                out[s:e][local_idx] += feats[src[src_positions]] @ w_sl
        return x.replace(out)
    def si_init(self,*a,**k): raise NotImplementedError
    def si_fwd(self,x): raise NotImplementedError
    mod.sparse_conv3d_init=sparse_conv3d_init; mod.sparse_conv3d_forward=sparse_conv3d_forward
    mod.sparse_inverse_conv3d_init=si_init; mod.sparse_inverse_conv3d_forward=si_fwd
    conv_dispatch._backends["none"]=mod
    from trellis2.modules import sparse as sp; sp.config.CONV="none"

def install_chunked_mlp():
    from trellis2.models.sc_vaes.sparse_unet_vae import SparseConvNeXtBlock3d
    MLP_CHUNK = 8192
    def _chunked_forward(self, x):
        h = self.conv(x); h = h.replace(self.norm(h.feats))
        feats = h.feats; n = feats.shape[0]
        if n <= MLP_CHUNK:
            h = h.replace(self.mlp(feats))
        else:
            outs = []
            for s in range(0, n, MLP_CHUNK):
                outs.append(self.mlp(feats[s:s+MLP_CHUNK]))
            h = h.replace(torch.cat(outs, dim=0))
        return h + x
    SparseConvNeXtBlock3d._forward = _chunked_forward


def setup_common():
    stub_o_voxel(); stub_cumesh(); stub_flex_gemm()
    install_sdpa_attention(); install_torch_sparse_conv(); install_chunked_mlp()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


# ── GLB export ───────────────────────────────────────────────────────

def export_mesh_glb(vertices, faces, path):
    import struct
    verts = vertices.cpu().numpy().astype(np.float32) if torch.is_tensor(vertices) else np.asarray(vertices, dtype=np.float32)
    tris = faces.cpu().numpy().astype(np.uint32) if torch.is_tensor(faces) else np.asarray(faces, dtype=np.uint32)
    
    # Compute vertex normals
    normals = np.zeros_like(verts)
    for t in tris:
        e1 = verts[t[1]] - verts[t[0]]
        e2 = verts[t[2]] - verts[t[0]]
        fn = np.cross(e1, e2)
        normals[t[0]] += fn; normals[t[1]] += fn; normals[t[2]] += fn
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    normals /= norms
    
    n_v = len(verts); n_f = len(tris)
    buf = b''
    pos_offset = len(buf); buf += verts.tobytes()
    norm_offset = len(buf); buf += normals.tobytes()
    while len(buf) % 4 != 0: buf += b'\x00'
    idx_offset = len(buf); buf += tris.tobytes()
    while len(buf) % 4 != 0: buf += b'\x00'
    buf_len = len(buf)
    
    vmin = verts.min(axis=0).tolist(); vmax = verts.max(axis=0).tolist()
    
    gltf = {
        "asset": {"version": "2.0", "generator": "pt_shape_decode_minimal"},
        "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1}, "indices": 2}]}],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": n_v, "type": "VEC3", "max": vmax, "min": vmin},
            {"bufferView": 1, "componentType": 5126, "count": n_v, "type": "VEC3"},
            {"bufferView": 2, "componentType": 5125, "count": n_f*3, "type": "SCALAR"},
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": pos_offset, "byteLength": n_v*12, "target": 34962},
            {"buffer": 0, "byteOffset": norm_offset, "byteLength": n_v*12, "target": 34962},
            {"buffer": 0, "byteOffset": idx_offset, "byteLength": n_f*3*4, "target": 34963},
        ],
        "buffers": [{"byteLength": buf_len}],
    }
    
    json_str = json.dumps(gltf, separators=(',',':'))
    while len(json_str) % 4 != 0: json_str += ' '
    json_bytes = json_str.encode('utf-8')
    
    glb_len = 12 + 8 + len(json_bytes) + 8 + buf_len
    with open(path, 'wb') as f:
        f.write(struct.pack('<III', 0x46546C67, 2, glb_len))
        f.write(struct.pack('<II', len(json_bytes), 0x4E4F534A))
        f.write(json_bytes)
        f.write(struct.pack('<II', buf_len, 0x004E4942))
        f.write(buf)
    
    print(f"  Exported GLB: {path} ({n_v} verts, {n_f} faces, {glb_len/1e6:.1f} MB)")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    setup_common()
    
    # 1. Load shape decoder directly from config + safetensors
    print("Loading shape decoder from config...")
    t0 = time.time()
    
    from trellis2.models.sc_vaes.fdg_vae import FlexiDualGridVaeDecoder
    from trellis2.modules.sparse import SparseTensor
    
    with open(SHAPE_DEC_CONFIG) as f:
        config = json.load(f)
    
    model_args = config['args']
    decoder = FlexiDualGridVaeDecoder(
        resolution=model_args['resolution'],
        model_channels=model_args['model_channels'],
        latent_channels=model_args['latent_channels'],
        num_blocks=model_args['num_blocks'],
        block_type=model_args['block_type'],
        up_block_type=model_args['up_block_type'],
        block_args=model_args['block_args'],
        use_fp16=model_args.get('use_fp16', True),
    )
    
    # Load weights from safetensors
    print(f"  Loading weights from {SHAPE_DEC_CKPT}...")
    state_dict = load_file(SHAPE_DEC_CKPT)
    decoder.load_state_dict(state_dict, strict=False)
    decoder.set_resolution(512)
    decoder.cpu()
    decoder.eval()
    
    t_load = time.time() - t0
    print(f"  Shape decoder loaded in {t_load:.1f}s")
    
    # 2. Load intermediate shape_slat
    print(f"Loading intermediate from {INTERMEDIATE_PATH}...")
    data = torch.load(INTERMEDIATE_PATH, map_location='cpu')
    feats = data['feats']
    coords = data['coords']
    spatial_shape = data['spatial_shape']
    layout = [slice(s, e) for s, e in data['layout_ranges']]
    
    shape_slat = SparseTensor(feats, coords=coords,
                               shape=torch.Size([len(layout), feats.shape[1]]),
                               layout=layout,
                               spatial_shape=spatial_shape)
    print(f"  Loaded {feats.shape[0]} points, channels={feats.shape[1]}")
    
    # 3. Run shape decode on CPU
    print("Running shape decode on CPU...")
    t1 = time.time()
    
    with torch.no_grad():
        meshes, subs = decoder(shape_slat, return_subs=True)
    
    t_dec = time.time() - t1
    
    n_verts = sum(m.vertices.shape[0] for m in meshes)
    n_faces = sum(m.faces.shape[0] for m in meshes)
    print(f"  Shape decode: {t_dec:.1f}s ({n_verts} verts, {n_faces} faces)")
    
    # 4. Export mesh
    if n_verts > 0:
        all_verts = torch.cat([m.vertices for m in meshes], dim=0)
        face_offset = 0
        all_faces_list = []
        for m in meshes:
            all_faces_list.append(m.faces + face_offset)
            face_offset += m.vertices.shape[0]
        all_faces = torch.cat(all_faces_list, dim=0)
        
        export_mesh_glb(all_verts, all_faces, MESH_OUTPUT)
    else:
        print("  WARNING: No vertices in mesh!")
    
    # 5. Save timing
    timing = {
        'load_s': round(t_load, 1),
        'decode_s': round(t_dec, 1),
        'n_verts': n_verts,
        'n_faces': n_faces,
    }
    with open("/tmp/pt_decode_timing.json", "w") as f:
        json.dump(timing, f, indent=2)
    print(f"  Decode timing saved to /tmp/pt_decode_timing.json")
    print(f"\n  TOTAL: load={t_load:.1f}s + decode={t_dec:.1f}s = {t_load+t_dec:.1f}s")


if __name__ == "__main__":
    main()
