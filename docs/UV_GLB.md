# UV Textured GLB — Gap Analysis

## Current state (2026-07-31)

| Path | Output | UVs | PBR textures |
|------|--------|-----|----------------|
| **Default** `mesh_to_glb` / `t2_generate --out-glb` | UV atlas GLB | `TEXCOORD_0` | `baseColorTexture` + `metallicRoughnessTexture` |
| `T2GLB_VERTEX=1` | Legacy vertex-colour GLB | Per-vertex `COLOR_0` | None (no embedded images) |
| Upstream PyTorch `o_voxel.postprocess.to_glb` | UV atlas GLB | Yes | Baked maps (+ optional WebP) |
| `scripts/glb_upstream_bridge.py` | Same as upstream | Yes | Full CUDA stack required |

Default C++ export decimates dense meshes (`T2GLB_DECIMATION`, default 500k tris), UV-unwraps with xatlas, and bakes PBR. When CGAL is available, texels are projected from the full-resolution source mesh (upstream-style reprojection).

## Environment variables

| Variable | Default | Meaning |
|----------|---------|---------|
| `T2GLB_VERTEX` | off | Opt into legacy per-vertex GLB |
| `T2GLB_DECIMATION` | 500000 (code default); e2e uses 50000 | Max triangles before atlas bake |
| `T2GLB_TEXTURE_SIZE` | 2048 (code); e2e uses 512 | Atlas resolution |
| `T2GLB_XATLAS` | legacy | Skip decimation (pre-simplified meshes only) |

## Upstream bridge (parity validation)

```bash
# Requires cumesh + nvdiffrast + flex_gemm + o_voxel (TRELLIS.2 CUDA env)
python scripts/glb_upstream_bridge.py \
  --mesh outputs/pbr_e2e/pbr_official_T.t2mesh \
  --image assets/example_image/T.png \
  --out outputs/pbr_e2e/pbr_official_T_upstream.glb
```

## Acceptance criteria for “UV GLB done”

- [x] Default GLB with embedded `baseColorTexture` / `metallicRoughnessTexture`
- [x] `TEXCOORD_0` present
- [ ] Visual parity vs upstream on `T.png` fixture (SSIM / manual)
- [x] File size ≪ vertex-colour GLB for decimated exports
