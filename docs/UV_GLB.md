# UV Textured GLB Export

How the C++ exporter turns a dense mesh (with 6-channel per-vertex PBR) into a
UV-atlas GLB that matches the structure upstream (`o_voxel.postprocess.to_glb`)
emits: one material with `baseColorTexture` (RGBA — alpha preserved) +
`metallicRoughnessTexture`, `alphaMode: BLEND`, `doubleSided: true`.

## Export paths

| Path | Output | UVs | PBR textures |
|------|--------|-----|----------------|
| **Default** `mesh_to_glb` / `t2_generate --out-glb` / `mesh2glb` | UV atlas GLB | `TEXCOORD_0` | `baseColorTexture` + `metallicRoughnessTexture` |
| `T2GLB_VERTEX=1` | Legacy full-density vertex-colour GLB | Per-vertex `COLOR_0` | None (no embedded images) |
| `T2GLB_XATLAS=1` | UV atlas GLB, xatlas chart unwrap | `TEXCOORD_0` | Same, but can be minutes-slow on non-manifold meshes (also forces past CuMesh in `TRELLIS2_CUMESH=ON` builds) |
| Upstream PyTorch `o_voxel.postprocess.to_glb` | UV atlas GLB | Yes | Baked maps (+ optional WebP) |
| `scripts/glb_upstream_bridge.py` | Same as upstream | Yes | Full CUDA stack required (cumesh + nvdiffrast + flex_gemm) |

Default pipeline inside `mesh_export.cpp`:

1. **component filter** (optional, `T2GLB_COMPONENTS`)
2. **decimation** — `T2GLB_DECIMATION` (default 500k target tris). Plain
   meshopt simplification caps at ~1.27M tris on flexible-dual-grid output
   (40% complex vertices, 4% non-manifold edges), so the exporter falls back to
   `meshopt_simplifySloppy` which reaches the target; near-degenerate slivers
   are dropped afterwards.
3. **manifoldize** — sloppy decimation leaves ~30% of edges shared by 3+
   triangles; those are split into duplicated vertices so the unwrapper sees a
   manifold mesh (xatlas charting is pathologically slow otherwise).
4. **unwrap** — default is CPU normal-cone chart clustering (`cone_cluster`
   — a dependency-free port of CuMesh's compute_charts) followed by xatlas
   per-chart parameterization/packing: the chart quality of the libtorch
   reference path with no PyTorch runtime (T.png 512: clustering 3.1 s, GLB
   bake 22 s total). In `TRELLIS2_CUMESH=ON` builds the GPU original replaces
   it automatically (same algorithm, ~1.1 s clustering). Unwrap priority is
   `T2GLB_XATLAS` → CuMesh → `cone_cluster` → `simple_unwrap`, first success
   wins. Bare-xatlas charting is minutes-slow on large meshes (measured
   >40 min historically), and `simple_unwrap` (chartless 6-bin projection)
   remains as the last-resort fallback — its fixed bins render far darker/
   noisier than chart-based unwraps.
5. **raster bake** — per-texel barycentric interpolation of the 6 PBR channels
   (base color, metallic, roughness, alpha), edge-padded, encoded as PNG.
   When CGAL is available, texels are projected from the full-resolution source
   mesh (upstream-style reprojection) instead.

## Environment variables

| Variable | Default | Meaning |
|----------|---------|---------|
| `T2GLB_VERTEX` | off | Opt into legacy per-vertex GLB |
| `T2GLB_DECIMATION` | 500000 | Max triangles before atlas bake |
| `T2GLB_DECIMATION_ERROR` | 0.2 | meshopt error budget for plain/sloppy simplification |
| `T2GLB_TEXTURE_SIZE` | 2048 (code); e2e uses 4096 | Atlas resolution |
| `T2GLB_XATLAS` | off | Use xatlas chart unwrap instead of `simple_unwrap` |
| `T2GLB_DUMP_MESH` | off | Debug: dump the decimated mesh to a path (T2MESH03) |
| `T2GLB_VERBOSE` | off | Stage log with elapsed seconds to stderr |

## CLI

```sh
./build/examples/mesh2glb in.t2mesh out.glb                 # default atlas bake
T2GLB_DECIMATION=500000 T2GLB_TEXTURE_SIZE=4096 ./build/examples/mesh2glb in.t2mesh out.glb
T2GLB_VERTEX=1 ./build/examples/mesh2glb in.t2mesh out.glb  # legacy vertex colour
```

`t2_generate --out-glb` calls the same exporter (`t2_bake_glb`, 2048 texture),
so the same environment variables apply.

## Upstream bridge (parity validation)

```bash
# Requires cumesh + nvdiffrast + flex_gemm + o_voxel (TRELLIS.2 CUDA env)
python scripts/glb_upstream_bridge.py \
  --mesh outputs/pbr_e2e/pbr_official_T.t2mesh \
  --image assets/example_image/T.png \
  --out outputs/pbr_e2e/pbr_official_T_upstream.glb
```

## Validation vs upstream (T.png fixture, 2026-08-25)

All six backend/quantization combos (CPU/CUDA/Vulkan × q8/f16) export with the
upstream material structure: `alphaMode: BLEND`, `doubleSided: true`,
`baseColorTexture` + `metallicRoughnessTexture`. Measured against
`trellis_T_upstream_q8.glb` (408,445 verts / 281,310 tris / 47.6 MB):

| Metric | Result |
|--------|--------|
| Mesh tris (decimated) | 262k–276k (within −7% of upstream) |
| bbox principal axis | z ≈ 1.0, all backends, matches upstream |
| Translucent texel share | 3.2–7.4% (upstream 5.0%) |
| File size | 22–24 MB (upstream 47.6 MB — smaller PNG, never 2× larger) |
| Rendered visual diff (CUDA q8) | ~3.9% mean absolute pixel error vs upstream |

Comparison tool: `scripts/stage_compare/compare_upstream.py OUTDIR` prints the
full geometry / texture / material / alpha table.

## Acceptance criteria for “UV GLB done”

- [x] Default GLB with embedded `baseColorTexture` / `metallicRoughnessTexture`
- [x] `TEXCOORD_0` present
- [x] `alphaMode: BLEND` + `doubleSided` (transparent regions preserved)
- [x] Export runs in seconds even on non-manifold dual-grid meshes
      (was: xatlas ComputeCharts >40 min and never finished)
- [x] File size ≪ vertex-colour GLB for decimated exports
- [ ] Visual parity vs upstream on `T.png` fixture (SSIM / manual) — close (≈4%),
      not yet gated in CI
