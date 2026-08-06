# Mesh → Dual Grid (QEF) Port Plan

## Problem

Standalone `t2_texture` needs a **decoded dual grid** (`T2GRID01` sidecar: 7-channel feats + coords).
Today this sidecar is produced only by the fine-path shape decoder inside `t2_generate`.
Arbitrary PLY/OBJ meshes cannot be retextured without upstream **o-voxel** `mesh_to_flexible_dual_grid`.

## Reference (PyTorch / o-voxel)

From `scripts/dump_texture_reference.py`:

```python
voxel_indices, dual_vertices, intersected = o_voxel.convert.mesh_to_flexible_dual_grid(
    vertices, faces, grid_size=R,
    aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
    face_weight=1.0, boundary_weight=0.2, regularization_weight=1e-2,
)
enc_vert_feats = dual_vertices * R - voxel_indices  # offset in voxel units
```

Output feeds `FlexiDualGridVaeEncoder` as sparse `(vert_offset, intersected)` at resolution R.

## Status

**Implemented** — via the vendored `o-voxel-fdg` native library
(`third_party/o-voxel-fdg/`, Eigen-only), exposed through
`examples/mesh_to_dual_grid.{h,cpp}` and validated by
`tests/test_mesh_to_dual_grid.cpp` against the PyTorch reference grid dump.

## What we already have

| Direction | Status | File |
|-----------|--------|------|
| dual grid **fields → mesh** | Done | `examples/flexible_dual_grid.h` |
| mesh + grid sidecar → PBR texture | Done | `t2_texture`, `T2GRID01` |
| mesh → dual grid (QEF) | Done (`o-voxel-fdg`) | `examples/mesh_to_dual_grid.{h,cpp}`, `tests/test_mesh_to_dual_grid.cpp` |

## Minimal port strategy

The port reuses the vendored `o-voxel-fdg` native implementation rather than a
hand-written C++ QEF solver, so the standalone `t2_mesh_to_grid --mesh M.obj
--resolution 512 --out M.t2grid` CLI feeds the mesh-only retexture path
(`run_pbr_e2e.sh`).

## Out of scope

- Adaptive / hierarchical grids
- UV unwrapping (separate track — see `UV_GLB.md`)
