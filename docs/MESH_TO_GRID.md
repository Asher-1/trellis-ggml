# Mesh → Dual Grid (QEF)

## Status

**Implemented** — via the vendored `o-voxel-fdg` native library
(`third_party/o-voxel-fdg/`, Eigen-only), exposed through
`examples/mesh_to_dual_grid.{h,cpp}` and validated by
tests/test_mesh_to_dual_grid.cpp against the PyTorch reference grid dump.
`examples/mesh_to_dual_grid` provides the standalone CLI
(`t2_mesh_to_grid --mesh M.obj --resolution 512 --out M.t2grid`) that feeds the
mesh-only retexture path (`run_pbr_e2e.sh`).

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

## Output

Feeds `FlexiDualGridVaeEncoder` as sparse `(vert_offset, intersected)` at resolution R.

## What we already have

| Direction | Status | File |
|-----------|--------|------|
| dual grid **fields → mesh** | Done | `examples/flexible_dual_grid.h` |
| mesh + grid sidecar → PBR texture | Done | `t2_texture`, `T2GRID01` |
| mesh → dual grid (QEF) | Done (`o-voxel-fdg`) | `examples/mesh_to_dual_grid.{h,cpp}`, `tests/test_mesh_to_dual_grid.cpp` |

## Design notes

- The port reuses the vendored `o-voxel-fdg` native implementation rather than a
  hand-written C++ QEF solver.
- QEF parameters mirror the reference call: `face_weight=1.0`,
  `boundary_weight=0.2`, `regularization_weight=1e-2`, grid `aabb` = centered
  unit cube.

## Out of scope

- Adaptive / hierarchical grids
- UV unwrapping (separate track — see `UV_GLB.md`)
