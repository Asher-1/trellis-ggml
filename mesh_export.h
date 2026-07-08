#pragma once
//
// mesh_export — CUDA-free port of o_voxel.postprocess.to_glb: take the demo's
// dense per-vertex-PBR mesh and produce a portable, image-textured glTF 2.0
// binary (GLB). The reference does this on the GPU (CuMesh simplify/unwrap/BVH,
// nvdiffrast raster, flex_gemm grid_sample); here every stage is pure C++:
//
//   QEM decimate (meshoptimizer) -> UV atlas -> CPU UV-space raster +
//   nearest-dense-vertex PBR bake -> dilation inpaint -> hand-written GLB.
//
// The dual-grid geometry is machine-generated and heavily non-manifold, which
// shatters chart-based unwrappers; the default atlas is therefore a fast,
// topology-agnostic per-triangle grid (the bake samples the dense source, so
// per-texel colour stays faithful). Set T2GLB_XATLAS to use xatlas charts on
// clean meshes instead. No ggml / CUDA dependency: plain float/int arrays.

#include <cstdint>
#include <string>
#include <vector>

namespace t2glb {

struct MeshExportOptions {
    int   texture_size  = 2048;    // baked atlas width == height (texels)
    int   target_tris   = 150000;  // QEM decimation target (triangles)
    int   padding       = 2;       // xatlas chart padding (texels; T2GLB_XATLAS)
    int   dilate        = 6;       // gutter dilation passes (kills UV seams)
};

// Bake a dense per-vertex-PBR mesh into a UV-atlas-textured GLB.
//
//   verts   3*nv  vertex positions (mesh/world space, as fdg::extract emits)
//   tris    3*nt  triangle vertex indices
//   pbr     5*nv  base_color rgb, metallic, roughness  (null -> untextured grey)
//
// On success fills `out` with the GLB bytes and returns true. On failure returns
// false with a message in `err`. Not reentrant (Simplify.h uses global state):
// serialized internally by a mutex.
bool mesh_to_glb(const float * verts, int nv,
                 const int32_t * tris, int nt,
                 const float * pbr,
                 const MeshExportOptions & opt,
                 std::vector<uint8_t> & out,
                 std::string & err);

} // namespace t2glb
