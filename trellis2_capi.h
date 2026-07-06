/*
** trellis2_capi.h — flat C ABI for embedding trellis2 in non-C++ hosts
** (the Go demo server dlopens libtrellis2.so and binds these by name).
**
** ABI version bumps whenever any signature or struct layout here changes.
*/
#pragma once

#include <stdint.h>

#ifdef TRELLIS2_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef TRELLIS2_BUILD
#            define TRELLIS2_CAPI __declspec(dllexport)
#        else
#            define TRELLIS2_CAPI __declspec(dllimport)
#        endif
#    else
#        define TRELLIS2_CAPI __attribute__ ((visibility ("default")))
#    endif
#else
#    define TRELLIS2_CAPI
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define T2_CAPI_ABI_VERSION 1

TRELLIS2_CAPI int t2_abi_version(void);

/* Pipeline stages reported by the progress callback. */
enum t2_stage {
    T2_STAGE_PREPROCESS = 0,   /* image decode + crop/premultiply/resize */
    T2_STAGE_DINO       = 1,   /* conditioning encoder                   */
    T2_STAGE_SS_FLOW    = 2,   /* sparse-structure flow sampling (steps) */
    T2_STAGE_SS_DEC     = 3,   /* occupancy decoder -> voxel scaffold    */
    T2_STAGE_SLAT_FLOW  = 4,   /* shape-SLAT flow sampling (steps)       */
    T2_STAGE_SHAPE_DEC  = 5,   /* shape decoder -> dual-grid fields      */
    T2_STAGE_MESH       = 6    /* mesh extraction                        */
};

/* step/total are meaningful for T2_STAGE_SS_FLOW; other stages send 0/0 at
** entry. Called from the generating thread. */
typedef void (*t2_progress_fn)(void * user, int stage, int step, int total);

typedef struct t2_pipeline    t2_pipeline;
typedef struct t2_mesh_result t2_mesh_result;

/* Load the pipeline models. The shape-SLAT models (slat_flow_gguf,
** shape_dec_gguf) are optional: pass NULL for either to run the coarse path
** only (marching cubes on the 64^3 occupancy). When both are present,
** t2_generate produces the fine dual-grid mesh instead. On failure returns
** NULL and, if err != NULL, writes a NUL-terminated reason into err. */
TRELLIS2_CAPI t2_pipeline * t2_pipeline_load(const char * dino_gguf,
                                             const char * ss_flow_gguf,
                                             const char * ss_dec_gguf,
                                             const char * slat_flow_gguf,
                                             const char * shape_dec_gguf,
                                             char * err, int err_len);

/* Which mesh path the loaded pipeline will use: 1 = fine dual-grid (SLAT
** models present), 0 = coarse marching-cubes preview. */
TRELLIS2_CAPI int t2_pipeline_is_fine(t2_pipeline * p);
TRELLIS2_CAPI void          t2_pipeline_free(t2_pipeline * p);
TRELLIS2_CAPI const char *  t2_pipeline_backend(t2_pipeline * p);

/* image bytes (PNG/JPEG/...; anything stb_image decodes) -> triangle mesh.
** steps <= 0 and guidance < 0 select the pipeline defaults (12 / 7.5).
** NOT thread-safe per pipeline: serialize calls on one t2_pipeline. */
TRELLIS2_CAPI t2_mesh_result * t2_generate(t2_pipeline * p,
                                           const void * image_bytes, int image_len,
                                           uint64_t seed, int steps, float guidance,
                                           t2_progress_fn progress, void * user,
                                           char * err, int err_len);

/* Mesh accessors. Vertices are in a centered unit cube ([-0.5, 0.5]^3, same
** axes as the voxel grid); normals are per-vertex unit vectors. Buffers stay
** valid until t2_mesh_free. */
TRELLIS2_CAPI int           t2_mesh_n_verts(const t2_mesh_result * r);
TRELLIS2_CAPI int           t2_mesh_n_tris (const t2_mesh_result * r);
TRELLIS2_CAPI const float * t2_mesh_verts  (const t2_mesh_result * r); /* 3*n_verts   */
TRELLIS2_CAPI const float * t2_mesh_normals(const t2_mesh_result * r); /* 3*n_verts   */
TRELLIS2_CAPI const int *   t2_mesh_tris   (const t2_mesh_result * r); /* 3*n_tris    */
TRELLIS2_CAPI void          t2_mesh_free   (t2_mesh_result * r);

/* Image decode + TRELLIS.2 preprocessing only (no models). out_rgb must hold
** out_size*out_size*3 bytes. Returns 0 on success, nonzero on failure (reason
** in err). This is the untrusted-input surface targeted by the fuzzers. */
TRELLIS2_CAPI int t2_preprocess_image_bytes(const void * image_bytes, int image_len,
                                            int out_size, unsigned char * out_rgb,
                                            char * err, int err_len);

#ifdef __cplusplus
}
#endif
