#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

/*
** ── DLL Export Decoration ────────────────────────────────────────────────
**
** TRELLIS2_API mirrors LLAMA_API / SD_API / SAM3_API: tag every public
** function so MSVC emits the correct __declspec when trellis2 is built /
** consumed as a shared library. For static builds the macro expands to
** nothing and there is no ABI surface.
*/
#ifdef TRELLIS2_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef TRELLIS2_BUILD
#            define TRELLIS2_API __declspec(dllexport)
#        else
#            define TRELLIS2_API __declspec(dllimport)
#        endif
#    else
#        define TRELLIS2_API __attribute__ ((visibility ("default")))
#    endif
#else
#    define TRELLIS2_API
#endif

/*
** ── Version ─────────────────────────────────────────────────────────────
*/

#define TRELLIS2_VERSION_MAJOR 0
#define TRELLIS2_VERSION_MINOR 1
#define TRELLIS2_VERSION_PATCH 0
#define TRELLIS2_VERSION       "0.1.0"

/*****************************************************************************
** Public Data Types
*****************************************************************************/

/*
** ── DINOv3 conditioning tensor ───────────────────────────────────────────
**
** The exact image-conditioning that TRELLIS.2 stage 1 (sparse-structure flow
** DiT) consumes as cross-attention K/V. It is the full DINOv3 ViT-L/16 token
** sequence of the LAST transformer layer with an AFFINE-FREE LayerNorm on top
** (NOT HF's learned-final-norm `last_hidden_state`).
**
** Canonical shape is [1, N, C] = [1, 1029, 1024], where the 1029 tokens are
** 1 CLS + 4 register + 1024 patch (32x32 @ 512px, patch-16). Cross-attention
** is permutation-invariant over these tokens (no positional embedding on the
** cond side), so token order is irrelevant — but the full set must be present.
**
** neg_cond (for classifier-free guidance) is exactly zeros_like(cond) and is
** therefore never stored on disk.
*/
struct trellis2_dino_cond {
    // Row-major (C-contiguous) shape. ndim is normally 3 -> {1, N, C}.
    std::vector<int64_t> shape;
    // prod(shape) float32 values, C-contiguous in `shape` order.
    std::vector<float>   data;

    uint32_t format_version = 0; // version field from the .dinodata header

    int64_t ndim()     const { return (int64_t) shape.size(); }
    int64_t batch()    const { return shape.size() == 3 ? shape[0] : 1; }
    int64_t tokens()   const { return shape.size() == 3 ? shape[1] : (shape.empty() ? 0 : shape[0]); }
    int64_t channels() const { return shape.empty() ? 0 : shape.back(); }
    size_t  count()    const { return data.size(); }
    bool    empty()    const { return data.empty(); }
};

/*
** Cheap fingerprints over the whole payload — these mirror exactly the values
** written into the `<stem>.dino.txt` JSON sidecars by the Python side, so they
** can be used to verify that this loader read the file bit-for-bit, and later
** that an ONNX-derived cond matches the reference.
*/
struct trellis2_dino_fingerprint {
    float  vmin  = 0.0f;
    float  vmax  = 0.0f;
    double mean  = 0.0;
    double sum   = 0.0;
    double l2    = 0.0;
    size_t count = 0;
};

/*
** ── Sparse-structure flow DiT (stage 1) ──────────────────────────────────
**
** Hyperparameters of the SparseStructureFlowModel, read from the GGUF KV
** store (`trellis2.ss_flow.*`). For the shipped 1.3B checkpoint these are:
** resolution 16, in/out 8, model_channels 1536, cond_channels 1024,
** num_blocks 30, num_heads 12, mlp_ratio 5.3334, rope PE, share_mod + QK-RMS
** norm on both self- and cross-attention.
*/
struct trellis2_ss_flow_hparams {
    int32_t resolution     = 0;
    int32_t in_channels    = 0;
    int32_t out_channels   = 0;
    int32_t model_channels = 0;
    int32_t cond_channels  = 0;
    int32_t num_blocks     = 0;
    int32_t num_heads      = 0;
    float   mlp_ratio      = 0.0f;
    int32_t share_mod         = 0; // bool
    int32_t qk_rms_norm       = 0; // bool
    int32_t qk_rms_norm_cross = 0; // bool
    float   rope_freq_min  = 1.0f;
    float   rope_freq_base = 10000.0f;
    char    pe_mode[16]    = {0};   // "rope" or "ape"
    int32_t file_type      = 0;     // 0=f32, 1=f16, 2=bf16

    int32_t head_dim() const { return num_heads ? model_channels / num_heads : 0; }
};

// Opaque handle to a loaded SS-flow model (weights + metadata).
struct trellis2_ss_flow_model;

// Lightweight description of one weight tensor, for inspection.
struct trellis2_tensor_info {
    std::string name;
    int         n_dims = 0;
    int64_t     ne[4]  = {1, 1, 1, 1}; // ggml order (ne[0] is fastest-moving)
    int         ggml_type = 0;
    std::string type_name;
    size_t      n_bytes = 0;
};

/*****************************************************************************
** Public API – Version
*****************************************************************************/

TRELLIS2_API const char * trellis2_version(void);

/*****************************************************************************
** Public API – Sparse-structure flow DiT (stage 1)
*****************************************************************************/

// Load the SS-flow DiT from a GGUF produced by convert_ss_flow_to_gguf.py.
// If `load_tensors` is true the weight data is read into host memory; if false
// only the metadata + tensor descriptors are parsed (fast, for inspection).
// Returns nullptr on failure and (if `error`) fills it with a reason.
TRELLIS2_API trellis2_ss_flow_model *
trellis2_ss_flow_load(const std::string & path,
                      bool load_tensors = true,
                      std::string * error = nullptr);

TRELLIS2_API void trellis2_ss_flow_free(trellis2_ss_flow_model * m);

// Name of the compute backend the model was loaded onto (e.g. the GPU device
// description, or "CPU"). Returns "none" if loaded metadata-only.
TRELLIS2_API const char * trellis2_ss_flow_backend_name(const trellis2_ss_flow_model * m);

TRELLIS2_API const trellis2_ss_flow_hparams &
trellis2_ss_flow_hparams_of(const trellis2_ss_flow_model * m);

TRELLIS2_API int  trellis2_ss_flow_n_tensors(const trellis2_ss_flow_model * m);
TRELLIS2_API bool trellis2_ss_flow_get_tensor_info(const trellis2_ss_flow_model * m,
                                                   int i, trellis2_tensor_info & out);
// True iff a tensor with this exact name is present.
TRELLIS2_API bool trellis2_ss_flow_has_tensor(const trellis2_ss_flow_model * m,
                                              const std::string & name);

// Flow-Euler sampler parameters for the sparse-structure stage. Defaults match
// the microsoft/TRELLIS.2-4B pipeline.json (FlowEulerGuidanceIntervalSampler).
struct trellis2_ss_sampler_params {
    int      steps                 = 12;
    float    guidance_strength     = 7.5f;
    float    guidance_rescale      = 0.7f;
    float    guidance_interval_min = 0.6f;
    float    guidance_interval_max = 1.0f;
    float    rescale_t             = 5.0f;
    float    sigma_min             = 1e-5f;
    uint64_t seed                  = 0;     // used only when noise is generated internally
    bool     verbose               = true;  // print per-step progress to stderr
};

// Run one forward pass of the SS-flow DiT (CPU backend), i.e. the velocity
// prediction v = model(x, t, cond). The model must have been loaded with
// load_tensors=true.
//
//   x    : [in_channels * resolution^3] floats, channel-major
//          (x[c*R^3 + n], n = i*R^2 + j*R + k over the R^3 grid) — exactly the
//          flattened latent the Python forward sees as [1, C, R, R, R].
//   t    : scalar timestep, fed verbatim to the sinusoidal timestep embedding.
//   cond : [cond_tokens * cond_channels] floats, token-major (the .dinodata
//          layout). Pass all-zero data for the CFG negative branch.
//   out  : [out_channels * resolution^3] floats, channel-major (same as x).
//
// Returns false on failure and (if `error`) fills it with a reason.
TRELLIS2_API bool
trellis2_ss_flow_forward(trellis2_ss_flow_model * m,
                         const float * x, float t,
                         const float * cond, int cond_tokens, int cond_channels,
                         float * out, std::string * error = nullptr);

// Run the full flow-Euler sampling loop (classifier-free guidance with interval
// + rescale) to produce the stage-1 sparse-structure latent z_s.
//
//   cond       : [cond_tokens * cond_channels] DINOv3 tokens (the .dinodata
//                layout). The negative branch uses zeros_like(cond) internally.
//   params     : sampler settings; nullptr uses the pipeline defaults above.
//   noise      : optional [in_channels * resolution^3] initial noise
//                (channel-major). If nullptr, standard-normal noise is generated
//                from params->seed.
//   out_latent : [in_channels * resolution^3] floats, channel-major — the z_s
//                latent to feed into the SS decoder.
//
// Returns false on failure and (if `error`) fills it with a reason.
TRELLIS2_API bool
trellis2_ss_flow_sample(trellis2_ss_flow_model * m,
                        const float * cond, int cond_tokens, int cond_channels,
                        const trellis2_ss_sampler_params * params,
                        const float * noise,
                        float * out_latent, std::string * error = nullptr);

/*****************************************************************************
** Public API – Sparse-structure decoder (stage 1)
**
** D_S in the paper: a dense 3D-conv ResNet (SparseStructureDecoder) that turns
** the sparse-structure latent z_s ([latent_channels, R_in^3]) into an occupancy
** logit grid ([out_channels, R_out^3]). For the shipped ss_dec_conv3d_16l8
** checkpoint: latent_channels 8, out_channels 1, channels [512,128,32],
** num_res_blocks 2, num_res_blocks_middle 2, "layer" (channel-LayerNorm) norm.
** Two pixel-shuffle upsamples take R_in=16 -> 32 -> R_out=64. The coarse voxel
** scaffold is `logit > 0`.
*****************************************************************************/

struct trellis2_ss_dec_hparams {
    int32_t out_channels         = 0;
    int32_t latent_channels      = 0;
    int32_t num_res_blocks       = 0;
    int32_t num_res_blocks_middle= 0;
    int32_t n_levels             = 0;
    int32_t channels[8]          = {0};
    float   norm_eps             = 1e-5f;
    int32_t file_type            = 0;   // 0=f32, 1=f16
    char    norm_type[16]        = {0}; // "layer"

    // The decoder upsamples by 2 per level transition (n_levels-1 of them).
    int32_t res_in()  const { return 16; }
    int32_t upscale() const { int u = 1; for (int i = 1; i < n_levels; ++i) u *= 2; return u; }
    int32_t res_out() const { return res_in() * upscale(); }
};

// Opaque handle to a loaded SS decoder (weights + metadata).
struct trellis2_ss_dec_model;

// Load the SS decoder from a GGUF produced by convert_ss_dec_to_gguf.py.
TRELLIS2_API trellis2_ss_dec_model *
trellis2_ss_dec_load(const std::string & path,
                     bool load_tensors = true,
                     std::string * error = nullptr);

TRELLIS2_API void trellis2_ss_dec_free(trellis2_ss_dec_model * m);

TRELLIS2_API const char * trellis2_ss_dec_backend_name(const trellis2_ss_dec_model * m);

TRELLIS2_API const trellis2_ss_dec_hparams &
trellis2_ss_dec_hparams_of(const trellis2_ss_dec_model * m);

TRELLIS2_API int  trellis2_ss_dec_n_tensors(const trellis2_ss_dec_model * m);
TRELLIS2_API bool trellis2_ss_dec_get_tensor_info(const trellis2_ss_dec_model * m,
                                                  int i, trellis2_tensor_info & out);

// Decode a sparse-structure latent z_s into an occupancy logit grid.
//
//   latent : [latent_channels * res_in^3] floats, channel-major
//            (latent[c*R^3 + n], n = i*R^2 + j*R + k) — exactly the z_s that
//            trellis2_ss_flow_sample() produces ([1, C, R, R, R] flattened).
//   out    : [out_channels * res_out^3] floats, channel-major. For the shipped
//            checkpoint out_channels=1 and res_out=64, so this is a 64^3 grid of
//            occupancy logits; the voxel scaffold is `out[n] > 0`.
//
// Returns false on failure and (if `error`) fills it with a reason.
TRELLIS2_API bool
trellis2_ss_dec_decode(trellis2_ss_dec_model * m,
                       const float * latent,
                       float * out, std::string * error = nullptr);

/*****************************************************************************
** Public API – DINOv3 conditioning (.dinodata)
*****************************************************************************/

// Load a .dinodata file produced by dump_dinodata.py.
// Returns true on success; on failure returns false and (if `error` != nullptr)
// fills it with a human-readable reason. On success `out` is fully populated.
TRELLIS2_API bool trellis2_load_dinodata(const std::string & path,
                                         trellis2_dino_cond & out,
                                         std::string * error = nullptr);

// Compute fingerprints over a loaded cond (min/max/mean/sum/l2/count).
TRELLIS2_API trellis2_dino_fingerprint
trellis2_dino_fingerprints(const trellis2_dino_cond & cond);
