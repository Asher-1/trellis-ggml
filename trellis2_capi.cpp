// Flat C ABI wrapper around the trellis2 C++ API (see trellis2_capi.h).
//
// Owns the image-bytes decode (stb_image) so hosts can hand us untrusted
// uploads directly; everything past decode reuses the validated C++ pipeline:
//   decode -> preprocess -> DINOv3 -> SS-flow sample -> SS-dec -> marching cubes

#include "trellis2_capi.h"
#include "trellis2.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "marching_cubes.h"
#include "flexible_dual_grid.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

void copy_err(char * err, int err_len, const std::string & msg) {
    if (!err || err_len <= 0) return;
    std::snprintf(err, (size_t) err_len, "%s", msg.c_str());
}

// Rough peak VRAM the shape decode's transient buffers need at each tier
// (measured on the reference image: ~2.4 GB at 512³, ~6.75 GB for the 1024³
// level-3 conv output; rounded up for headroom). Mesh-dependent, so treated as
// a threshold, not an exact reservation — the free-flows fallback then gives a
// several-GB cushion if the estimate is low.
size_t decode_vram_peak(int pipeline_type) {
    const double GB = (double) (1ULL << 30);
    return (size_t) ((pipeline_type == T2_PIPE_1024 ? 7.5 : 3.0) * GB);
}

} // namespace

struct t2_pipeline {
    trellis2_dino_model      * dino     = nullptr;
    trellis2_ss_flow_model   * flow     = nullptr;
    trellis2_ss_dec_model    * dec      = nullptr;
    trellis2_slat_flow_model * slat     = nullptr;   // 512 model (fine path)
    trellis2_slat_flow_model * slat_hr  = nullptr;   // 1024 model (cascade)
    trellis2_shape_dec_model * shapedec = nullptr;   // shared by 512 + cascade
    std::string backend;
    bool fine = false;      // 512 dual-grid available
    bool cascade = false;   // 1024 cascade available
    int  flags = 0;
    bool shapedec_gpu = false;   // shape decoder placed on the GPU (VRAM permitting)
    // gguf paths, so the flow DiTs can be freed to make VRAM room for a GPU
    // decode and lazily reloaded on the next generate (see ensure_decode_vram).
    std::string ss_flow_path, slat_path, slat_hr_path;
};

// A generated mesh: verts (3/vertex), normals (3/vertex), tris (3/tri).
struct t2_mesh_result {
    std::vector<float> verts;
    std::vector<float> normals;
    std::vector<int>   tris;
};

namespace {

// Reload any flow DiT freed by a previous GPU decode (see ensure_decode_vram).
// No-op the common case where nothing was freed (pointers still set).
bool reload_flows(t2_pipeline * p, std::string & e) {
    if (!p->flow && !p->ss_flow_path.empty()) {
        p->flow = trellis2_ss_flow_load(p->ss_flow_path.c_str(), true, &e);
        if (!p->flow) return false;
    }
    if (!p->slat && !p->slat_path.empty()) {
        p->slat = trellis2_slat_flow_load(p->slat_path.c_str(), true, &e);
        if (!p->slat) return false;
    }
    if (!p->slat_hr && !p->slat_hr_path.empty()) {
        p->slat_hr = trellis2_slat_flow_load(p->slat_hr_path.c_str(), true, &e);
        if (!p->slat_hr) return false;
    }
    return true;
}

// Before a GPU shape decode, make room: if the decode's transient buffers would
// not fit in current free VRAM, free the flow DiTs (all finished by decode time)
// to reclaim their ~5-7 GB. reload_flows() brings them back on the next
// generate. No-op for a CPU decoder or when the decode already fits.
void ensure_decode_vram(t2_pipeline * p, int pipeline_type) {
    if (!p->shapedec_gpu) return;
    if (trellis2_gpu_free_vram() >= decode_vram_peak(pipeline_type)) return;
    trellis2_ss_flow_free(p->flow);      p->flow    = nullptr;
    trellis2_slat_flow_free(p->slat);    p->slat    = nullptr;
    trellis2_slat_flow_free(p->slat_hr); p->slat_hr = nullptr;
}

} // namespace

extern "C" {

int t2_abi_version(void) { return T2_CAPI_ABI_VERSION; }

t2_pipeline * t2_pipeline_load(const char * dino_gguf,
                               const char * ss_flow_gguf,
                               const char * ss_dec_gguf,
                               const char * slat_flow_gguf,
                               const char * slat_hr_flow_gguf,
                               const char * shape_dec_gguf,
                               int flags,
                               char * err, int err_len) {
    std::string e;
    auto * p = new t2_pipeline();
    p->flags = flags;
    p->dino = trellis2_dino_load(dino_gguf, true, &e);
    if (!p->dino) { copy_err(err, err_len, "dino: " + e); t2_pipeline_free(p); return nullptr; }
    // Free VRAM before the flow DiTs are loaded == the VRAM reclaimable by
    // freeing them again at decode time. Drives the shape-decoder placement.
    const size_t free_pre_flows = trellis2_gpu_free_vram();
    p->flow = trellis2_ss_flow_load(ss_flow_gguf, true, &e);
    if (!p->flow) { copy_err(err, err_len, "ss_flow: " + e); t2_pipeline_free(p); return nullptr; }
    // The SS occupancy decoder uses a genuine dense CONV_3D, for which ggml has
    // no CUDA kernel, so it stays on the CPU (it is only ~3 s / 4 % anyway).
    p->dec = trellis2_ss_dec_load(ss_dec_gguf, true, &e, "cpu");
    if (!p->dec) { copy_err(err, err_len, "ss_dec: " + e); t2_pipeline_free(p); return nullptr; }

    auto present = [](const char * s) { return s && s[0]; };

    if (present(slat_flow_gguf) && present(shape_dec_gguf)) {
        p->slat = trellis2_slat_flow_load(slat_flow_gguf, true, &e);
        if (!p->slat) { copy_err(err, err_len, "slat_flow: " + e); t2_pipeline_free(p); return nullptr; }
        const bool will_cascade = present(slat_hr_flow_gguf);

        // Remember the flow-DiT gguf paths so a GPU decode can free them for VRAM
        // and reload them next generate (ensure_decode_vram / reload_flows).
        p->ss_flow_path = ss_flow_gguf ? ss_flow_gguf : "";
        p->slat_path    = slat_flow_gguf;
        p->slat_hr_path = will_cascade ? slat_hr_flow_gguf : "";

        // Auto-place the shape (FlexiDualGrid VAE) decoder — the biggest fine-path
        // stage (~44 s CPU / 59 %). Its mask-based submanifold conv runs ~20x
        // faster on the GPU (~2 s), but the decode's transient buffers need the
        // flow DiTs' VRAM freed first (done per-request in ensure_decode_vram).
        // Put it on the GPU when the card can hold that decode once the flows are
        // freed — i.e. free_pre_flows covers the tier's decode peak plus the
        // decoder weights (~1 GB) and a margin. TRELLIS2_SHAPE_DEC_{GPU,CPU}
        // force it. On the CPU (no GPU) or too small a card it stays on the CPU,
        // so there is never a mid-generation OOM.
        bool sd_gpu;
        if      (std::getenv("TRELLIS2_SHAPE_DEC_CPU")) sd_gpu = false;
        else if (std::getenv("TRELLIS2_SHAPE_DEC_GPU")) sd_gpu = free_pre_flows > 0;
        else if (free_pre_flows == 0)                   sd_gpu = false;   // no GPU
        else {
            const size_t margin = (size_t) 3 << 29;   // ~1.5 GB (weights + slack)
            sd_gpu = free_pre_flows >= decode_vram_peak(will_cascade ? T2_PIPE_1024
                                                                     : T2_PIPE_512) + margin;
        }
        p->shapedec = trellis2_shape_dec_load(shape_dec_gguf, true, &e, sd_gpu ? nullptr : "cpu");
        if (!p->shapedec && sd_gpu) {   // unexpected GPU load OOM — fall back to CPU
            sd_gpu = false;
            p->shapedec = trellis2_shape_dec_load(shape_dec_gguf, true, &e, "cpu");
        }
        if (!p->shapedec) { copy_err(err, err_len, "shape_dec: " + e); t2_pipeline_free(p); return nullptr; }
        p->shapedec_gpu = sd_gpu;
        p->fine = true;

        // The 1024 model is optional; when present the cascade path is enabled
        // and reuses p->shapedec for both the upsample and the 1024^3 decode.
        if (will_cascade) {
            p->slat_hr = trellis2_slat_flow_load(slat_hr_flow_gguf, true, &e);
            if (!p->slat_hr) { copy_err(err, err_len, "slat_hr_flow: " + e); t2_pipeline_free(p); return nullptr; }
            p->cascade = true;
        }
    }

    p->backend = trellis2_ss_flow_backend_name(p->flow);
    return p;
}

int t2_pipeline_caps(t2_pipeline * p) {
    if (!p) return 0;
    int c = T2_CAP_COARSE;
    if (p->fine)    c |= T2_CAP_512;
    if (p->cascade) c |= T2_CAP_1024;
    return c;
}

int t2_pipeline_is_fine(t2_pipeline * p) { return p && (p->fine || p->cascade) ? 1 : 0; }

void t2_pipeline_free(t2_pipeline * p) {
    if (!p) return;
    trellis2_dino_free(p->dino);
    trellis2_ss_flow_free(p->flow);
    trellis2_ss_dec_free(p->dec);
    trellis2_slat_flow_free(p->slat);
    trellis2_slat_flow_free(p->slat_hr);
    trellis2_shape_dec_free(p->shapedec);
    delete p;
}

const char * t2_pipeline_backend(t2_pipeline * p) {
    return p ? p->backend.c_str() : "none";
}

int t2_preprocess_image_bytes(const void * image_bytes, int image_len,
                              int out_size, unsigned char * out_rgb,
                              char * err, int err_len) {
    if (!image_bytes || image_len <= 0 || out_size <= 0 || !out_rgb) {
        copy_err(err, err_len, "invalid arguments");
        return 1;
    }
    // Reject absurd dimensions before decoding (upload DoS / decompression
    // bombs): 16 MPixel is far beyond anything useful at a 512^2 target.
    int w = 0, h = 0, comp = 0;
    if (!stbi_info_from_memory((const stbi_uc *) image_bytes, image_len, &w, &h, &comp)) {
        copy_err(err, err_len, std::string("image probe failed: ") + stbi_failure_reason());
        return 1;
    }
    if (w <= 0 || h <= 0 || (int64_t) w * h > (int64_t) 16 * 1024 * 1024) {
        copy_err(err, err_len, "image dimensions out of range");
        return 1;
    }
    unsigned char * px = stbi_load_from_memory(
        (const stbi_uc *) image_bytes, image_len, &w, &h, &comp, 4);
    if (!px) {
        copy_err(err, err_len, std::string("image decode failed: ") + stbi_failure_reason());
        return 1;
    }

    // Images without an alpha channel are used as-is (no background removal
    // net): force full opacity so the crop covers the whole subject square.
    if (comp < 4) {
        for (int i = 0; i < w * h; ++i) px[(size_t) i * 4 + 3] = 255;
    }

    std::string e;
    std::vector<uint8_t> rgb;
    const bool ok = trellis2_preprocess_rgba(px, w, h, out_size, rgb, &e);
    stbi_image_free(px);
    if (!ok) {
        copy_err(err, err_len, "preprocess failed: " + e);
        return 1;
    }
    std::memcpy(out_rgb, rgb.data(), rgb.size());
    return 0;
}

t2_mesh_result * t2_generate(t2_pipeline * p,
                             const void * image_bytes, int image_len,
                             int pipeline_type,
                             uint64_t seed, int steps, float guidance,
                             t2_progress_fn progress, void * user,
                             char * err, int err_len) {
    if (!p) { copy_err(err, err_len, "null pipeline"); return nullptr; }
    std::string e;

    // Reload any flow DiT a previous GPU decode freed for VRAM (usually a no-op).
    if (!reload_flows(p, e)) {
        copy_err(err, err_len, "reload flow models: " + e);
        return nullptr;
    }

    // Resolve the requested path to what is actually loaded.
    int pt = pipeline_type;
    if (pt == T2_PIPE_AUTO) {
        pt = p->cascade ? T2_PIPE_1024 : (p->fine ? T2_PIPE_512 : T2_PIPE_COARSE);
    }
    if (pt == T2_PIPE_1024 && !p->cascade) pt = p->fine ? T2_PIPE_512 : T2_PIPE_COARSE;
    if (pt == T2_PIPE_512  && !p->fine)    pt = T2_PIPE_COARSE;

    const int S = 512;

    if (progress) progress(user, T2_STAGE_PREPROCESS, 0, 0);
    std::vector<unsigned char> rgb((size_t) S * S * 3);
    char perr[256] = {0};
    if (t2_preprocess_image_bytes(image_bytes, image_len, S, rgb.data(), perr, sizeof(perr))) {
        copy_err(err, err_len, perr);
        return nullptr;
    }

    if (progress) progress(user, T2_STAGE_DINO, 0, 0);
    trellis2_dino_cond cond;   // 512-res conditioning (SS + LR flow)
    if (!trellis2_dino_encode_rgb(p->dino, rgb.data(), S, cond, &e)) {
        copy_err(err, err_len, "dino encode: " + e);
        return nullptr;
    }

    const trellis2_ss_flow_hparams & fhp = trellis2_ss_flow_hparams_of(p->flow);
    if (cond.channels() != fhp.cond_channels) {
        copy_err(err, err_len, "cond/flow channel mismatch");
        return nullptr;
    }

    trellis2_ss_sampler_params sp;
    if (steps > 0)       sp.steps = steps;
    if (guidance >= 0)   sp.guidance_strength = guidance;
    sp.seed    = seed;
    sp.verbose = false;
    struct cb_ctx { t2_progress_fn fn; void * user; int stage; } cbc{progress, user, T2_STAGE_SS_FLOW};
    if (progress) {
        progress(user, T2_STAGE_SS_FLOW, 0, sp.steps);
        sp.progress = [](void * u, int step, int total) {
            auto * c = (cb_ctx *) u;
            c->fn(c->user, c->stage, step, total);
        };
        sp.progress_user = &cbc;
    }

    const int R = fhp.resolution;
    std::vector<float> latent((size_t) fhp.in_channels * R * R * R);
    if (!trellis2_ss_flow_sample(p->flow, cond.data.data(),
                                 (int) cond.tokens(), (int) cond.channels(),
                                 &sp, nullptr, latent.data(), &e)) {
        copy_err(err, err_len, "ss_flow sample: " + e);
        return nullptr;
    }

    if (progress) progress(user, T2_STAGE_SS_DEC, 0, 0);
    const trellis2_ss_dec_hparams & dechp = trellis2_ss_dec_hparams_of(p->dec);
    const int Rout = dechp.res_out();   // 64
    std::vector<float> occ((size_t) dechp.out_channels * Rout * Rout * Rout);
    if (!trellis2_ss_dec_decode(p->dec, latent.data(), occ.data(), &e)) {
        copy_err(err, err_len, "ss_dec decode: " + e);
        return nullptr;
    }

    auto * r = new t2_mesh_result();

    if (pt == T2_PIPE_COARSE) {
        // ── coarse path: marching cubes on the 64^3 occupancy ────────────────
        if (progress) progress(user, T2_STAGE_MESH, 0, 0);
        mc::Mesh mesh = mc::extract(occ.data(), Rout, Rout, Rout, /*iso*/ 0.0f);
        if (mesh.verts.empty()) {
            copy_err(err, err_len, "empty mesh (no occupied voxels at iso 0)");
            delete r; return nullptr;
        }
        const float inv = 1.0f / (float) Rout;
        for (size_t i = 0; i < mesh.verts.size(); ++i) mesh.verts[i] = mesh.verts[i] * inv - 0.5f;
        r->verts   = std::move(mesh.verts);
        r->normals = std::move(mesh.normals);
        r->tris    = std::move(mesh.tris);
        return r;
    }

    // ── fine / cascade: 64^3 occupancy -> 32^3 voxel scaffold ────────────────
    const trellis2_slat_flow_hparams & shp = trellis2_slat_flow_hparams_of(p->slat);
    const int ss_res = shp.resolution;   // 32
    const int ratio = Rout / ss_res;     // 2 (max-pool 64 -> 32)
    std::vector<int32_t> coords;
    for (int x = 0; x < ss_res; ++x)
    for (int y = 0; y < ss_res; ++y)
    for (int z = 0; z < ss_res; ++z) {
        bool any = false;
        for (int dx = 0; dx < ratio && !any; ++dx)
        for (int dy = 0; dy < ratio && !any; ++dy)
        for (int dz = 0; dz < ratio && !any; ++dz) {
            const int xi = x*ratio+dx, yi = y*ratio+dy, zi = z*ratio+dz;
            const size_t idx = ((size_t) xi * Rout + yi) * Rout + zi;
            if (occ[idx] > 0.0f) any = true;
        }
        if (any) { coords.push_back(x); coords.push_back(y); coords.push_back(z); }
    }
    int L = (int) (coords.size() / 3);
    if (L == 0) { copy_err(err, err_len, "empty voxel scaffold"); delete r; return nullptr; }

    // shape-SLAT sampler params (shared LR + HR)
    auto make_slp = [&](int stage, uint64_t sd) {
        trellis2_ss_sampler_params slp;
        if (steps > 0)     slp.steps = steps;
        if (guidance >= 0) slp.guidance_strength = guidance;
        slp.guidance_rescale = 0.5f;
        slp.rescale_t = 3.0f;
        slp.seed = sd;
        slp.verbose = false;
        if (progress) {
            progress(user, stage, 0, slp.steps);
            slp.progress = [](void * u, int step, int total) {
                auto * c = (cb_ctx *) u;
                c->fn(c->user, c->stage, step, total);
            };
            cbc.stage = stage;
            slp.progress_user = &cbc;
        }
        return slp;
    };

    // ── LR shape-SLAT flow (512 model, 512-res cond) ─────────────────────────
    std::vector<float> slat((size_t) L * shp.in_channels);
    {
        trellis2_ss_sampler_params slp = make_slp(T2_STAGE_SLAT_FLOW, seed ^ 0x51a7ULL);
        if (!trellis2_slat_flow_sample(p->slat, L, coords.data(),
                                       cond.data.data(), (int) cond.tokens(), (int) cond.channels(),
                                       &slp, nullptr, /*denormalize*/ true, slat.data(), &e)) {
            copy_err(err, err_len, "slat sample: " + e);
            delete r; return nullptr;
        }
    }

    const trellis2_shape_dec_hparams & dhp2 = trellis2_shape_dec_hparams_of(p->shapedec);
    std::vector<float> dec_feats;      // 7-ch decoder output to mesh
    std::vector<int32_t> dec_coords;
    int grid = 0;

    if (pt == T2_PIPE_512) {
        // ── 512 fine: decode the LR slat directly at grid 512 ────────────────
        if (progress) progress(user, T2_STAGE_SHAPE_DEC, 0, 0);
        ensure_decode_vram(p, T2_PIPE_512);   // free the flow DiTs if a GPU decode needs the room
        if (!trellis2_shape_dec_decode(p->shapedec, slat.data(), L, coords.data(),
                                       dec_feats, dec_coords, nullptr, &e)) {
            copy_err(err, err_len, "shape decode: " + e);
            delete r; return nullptr;
        }
        grid = ss_res * dhp2.upscale();   // 32 * 16 = 512
    } else {
        // ── 1024 cascade: upsample -> quantize -> HR flow -> decode grid 1024 ─
        if (progress) progress(user, T2_STAGE_UPSAMPLE, 0, 0);
        std::vector<int32_t> up_coords;   // 512^3 candidate coords
        if (!trellis2_shape_dec_upsample(p->shapedec, slat.data(), L, coords.data(),
                                         /*upsample_times*/ 4, up_coords, &e)) {
            copy_err(err, err_len, "shape upsample: " + e);
            delete r; return nullptr;
        }
        // quantize (c+0.5)/512*64 and dedup into the 64^3 HR scaffold
        const int lr_res = ss_res * dhp2.upscale();   // 512
        const int hr_grid = shp.resolution * 2;       // 64 (HR flow resolution)
        std::unordered_set<uint64_t> seen;
        std::vector<int32_t> hr_coords;
        auto key = [](int32_t a, int32_t b, int32_t c) {
            return ((uint64_t)(uint32_t)a<<40) | ((uint64_t)(uint32_t)b<<20) | (uint64_t)(uint32_t)c;
        };
        for (size_t i = 0; i < up_coords.size(); i += 3) {
            int32_t qx = (int32_t)((up_coords[i]     + 0.5f) / lr_res * hr_grid);
            int32_t qy = (int32_t)((up_coords[i + 1] + 0.5f) / lr_res * hr_grid);
            int32_t qz = (int32_t)((up_coords[i + 2] + 0.5f) / lr_res * hr_grid);
            if (seen.insert(key(qx, qy, qz)).second) {
                hr_coords.push_back(qx); hr_coords.push_back(qy); hr_coords.push_back(qz);
            }
        }
        const int Lhr = (int) (hr_coords.size() / 3);
        if (Lhr == 0) { copy_err(err, err_len, "empty HR scaffold"); delete r; return nullptr; }

        // 1024-res conditioning (separate preprocess + encode at 1024)
        std::vector<unsigned char> rgb1024((size_t) 1024 * 1024 * 3);
        if (t2_preprocess_image_bytes(image_bytes, image_len, 1024, rgb1024.data(), perr, sizeof(perr))) {
            copy_err(err, err_len, perr); delete r; return nullptr;
        }
        trellis2_dino_cond cond1024;
        if (!trellis2_dino_encode_rgb(p->dino, rgb1024.data(), 1024, cond1024, &e)) {
            copy_err(err, err_len, "dino encode 1024: " + e); delete r; return nullptr;
        }

        // HR shape-SLAT flow (1024 model)
        const trellis2_slat_flow_hparams & shp_hr = trellis2_slat_flow_hparams_of(p->slat_hr);
        std::vector<float> hr_slat((size_t) Lhr * shp_hr.in_channels);
        trellis2_ss_sampler_params slp = make_slp(T2_STAGE_SLAT_FLOW_HR, seed ^ 0x1024ULL);
        if (!trellis2_slat_flow_sample(p->slat_hr, Lhr, hr_coords.data(),
                                       cond1024.data.data(), (int) cond1024.tokens(), (int) cond1024.channels(),
                                       &slp, nullptr, /*denormalize*/ true, hr_slat.data(), &e)) {
            copy_err(err, err_len, "HR slat sample: " + e);
            delete r; return nullptr;
        }

        if (progress) progress(user, T2_STAGE_SHAPE_DEC_HR, 0, 0);
        ensure_decode_vram(p, T2_PIPE_1024);   // free the flow DiTs (all done) for the 1024³ decode
        if (!trellis2_shape_dec_decode(p->shapedec, hr_slat.data(), Lhr, hr_coords.data(),
                                       dec_feats, dec_coords, nullptr, &e)) {
            copy_err(err, err_len, "HR shape decode: " + e);
            delete r; return nullptr;
        }
        grid = hr_grid * dhp2.upscale();   // 64 * 16 = 1024
    }

    // ── mesh extraction (shared) ─────────────────────────────────────────────
    if (progress) progress(user, T2_STAGE_MESH, 0, 0);
    const int nvox = (int) (dec_coords.size() / 3);
    fdg::Mesh mesh = fdg::extract(dec_feats.data(), dec_coords.data(), nvox, grid);
    if (mesh.verts.empty()) {
        copy_err(err, err_len, "empty mesh (dual grid found no faces)");
        delete r; return nullptr;
    }
    r->verts   = std::move(mesh.verts);
    r->tris    = std::move(mesh.tris);
    r->normals = fdg::vertex_normals(fdg::Mesh{r->verts, r->tris});
    return r;
}

int t2_mesh_n_verts(const t2_mesh_result * r) { return r ? (int) (r->verts.size() / 3) : 0; }
int t2_mesh_n_tris (const t2_mesh_result * r) { return r ? (int) (r->tris.size()  / 3) : 0; }
const float * t2_mesh_verts  (const t2_mesh_result * r) { return r ? r->verts.data()   : nullptr; }
const float * t2_mesh_normals(const t2_mesh_result * r) { return r ? r->normals.data() : nullptr; }
const int *   t2_mesh_tris   (const t2_mesh_result * r) { return r ? r->tris.data()    : nullptr; }
void t2_mesh_free(t2_mesh_result * r) { delete r; }

} // extern "C"
