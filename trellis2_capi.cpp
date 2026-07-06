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

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

void copy_err(char * err, int err_len, const std::string & msg) {
    if (!err || err_len <= 0) return;
    std::snprintf(err, (size_t) err_len, "%s", msg.c_str());
}

} // namespace

struct t2_pipeline {
    trellis2_dino_model      * dino     = nullptr;
    trellis2_ss_flow_model   * flow     = nullptr;
    trellis2_ss_dec_model    * dec      = nullptr;
    trellis2_slat_flow_model * slat     = nullptr;   // optional (fine path)
    trellis2_shape_dec_model * shapedec = nullptr;   // optional (fine path)
    std::string backend;
    bool fine = false;
};

// A generated mesh: verts (3/vertex), normals (3/vertex), tris (3/tri).
struct t2_mesh_result {
    std::vector<float> verts;
    std::vector<float> normals;
    std::vector<int>   tris;
};

extern "C" {

int t2_abi_version(void) { return T2_CAPI_ABI_VERSION; }

t2_pipeline * t2_pipeline_load(const char * dino_gguf,
                               const char * ss_flow_gguf,
                               const char * ss_dec_gguf,
                               const char * slat_flow_gguf,
                               const char * shape_dec_gguf,
                               char * err, int err_len) {
    std::string e;
    auto * p = new t2_pipeline();
    p->dino = trellis2_dino_load(dino_gguf, true, &e);
    if (!p->dino) { copy_err(err, err_len, "dino: " + e); t2_pipeline_free(p); return nullptr; }
    p->flow = trellis2_ss_flow_load(ss_flow_gguf, true, &e);
    if (!p->flow) { copy_err(err, err_len, "ss_flow: " + e); t2_pipeline_free(p); return nullptr; }
    // The 3D-conv decoders (CONV_3D and the sparse gather-GEMM) run on CPU:
    // ggml has no CUDA CONV_3D kernel, and they are a small fraction of the
    // total inference time compared with the flow DiTs.
    p->dec = trellis2_ss_dec_load(ss_dec_gguf, true, &e, "cpu");
    if (!p->dec) { copy_err(err, err_len, "ss_dec: " + e); t2_pipeline_free(p); return nullptr; }

    if (slat_flow_gguf && shape_dec_gguf && slat_flow_gguf[0] && shape_dec_gguf[0]) {
        p->slat = trellis2_slat_flow_load(slat_flow_gguf, true, &e);
        if (!p->slat) { copy_err(err, err_len, "slat_flow: " + e); t2_pipeline_free(p); return nullptr; }
        p->shapedec = trellis2_shape_dec_load(shape_dec_gguf, true, &e, "cpu");
        if (!p->shapedec) { copy_err(err, err_len, "shape_dec: " + e); t2_pipeline_free(p); return nullptr; }
        p->fine = true;
    }

    p->backend = trellis2_ss_flow_backend_name(p->flow);
    return p;
}

int t2_pipeline_is_fine(t2_pipeline * p) { return p && p->fine ? 1 : 0; }

void t2_pipeline_free(t2_pipeline * p) {
    if (!p) return;
    trellis2_dino_free(p->dino);
    trellis2_ss_flow_free(p->flow);
    trellis2_ss_dec_free(p->dec);
    trellis2_slat_flow_free(p->slat);
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
                             uint64_t seed, int steps, float guidance,
                             t2_progress_fn progress, void * user,
                             char * err, int err_len) {
    if (!p) { copy_err(err, err_len, "null pipeline"); return nullptr; }
    std::string e;

    const trellis2_dino_hparams & dhp = trellis2_dino_hparams_of(p->dino);
    const int S = 512;   // "512" pipeline type

    if (progress) progress(user, T2_STAGE_PREPROCESS, 0, 0);
    std::vector<unsigned char> rgb((size_t) S * S * 3);
    char perr[256] = {0};
    if (t2_preprocess_image_bytes(image_bytes, image_len, S, rgb.data(), perr, sizeof(perr))) {
        copy_err(err, err_len, perr);
        return nullptr;
    }

    if (progress) progress(user, T2_STAGE_DINO, 0, 0);
    trellis2_dino_cond cond;
    if (!trellis2_dino_encode_rgb(p->dino, rgb.data(), S, cond, &e)) {
        copy_err(err, err_len, "dino encode: " + e);
        return nullptr;
    }
    (void) dhp;

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
    struct cb_ctx { t2_progress_fn fn; void * user; } cbc{progress, user};
    if (progress) {
        progress(user, T2_STAGE_SS_FLOW, 0, sp.steps);
        sp.progress = [](void * u, int step, int total) {
            auto * c = (cb_ctx *) u;
            c->fn(c->user, T2_STAGE_SS_FLOW, step, total);
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

    if (p->fine) {
        // ── fine path: coarse occupancy -> voxel scaffold -> SLAT -> mesh ────
        const trellis2_slat_flow_hparams & shp = trellis2_slat_flow_hparams_of(p->slat);
        const int ss_res = shp.resolution;   // 32
        const int ratio = Rout / ss_res;     // 2 (max-pool 64 -> 32)

        // max_pool3d(occupancy>0) then argwhere -> voxel coords
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
        const int L = (int) (coords.size() / 3);
        if (L == 0) { copy_err(err, err_len, "empty voxel scaffold"); delete r; return nullptr; }

        trellis2_ss_sampler_params slp;
        if (steps > 0)     slp.steps = steps;
        if (guidance >= 0) slp.guidance_strength = guidance;
        slp.guidance_rescale = 0.5f;   // shape-SLAT sampler defaults
        slp.rescale_t = 3.0f;
        slp.seed = seed ^ 0x51a7ULL;
        slp.verbose = false;
        if (progress) {
            progress(user, T2_STAGE_SLAT_FLOW, 0, slp.steps);
            slp.progress = [](void * u, int step, int total) {
                auto * c = (cb_ctx *) u;
                c->fn(c->user, T2_STAGE_SLAT_FLOW, step, total);
            };
            slp.progress_user = &cbc;
        }

        std::vector<float> slat((size_t) L * shp.in_channels);
        if (!trellis2_slat_flow_sample(p->slat, L, coords.data(),
                                       cond.data.data(), (int) cond.tokens(), (int) cond.channels(),
                                       &slp, nullptr, /*denormalize*/ true, slat.data(), &e)) {
            copy_err(err, err_len, "slat sample: " + e);
            delete r; return nullptr;
        }

        if (progress) progress(user, T2_STAGE_SHAPE_DEC, 0, 0);
        std::vector<float> out_feats;
        std::vector<int32_t> out_coords;
        if (!trellis2_shape_dec_decode(p->shapedec, slat.data(), L, coords.data(),
                                       out_feats, out_coords, nullptr, &e)) {
            copy_err(err, err_len, "shape decode: " + e);
            delete r; return nullptr;
        }

        if (progress) progress(user, T2_STAGE_MESH, 0, 0);
        const trellis2_shape_dec_hparams & dhp2 = trellis2_shape_dec_hparams_of(p->shapedec);
        const int grid = ss_res * dhp2.upscale();   // 32 * 16 = 512
        const int nvox = (int) (out_coords.size() / 3);
        fdg::Mesh mesh = fdg::extract(out_feats.data(), out_coords.data(), nvox, grid);
        if (mesh.verts.empty()) {
            copy_err(err, err_len, "empty mesh (dual grid found no faces)");
            delete r; return nullptr;
        }
        r->verts   = std::move(mesh.verts);
        r->tris    = std::move(mesh.tris);
        r->normals = fdg::vertex_normals(fdg::Mesh{r->verts, r->tris});
    } else {
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
    }

    return r;
}

int t2_mesh_n_verts(const t2_mesh_result * r) { return r ? (int) (r->verts.size() / 3) : 0; }
int t2_mesh_n_tris (const t2_mesh_result * r) { return r ? (int) (r->tris.size()  / 3) : 0; }
const float * t2_mesh_verts  (const t2_mesh_result * r) { return r ? r->verts.data()   : nullptr; }
const float * t2_mesh_normals(const t2_mesh_result * r) { return r ? r->normals.data() : nullptr; }
const int *   t2_mesh_tris   (const t2_mesh_result * r) { return r ? r->tris.data()    : nullptr; }
void t2_mesh_free(t2_mesh_result * r) { delete r; }

} // extern "C"
