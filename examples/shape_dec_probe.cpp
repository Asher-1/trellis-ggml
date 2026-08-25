// shape_dec_probe — isolate the fine-path shape decoder: sample the slat ONCE
// on CPU, then decode the SAME slat with shape_dec on two devices (e.g. cpu vs
// auto/cuda) and report the per-level subdivision divergence (L_child, subdiv
// logit diffs) plus final feats/coords, so backend differences are attributable
// to the decoder itself rather than to upstream slat/cond noise.
//
//   usage: shape_dec_probe --models-dir DIR --input IMG --out-prefix P
//                          [--steps N] [--seed N] [--slat-device cpu|auto]
//
// Writes:
//   P.cpu.out_feats.bin / P.gpu.out_feats.bin   (nvox*7 float32)
//   P.cpu.out_coords.bin / P.gpu.out_coords.bin (nvox*3 int32)
//   P.slat.bin                                  (slat latent, sampled with --slat-device)
// and prints per-level voxel counts for both backends.
#include "trellis2.h"
#include "trellis2_capi.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static bool read_file(const char * path, std::vector<uint8_t> & out) {
    FILE * f = std::fopen(path, "rb");
    if (!f) return false;
    std::fseek(f, 0, SEEK_END);
    long n = std::ftell(f);
    if (n < 0) { std::fclose(f); return false; }
    std::fseek(f, 0, SEEK_SET);
    out.resize((size_t) n);
    const size_t got = std::fread(out.data(), 1, out.size(), f);
    std::fclose(f);
    return got == out.size();
}

static bool dump_bin(const std::string & path, const void * p, size_t n) {
    FILE * f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    const bool ok = std::fwrite(p, 1, n, f) == n;
    std::fclose(f);
    return ok;
}

int main(int argc, char ** argv) {
    std::string models_dir = "models";
    std::string input, prefix;
    int steps = 12;
    uint64_t seed = 0;
    std::string slat_device = "cpu";

    for (int i = 1; i < argc; ++i) {
        auto need = [&](const char * flag) -> const char * {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", flag); std::exit(2); }
            return argv[++i];
        };
        if (std::strcmp(argv[i], "--models-dir") == 0) models_dir = need("--models-dir");
        else if (std::strcmp(argv[i], "--input") == 0) input = need("--input");
        else if (std::strcmp(argv[i], "--out-prefix") == 0) prefix = need("--out-prefix");
        else if (std::strcmp(argv[i], "--steps") == 0) steps = std::atoi(need("--steps"));
        else if (std::strcmp(argv[i], "--seed") == 0) seed = (uint64_t) std::strtoull(need("--seed"), nullptr, 10);
        else if (std::strcmp(argv[i], "--slat-device") == 0) slat_device = need("--slat-device");
        else { std::fprintf(stderr, "unknown arg: %s\n", argv[i]); return 2; }
    }
    if (input.empty() || prefix.empty()) {
        std::fprintf(stderr, "required: --input --out-prefix\n");
        return 2;
    }

    auto mp = [&](const char * name) { return models_dir + "/" + name; };
    std::string err;

    // ── 1. preprocess ────────────────────────────────────────────────────────
    std::vector<uint8_t> img;
    if (!read_file(input.c_str(), img)) { std::fprintf(stderr, "read %s failed\n", input.c_str()); return 1; }
    const int S = 512;
    std::vector<unsigned char> rgb((size_t) S * S * 3);
    char perr[256] = {0};
    if (t2_preprocess_image_bytes(img.data(), (int) img.size(), S, rgb.data(), perr, (int) sizeof perr)) {
        std::fprintf(stderr, "preprocess failed: %s\n", perr);
        return 1;
    }

    // ── 2. DINOv3 encode (CPU for determinism) ───────────────────────────────
    trellis2_dino_model * dino = trellis2_dino_load(mp("dino_f16.gguf").c_str(), true, &err, "cpu");
    if (!dino) { std::fprintf(stderr, "dino load: %s\n", err.c_str()); return 1; }
    trellis2_dino_cond cond;
    if (!trellis2_dino_encode_rgb(dino, rgb.data(), S, cond, &err)) {
        std::fprintf(stderr, "dino encode: %s\n", err.c_str()); return 1;
    }
    trellis2_dino_free(dino);

    // ── 3. SS-flow sample + SS-dec (both CPU, like the pipeline) ─────────────
    trellis2_ss_flow_model * flow = trellis2_ss_flow_load(mp("ss_flow_f16.gguf").c_str(), true, &err, "cpu");
    if (!flow) { std::fprintf(stderr, "ss_flow load: %s\n", err.c_str()); return 1; }
    const trellis2_ss_flow_hparams & fhp = trellis2_ss_flow_hparams_of(flow);
    const int R = fhp.resolution;
    std::vector<float> latent((size_t) fhp.in_channels * R * R * R);

    trellis2_ss_sampler_params sp;
    sp.steps = steps;
    sp.guidance_strength = 7.5f;
    sp.seed = seed;
    sp.verbose = false;
    std::vector<float> noise((size_t) fhp.in_channels * R * R * R);
    {
        std::mt19937_64 rng(seed);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        for (size_t i = 0; i < noise.size(); ++i) noise[i] = nd(rng);
    }
    if (!trellis2_ss_flow_sample(flow, cond.data.data(), (int) cond.tokens(), (int) cond.channels(),
                                 &sp, noise.data(), latent.data(), &err)) {
        std::fprintf(stderr, "ss_flow sample: %s\n", err.c_str()); return 1;
    }
    trellis2_ss_flow_free(flow);

    trellis2_ss_dec_model * dec = trellis2_ss_dec_load(mp("ss_dec_f16.gguf").c_str(), true, &err, "cpu");
    if (!dec) { std::fprintf(stderr, "ss_dec load: %s\n", err.c_str()); return 1; }
    const trellis2_ss_dec_hparams & dechp = trellis2_ss_dec_hparams_of(dec);
    const int Rout = dechp.res_out();
    std::vector<float> occ((size_t) dechp.out_channels * Rout * Rout * Rout);
    if (!trellis2_ss_dec_decode(dec, latent.data(), occ.data(), &err)) {
        std::fprintf(stderr, "ss_dec decode: %s\n", err.c_str()); return 1;
    }
    trellis2_ss_dec_free(dec);

    // 32^3 scaffold (max-pool 64 -> 32), same as the pipeline
    const int ss_res = 32;
    const int ratio = Rout / ss_res;
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
    std::printf("scaffold: %d voxels (32^3)\n", L);

    // ── 4. SLAT flow sample (on --slat-device) ───────────────────────────────
    const char * sdev = slat_device == "cpu" ? "cpu" : nullptr;
    trellis2_slat_flow_model * sflow = trellis2_slat_flow_load(mp("slat_flow_f16.gguf").c_str(), true, &err, sdev);
    if (!sflow) { std::fprintf(stderr, "slat_flow load: %s\n", err.c_str()); return 1; }
    std::printf("slat_flow backend: %s\n", trellis2_slat_flow_backend_name(sflow));
    const trellis2_slat_flow_hparams & shp2 = trellis2_slat_flow_hparams_of(sflow);
    std::vector<float> slat((size_t) L * shp2.in_channels);
    {
        trellis2_ss_sampler_params slp;
        slp.steps = steps;
        slp.guidance_strength = 7.5f;
        slp.guidance_rescale = 0.5f;
        slp.rescale_t = 3.0f;
        slp.seed = seed ^ 0x51a7ULL;
        slp.verbose = false;
        std::vector<float> snoise((size_t) L * shp2.in_channels);
        {
            std::mt19937_64 rng(slp.seed);
            std::normal_distribution<float> nd(0.0f, 1.0f);
            for (size_t i = 0; i < snoise.size(); ++i) snoise[i] = nd(rng);
        }
        if (!trellis2_slat_flow_sample(sflow, L, coords.data(),
                                       cond.data.data(), (int) cond.tokens(), (int) cond.channels(),
                                       &slp, snoise.data(), /*denormalize*/ true, slat.data(), &err)) {
            std::fprintf(stderr, "slat sample: %s\n", err.c_str()); return 1;
        }
    }
    trellis2_slat_flow_free(sflow);
    dump_bin(prefix + ".slat.bin", slat.data(), slat.size() * sizeof(float));
    std::printf("slat: %d voxels x %d ch -> %s.slat.bin\n", L, shp2.in_channels, prefix.c_str());

    // ── 5. shape_dec decode: same slat, two devices ──────────────────────────
    const char * devices[2] = { "cpu", nullptr };   // nullptr -> env / auto (GPU)
    const char * tags[2]    = { "cpu", "gpu" };
    for (int d = 0; d < 2; ++d) {
        trellis2_shape_dec_model * sd = trellis2_shape_dec_load(mp("shape_dec_f16.gguf").c_str(), true, &err, devices[d]);
        if (!sd) { std::fprintf(stderr, "shape_dec(%s) load: %s\n", tags[d], err.c_str()); return 1; }
        std::printf("shape_dec[%s] backend: %s\n", tags[d], trellis2_shape_dec_backend_name(sd));

        std::vector<float> out_feats;
        std::vector<int32_t> out_coords;
        std::vector<trellis2_subdiv_level> subs;
        if (!trellis2_shape_dec_decode_with_subs(sd, slat.data(), L, coords.data(),
                                                 out_feats, out_coords, subs, nullptr, &err)) {
            std::fprintf(stderr, "shape_dec(%s) decode: %s\n", tags[d], err.c_str());
            return 1;
        }
        trellis2_shape_dec_free(sd);

        std::printf("shape_dec[%s]: levels=%zu", tags[d], subs.size());
        for (size_t l = 0; l < subs.size(); ++l) {
            std::printf("  L%d=%zu", (int) l + 1, subs[l].fine_coords.size() / 3);
        }
        std::printf("  out=%zu\n", out_coords.size() / 3);

        dump_bin(prefix + "." + tags[d] + ".out_feats.bin", out_feats.data(), out_feats.size() * sizeof(float));
        dump_bin(prefix + "." + tags[d] + ".out_coords.bin", out_coords.data(), out_coords.size() * sizeof(int32_t));
    }

    return 0;
}
