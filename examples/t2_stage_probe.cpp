// t2_stage_probe — dump per-stage intermediates of the coarse pipeline so
// different backends / quantizations can be compared element-wise.
//
//   usage: t2_stage_probe --models-dir DIR --input IMG --out-prefix P
//                         [--quantization q8|f16] [--device cpu|auto]
//                         [--steps N] [--seed N] [--dump-steps 0|1]
//
// Stages (same order as t2_generate's coarse path):
//   1. preprocess        -> P.rgb.bin        (512*512*3 uint8)
//   2. DINOv3 encode     -> P.cond.bin       (float, tokens*channels)
//   3. SS-flow sample    -> P.step_KK.bin    (per-step x_0 estimates, opt-in)
//                        -> P.latent.bin     (final z_s)
//   4. SS decode         -> P.occ.bin        (64^3 occupancy logits)
//
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
    std::string quant = "q8";
    std::string device = "auto";
    int steps = 2;
    uint64_t seed = 42;
    bool dump_steps = false;

    for (int i = 1; i < argc; ++i) {
        auto need = [&](const char * flag) -> const char * {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", flag); std::exit(2); }
            return argv[++i];
        };
        if (std::strcmp(argv[i], "--models-dir") == 0) models_dir = need("--models-dir");
        else if (std::strcmp(argv[i], "--input") == 0) input = need("--input");
        else if (std::strcmp(argv[i], "--out-prefix") == 0) prefix = need("--out-prefix");
        else if (std::strcmp(argv[i], "--quantization") == 0) quant = need("--quantization");
        else if (std::strcmp(argv[i], "--device") == 0) device = need("--device");
        else if (std::strcmp(argv[i], "--steps") == 0) steps = std::atoi(need("--steps"));
        else if (std::strcmp(argv[i], "--seed") == 0) seed = (uint64_t) std::strtoull(need("--seed"), nullptr, 10);
        else if (std::strcmp(argv[i], "--dump-steps") == 0) dump_steps = std::atoi(need("--dump-steps")) != 0;
        else { std::fprintf(stderr, "unknown arg: %s\n", argv[i]); return 2; }
    }
    if (input.empty() || prefix.empty()) {
        std::fprintf(stderr, "required: --input --out-prefix\n");
        return 2;
    }

    // Model path helper: dino/ss_flow use the requested quant; ss_dec is q8-capable too.
    auto mp = [&](const char * name) {
        std::string s = models_dir + "/" + name;
        if (quant != "f16") {
            const std::string from = "_f16.gguf";
            auto pos = s.rfind(from);
            if (pos != std::string::npos) s.replace(pos, from.size(), "_" + quant + ".gguf");
        }
        return s;
    };

    const char * dev = device == "cpu" ? "cpu" : nullptr;   // nullptr -> env / auto
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
    dump_bin(prefix + ".rgb.bin", rgb.data(), rgb.size());
    std::printf("preprocess: wrote %s.rgb.bin (%zu bytes)\n", prefix.c_str(), rgb.size());

    // ── 2. DINOv3 encode ─────────────────────────────────────────────────────
    trellis2_dino_model * dino = trellis2_dino_load(mp("dino_f16.gguf"), true, &err, dev);
    if (!dino) { std::fprintf(stderr, "dino load: %s\n", err.c_str()); return 1; }
    std::printf("dino backend: %s\n", trellis2_dino_backend_name(dino));
    trellis2_dino_cond cond;
    if (!trellis2_dino_encode_rgb(dino, rgb.data(), S, cond, &err)) {
        std::fprintf(stderr, "dino encode: %s\n", err.c_str()); return 1;
    }
    trellis2_dino_free(dino);
    dump_bin(prefix + ".cond.bin", cond.data.data(), cond.data.size() * sizeof(float));
    std::printf("dino: cond tokens=%lld channels=%lld -> %s.cond.bin\n",
                (long long) cond.tokens(), (long long) cond.channels(), prefix.c_str());

    // ── 3. SS-flow sample ────────────────────────────────────────────────────
    trellis2_ss_flow_model * flow = trellis2_ss_flow_load(mp("ss_flow_f16.gguf"), true, &err, dev);
    if (!flow) { std::fprintf(stderr, "ss_flow load: %s\n", err.c_str()); return 1; }
    std::printf("ss_flow backend: %s\n", trellis2_ss_flow_backend_name(flow));
    const trellis2_ss_flow_hparams & fhp = trellis2_ss_flow_hparams_of(flow);
    const int R = fhp.resolution;
    std::vector<float> latent((size_t) fhp.in_channels * R * R * R);

    trellis2_ss_sampler_params sp;
    sp.steps = steps;
    sp.guidance_strength = 7.5f;
    sp.seed = seed;
    sp.verbose = false;
    struct step_ctx { std::string prefix; int channels; int res; bool on; };
    step_ctx sc{prefix, fhp.in_channels, R, dump_steps};
    sp.preview = [](void * u, int step, int total, const float * lat, int n) {
        auto * c = (step_ctx *) u;
        if (!c->on) return;
        char name[256];
        std::snprintf(name, sizeof name, "%s.step_%02d.bin", c->prefix.c_str(), step);
        dump_bin(name, lat, (size_t) n * sizeof(float));
        std::printf("ss_flow: step %d/%d x0 -> %s\n", step, total, name);
    };
    sp.preview_user = &sc;

    // Reproduce the sampler's internal initial noise (same RNG/order) so it
    // can be dumped for downstream single-step forward probes.
    std::vector<float> noise((size_t) fhp.in_channels * R * R * R);
    {
        std::mt19937_64 rng(seed);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        for (size_t i = 0; i < noise.size(); ++i) noise[i] = nd(rng);
    }
    dump_bin(prefix + ".noise.bin", noise.data(), noise.size() * sizeof(float));
    std::printf("ss_flow: initial noise -> %s.noise.bin\n", prefix.c_str());

    if (!trellis2_ss_flow_sample(flow, cond.data.data(), (int) cond.tokens(), (int) cond.channels(),
                                 &sp, noise.data(), latent.data(), &err)) {
        std::fprintf(stderr, "ss_flow sample: %s\n", err.c_str()); return 1;
    }
    trellis2_ss_flow_free(flow);
    dump_bin(prefix + ".latent.bin", latent.data(), latent.size() * sizeof(float));
    std::printf("ss_flow: final latent %zu floats -> %s.latent.bin\n", latent.size(), prefix.c_str());

    // ── 4. SS decode ─────────────────────────────────────────────────────────
    // coarse path runs the decoder on CPU (conv3d has no GPU kernel here)
    trellis2_ss_dec_model * dec = trellis2_ss_dec_load(mp("ss_dec_f16.gguf"), true, &err, "cpu");
    if (!dec) { std::fprintf(stderr, "ss_dec load: %s\n", err.c_str()); return 1; }
    std::printf("ss_dec backend: %s\n", trellis2_ss_dec_backend_name(dec));
    const trellis2_ss_dec_hparams & dechp = trellis2_ss_dec_hparams_of(dec);
    const int Rout = dechp.res_out();
    std::vector<float> occ((size_t) dechp.out_channels * Rout * Rout * Rout);
    if (!trellis2_ss_dec_decode(dec, latent.data(), occ.data(), &err)) {
        std::fprintf(stderr, "ss_dec decode: %s\n", err.c_str()); return 1;
    }
    trellis2_ss_dec_free(dec);
    dump_bin(prefix + ".occ.bin", occ.data(), occ.size() * sizeof(float));
    std::printf("ss_dec: occ %zu floats (res %d) -> %s.occ.bin\n", occ.size(), Rout, prefix.c_str());

    return 0;
}
