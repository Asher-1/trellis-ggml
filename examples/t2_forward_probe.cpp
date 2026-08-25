// t2_forward_probe — run a single SS-flow DiT forward on externally supplied
// x/t/cond and dump the velocity output, so single-step numerics can be
// compared across backends.
//
//   usage: t2_forward_probe <model.gguf> <x.bin> <t_ms> <cond.bin> <n_tokens> <out.bin>
//
//   x.bin    : [in_channels * resolution^3] floats
//   cond.bin : [cond_channels * n_tokens] floats
//   t_ms     : timestep in milliseconds (as the forward expects: 1000*t)
//
#include "trellis2.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static bool read_bin(const char * path, std::vector<float> & out) {
    FILE * f = std::fopen(path, "rb");
    if (!f) return false;
    std::fseek(f, 0, SEEK_END);
    long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (n <= 0 || n % 4 != 0) { std::fclose(f); return false; }
    out.resize((size_t) n / 4);
    const size_t got = std::fread(out.data(), 4, out.size(), f);
    std::fclose(f);
    return got == out.size();
}

static bool dump_bin(const char * path, const float * p, size_t n) {
    FILE * f = std::fopen(path, "wb");
    if (!f) return false;
    const bool ok = std::fwrite(p, 4, n, f) == n;
    std::fclose(f);
    return ok;
}

int main(int argc, char ** argv) {
    if (argc < 7) {
        std::fprintf(stderr, "usage: %s <model.gguf> <x.bin> <t_ms> <cond.bin> <n_tokens> <out.bin>\n", argv[0]);
        return 2;
    }
    const std::string model_path = argv[1];
    const float t_ms = (float) std::atof(argv[3]);
    const int n_tokens = std::atoi(argv[5]);

    std::vector<float> x, cond;
    if (!read_bin(argv[2], x) || !read_bin(argv[4], cond)) {
        std::fprintf(stderr, "read input failed\n");
        return 1;
    }

    std::string err;
    trellis2_ss_flow_model * m = trellis2_ss_flow_load(model_path, true, &err);
    if (!m) { std::fprintf(stderr, "load error: %s\n", err.c_str()); return 1; }
    std::fprintf(stderr, "backend: %s\n", trellis2_ss_flow_backend_name(m));

    const trellis2_ss_flow_hparams & hp = trellis2_ss_flow_hparams_of(m);
    const size_t N = (size_t) hp.resolution * hp.resolution * hp.resolution;
    const size_t n = (size_t) hp.in_channels * N;
    if (x.size() != n || cond.size() != (size_t) n_tokens * hp.cond_channels) {
        std::fprintf(stderr, "shape mismatch: x=%zu (want %zu), cond=%zu (want %zu)\n",
                     x.size(), n, cond.size(), (size_t) n_tokens * hp.cond_channels);
        trellis2_ss_flow_free(m);
        return 1;
    }

    std::vector<float> out(n, 0.0f);
    if (!trellis2_ss_flow_forward(m, x.data(), t_ms, cond.data(), n_tokens, hp.cond_channels, out.data(), &err)) {
        std::fprintf(stderr, "forward error: %s\n", err.c_str());
        trellis2_ss_flow_free(m);
        return 1;
    }
    trellis2_ss_flow_free(m);

    if (!dump_bin(argv[6], out.data(), out.size())) {
        std::fprintf(stderr, "write %s failed\n", argv[6]);
        return 1;
    }
    std::fprintf(stderr, "wrote %s (%zu floats)\n", argv[6], out.size());
    return 0;
}
