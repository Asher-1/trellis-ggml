// End-to-end image -> 3D mesh via t2_generate (same path as the demo server).
//
// usage:
//   t2_generate --models-dir <dir> --input <image> --out-mesh <.t2mesh> [--out-glb <.glb>]
//               [--quality auto|coarse|512|1024] [--seed N] [--steps N]
//
#include "trellis2_capi.h"
#include "t2_grid_io.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
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

static bool write_mesh_bin(const char * path, const t2_mesh_result * mesh) {
    const int nv = t2_mesh_n_verts(mesh);
    const int nt = t2_mesh_n_tris(mesh);
    const float * verts   = t2_mesh_verts(mesh);
    const float * normals = t2_mesh_normals(mesh);
    const int   * tris    = t2_mesh_tris(mesh);
    const float * pbr     = t2_mesh_has_pbr(mesh) ? t2_mesh_pbr(mesh) : nullptr;

    FILE * f = std::fopen(path, "wb");
    if (!f) return false;
    char magic[8];
    std::memcpy(magic, pbr ? "T2MESH03" : "T2MESH01", 8);
    const uint32_t unv = (uint32_t) nv, unt = (uint32_t) nt;
    auto wr = [&](const void * p, size_t n) { return std::fwrite(p, 1, n, f) == n; };
    bool ok = wr(magic, 8) && wr(&unv, 4) && wr(&unt, 4)
           && wr(verts, (size_t) nv * 3 * sizeof(float))
           && wr(normals, (size_t) nv * 3 * sizeof(float));
    if (ok && pbr) ok = wr(pbr, (size_t) nv * 6 * sizeof(float));
    if (ok) ok = wr(tris, (size_t) nt * 3 * sizeof(int));
    std::fclose(f);
    return ok;
}

static int pipe_type(const char * q) {
    if (!q || !q[0] || std::strcmp(q, "auto") == 0) return T2_PIPE_AUTO;
    if (std::strcmp(q, "coarse") == 0) return T2_PIPE_COARSE;
    if (std::strcmp(q, "512") == 0) return T2_PIPE_512;
    if (std::strcmp(q, "1024") == 0) return T2_PIPE_1024;
    return T2_PIPE_AUTO;
}

static void on_progress(void *, int stage, int step, int total) {
    std::fprintf(stderr, "  stage %d  step %d/%d\n", stage, step, total);
}

int main(int argc, char ** argv) {
    std::string models_dir = "models";
    std::string input, out_mesh, out_glb;
    std::string quality = "auto";
    std::string quant = "f16"; // default chain: cuda/vulkan f16 (q8 saves ~nothing now, only ss_dec quantizes)
    uint64_t seed = 0;
    int steps = 12;
    int texture_steps = 12;
    bool no_texture = false;
    int background_mode = T2_BACKGROUND_AUTO;
    std::string save_grid;
    std::string rmbg_model;
    std::string rmbg_device = "auto";

    for (int i = 1; i < argc; ++i) {
        auto need = [&](const char * flag) -> const char * {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", flag);
                std::exit(2);
            }
            return argv[++i];
        };
        if (std::strcmp(argv[i], "--models-dir") == 0) models_dir = need("--models-dir");
        else if (std::strcmp(argv[i], "--quantization") == 0) quant = need("--quantization");
        else if (std::strcmp(argv[i], "--input") == 0) input = need("--input");
        else if (std::strcmp(argv[i], "--out-mesh") == 0) out_mesh = need("--out-mesh");
        else if (std::strcmp(argv[i], "--out-glb") == 0) out_glb = need("--out-glb");
        else if (std::strcmp(argv[i], "--quality") == 0) quality = need("--quality");
        else if (std::strcmp(argv[i], "--seed") == 0) seed = (uint64_t) std::strtoull(need("--seed"), nullptr, 10);
        else if (std::strcmp(argv[i], "--steps") == 0) steps = std::atoi(need("--steps"));
        else if (std::strcmp(argv[i], "--texture-steps") == 0) texture_steps = std::atoi(need("--texture-steps"));
        else if (std::strcmp(argv[i], "--no-texture") == 0) no_texture = true;
        else if (std::strcmp(argv[i], "--save-grid") == 0) save_grid = need("--save-grid");
        else if (std::strcmp(argv[i], "--rmbg") == 0) rmbg_model = need("--rmbg");
        else if (std::strcmp(argv[i], "--rmbg-device") == 0) rmbg_device = need("--rmbg-device");
        else if (std::strcmp(argv[i], "--background") == 0) {
            const char * bg = need("--background");
            if (std::strcmp(bg, "auto") == 0) background_mode = T2_BACKGROUND_AUTO;
            else if (std::strcmp(bg, "keep") == 0) background_mode = T2_BACKGROUND_KEEP;
            else if (std::strcmp(bg, "black") == 0) background_mode = T2_BACKGROUND_BLACK;
            else if (std::strcmp(bg, "white") == 0) background_mode = T2_BACKGROUND_WHITE;
            else { std::fprintf(stderr, "unknown --background %s\n", bg); return 2; }
        }
        else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            std::fprintf(stderr,
                "usage: %s --models-dir DIR --input IMAGE --out-mesh OUT.t2mesh [--out-glb OUT.glb]\n"
                "       [--quantization q8|f16|f32]  model weight precision (default: f16;\n"
                "       f32 upgrades the chaotic chain (dino/flows/ss_dec/shape_dec) to full-f32\n"
                "       weights, texture chain stays f16 — the alignment/reference mode)\n"
                "       [--quality auto|coarse|512|1024] [--seed N] [--steps N]\n"
                "       [--texture-steps N] [--no-texture] [--save-grid OUT.t2grid]\n"
                "       [--background auto|keep|black|white]\n"
                "       [--rmbg MODEL.gguf]  AI background removal (RMBG-2.0)\n"
                "       [--rmbg-device cpu|cuda|vulkan|auto]  RMBG compute device\n", argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "unknown arg: %s\n", argv[i]);
            return 2;
        }
    }
    if (input.empty() || out_mesh.empty()) {
        std::fprintf(stderr, "required: --models-dir --input --out-mesh\n");
        return 2;
    }

    // Build model path: replace _f16 with _{quant} in filename.
    // Only weights feeding the chaotic CFG samplers (dino cond, all flow
    // models) stay f16 in q8 mode: even Q8_0 cond error of ~7e-3 rel-L2 is
    // amplified into completely different voxel sets. The three VAE stages
    // are also experiment-confirmed f16-only (2026-08-30, CUDA+Vulkan e2e):
    //   shape_dec q8 -> level-0 subdivision logits all flip negative
    //     ("no children at level 0") — they live within ~Q8 weight noise
    //     of zero; tex_dec q8 -> collapsed saturated PBR material;
    //   shape_enc q8 -> its condition error is chaotically amplified by the
    //     12-step texture sampler into the same saturated collapse.
    // So q8 only ever applies to the post-sampling ss_dec (whose dense conv3d
    // weight layout [3,3,3,N] is not Q8-blockable anyway — its q8 gguf is a
    // byte clone of f16), i.e. q8 mode currently saves nothing here.
    // f32 upgrades the chaotic chain to full-f32 weights for exact mode.
    auto p = [&](const char * name) -> std::string {
        std::string s = models_dir + "/" + name;
        const std::string n(name);
        bool keep_f16;
        if (quant == "f32") {
            // texture chain has no f32 ggufs (its noise affects appearance,
            // not the chaotic voxel set); the chaotic chain goes full f32.
            keep_f16 = n.find("shape_enc") != std::string::npos
                    || n.find("tex_dec") != std::string::npos
                    || n.find("tex_slat_flow") != std::string::npos;
        } else if (quant == "q8") {
            keep_f16 = n.find("shape_dec") != std::string::npos
                    || n.find("tex_dec") != std::string::npos
                    || n.find("shape_enc") != std::string::npos
                    || n.find("dino") != std::string::npos
                    || n.find("ss_flow") != std::string::npos
                    || n.find("slat_flow") != std::string::npos;
        } else {
            keep_f16 = true;
        }
        if (!keep_f16) {
            std::string from = "_f16.gguf";
            std::string to = "_" + quant + ".gguf";
            auto pos = s.rfind(from);
            if (pos != std::string::npos)
                s.replace(pos, from.size(), to);
        }
        return s;
    };
    char err[512] = {0};
    t2_pipeline * pipe = t2_pipeline_load(
        p("dino_f16.gguf").c_str(),
        p("ss_flow_f16.gguf").c_str(),
        p("ss_dec_f16.gguf").c_str(),
        p("slat_flow_f16.gguf").c_str(),
        p("slat_flow_1024_f16.gguf").c_str(),
        p("shape_dec_f16.gguf").c_str(),
        no_texture ? "" : p("shape_enc_f16.gguf").c_str(),
        no_texture ? "" : p("tex_dec_f16.gguf").c_str(),
        no_texture ? "" : p("tex_slat_flow_512_f16.gguf").c_str(),
        no_texture ? "" : p("tex_slat_flow_1024_f16.gguf").c_str(),
        0, err, (int) sizeof err);
    if (!pipe) {
        std::fprintf(stderr, "t2_pipeline_load failed: %s\n", err[0] ? err : "(no message)");
        return 1;
    }
    std::fprintf(stderr, "backend: %s  caps: 0x%x\n", t2_pipeline_backend(pipe), t2_pipeline_caps(pipe));

    // Optional RMBG-2.0 AI background removal model.
    if (!rmbg_model.empty()) {
        if (t2_rmbg_available()) {
            if (t2_pipeline_set_rmbg(pipe, rmbg_model.c_str(), rmbg_device.c_str(), err, (int) sizeof err) != 0) {
                std::fprintf(stderr, "RMBG load failed: %s\n", err[0] ? err : "(no message)");
                t2_pipeline_free(pipe);
                return 1;
            }
            std::fprintf(stderr, "RMBG-2.0 background removal enabled (%s)\n", rmbg_model.c_str());
        } else {
            std::fprintf(stderr, "warning: library built without RMBG support, --rmbg ignored\n");
        }
    }

    std::vector<uint8_t> img;
    if (!read_file(input.c_str(), img)) {
        std::fprintf(stderr, "read %s failed\n", input.c_str());
        t2_pipeline_free(pipe);
        return 1;
    }

    std::fprintf(stderr, "generating from %s (quality=%s, steps=%d)...\n",
                 input.c_str(), quality.c_str(), steps);
    t2_mesh_result * mesh = t2_generate(
        pipe, img.data(), (int) img.size(),
        pipe_type(quality.c_str()), background_mode,
        seed, steps, 7.5f, texture_steps,
        on_progress, nullptr, nullptr, nullptr,
        err, (int) sizeof err);
    t2_pipeline_free(pipe);
    if (!mesh) {
        std::fprintf(stderr, "t2_generate failed: %s\n", err[0] ? err : "(no message)");
        return 1;
    }

    std::fprintf(stderr, "mesh: %d verts, %d tris, pbr=%d\n",
                 t2_mesh_n_verts(mesh), t2_mesh_n_tris(mesh), t2_mesh_has_pbr(mesh));

    if (!write_mesh_bin(out_mesh.c_str(), mesh)) {
        std::fprintf(stderr, "write %s failed\n", out_mesh.c_str());
        t2_mesh_free(mesh);
        return 1;
    }
    std::fprintf(stderr, "wrote %s\n", out_mesh.c_str());

    if (!save_grid.empty()) {
        t2_grid_sidecar grid;
        grid.grid_res = t2_mesh_grid_res(mesh);
        const int nvox = t2_mesh_grid_nvox(mesh);
        if (grid.grid_res <= 0 || nvox <= 0) {
            std::fprintf(stderr, "warning: no dual grid to save (coarse path?)\n");
        } else {
            grid.feats.assign(t2_mesh_grid_feats(mesh), t2_mesh_grid_feats(mesh) + (size_t) nvox * 7);
            grid.coords.assign(t2_mesh_grid_coords(mesh), t2_mesh_grid_coords(mesh) + (size_t) nvox * 3);
            if (!write_t2grid(save_grid.c_str(), grid)) {
                std::fprintf(stderr, "write grid %s failed\n", save_grid.c_str());
                t2_mesh_free(mesh);
                return 1;
            }
            std::fprintf(stderr, "wrote %s (res=%d nvox=%d)\n", save_grid.c_str(), grid.grid_res, nvox);
        }
    }

    if (!out_glb.empty()) {
        int glen = 0;
        int tex_size = 2048;
        if (const char * ts = std::getenv("T2GLB_TEXTURE_SIZE")) {
            int v = std::atoi(ts);
            if (v >= 16 && v <= 8192) tex_size = v;
        }
        uint8_t * glb = t2_bake_glb(
            t2_mesh_verts(mesh), t2_mesh_n_verts(mesh),
            t2_mesh_tris(mesh), t2_mesh_n_tris(mesh),
            t2_mesh_has_pbr(mesh) ? t2_mesh_pbr(mesh) : nullptr,
            tex_size, 2, &glen, err, (int) sizeof err);
        if (!glb) {
            std::fprintf(stderr, "t2_bake_glb failed: %s\n", err[0] ? err : "(no message)");
            t2_mesh_free(mesh);
            return 1;
        }
        FILE * gf = std::fopen(out_glb.c_str(), "wb");
        if (!gf || (size_t) std::fwrite(glb, 1, (size_t) glen, gf) != (size_t) glen) {
            std::fprintf(stderr, "write %s failed\n", out_glb.c_str());
            t2_free_buffer(glb);
            t2_mesh_free(mesh);
            return 1;
        }
        std::fclose(gf);
        t2_free_buffer(glb);
        std::fprintf(stderr, "wrote %s (%d bytes)\n", out_glb.c_str(), glen);
    }

    t2_mesh_free(mesh);
    return 0;
}
