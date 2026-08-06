// Standalone PBR texturing: mesh + reference image -> textured mesh.
// With --grid: reuse decoded dual grid from t2_generate --save-grid.
// Without --grid: mesh -> QEF -> shape_enc (arbitrary mesh path).
//
// usage:
//   t2_texture --models-dir DIR --mesh M.t2mesh --input IMAGE --out-mesh OUT \
//              [--grid M.t2grid] [--out-glb OUT.glb] [--quality 512|1024]
//
#include "trellis2_capi.h"
#include "t2_grid_io.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static bool write_mesh_bin(const char * path, const t2_mesh_result * mesh) {
    const int nv = t2_mesh_n_verts(mesh);
    const int nt = t2_mesh_n_tris(mesh);
    const float * pbr = t2_mesh_has_pbr(mesh) ? t2_mesh_pbr(mesh) : nullptr;
    FILE * f = std::fopen(path, "wb");
    if (!f) return false;
    const char magic[8] = {'T','2','M','E','S','H','0','3'};
    const uint32_t unv = (uint32_t) nv, unt = (uint32_t) nt;
    auto wr = [&](const void * p, size_t n) { return std::fwrite(p, 1, n, f) == n; };
    bool ok = wr(magic, 8) && wr(&unv, 4) && wr(&unt, 4)
           && wr(t2_mesh_verts(mesh), (size_t) nv * 3 * sizeof(float))
           && wr(t2_mesh_normals(mesh), (size_t) nv * 3 * sizeof(float));
    if (ok && pbr) ok = wr(pbr, (size_t) nv * 6 * sizeof(float));
    if (ok) ok = wr(t2_mesh_tris(mesh), (size_t) nt * 3 * sizeof(int));
    std::fclose(f);
    return ok;
}

static int pipe_type(const char * q) {
    if (!q || std::strcmp(q, "512") == 0) return T2_PIPE_512;
    if (std::strcmp(q, "1024") == 0) return T2_PIPE_1024;
    return T2_PIPE_512;
}

static int grid_res_from_quality(const char * q) {
    if (q && std::strcmp(q, "1024") == 0) return 1024;
    return 512;
}

static void on_progress(void *, int stage, int step, int total) {
    std::fprintf(stderr, "  stage %d  step %d/%d\n", stage, step, total);
}

int main(int argc, char ** argv) {
    std::string models_dir = "models";
    std::string mesh_path, grid_path, input, out_mesh, out_glb;
    std::string quality = "512";
    std::string quant = "q8";
    uint64_t seed = 0;
    int texture_steps = 12;

    for (int i = 1; i < argc; ++i) {
        auto need = [&](const char * flag) -> const char * {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", flag); std::exit(2); }
            return argv[++i];
        };
        if (std::strcmp(argv[i], "--models-dir") == 0) models_dir = need("--models-dir");
        else if (std::strcmp(argv[i], "--quantization") == 0) quant = need("--quantization");
        else if (std::strcmp(argv[i], "--mesh") == 0) mesh_path = need("--mesh");
        else if (std::strcmp(argv[i], "--grid") == 0) grid_path = need("--grid");
        else if (std::strcmp(argv[i], "--input") == 0) input = need("--input");
        else if (std::strcmp(argv[i], "--out-mesh") == 0) out_mesh = need("--out-mesh");
        else if (std::strcmp(argv[i], "--out-glb") == 0) out_glb = need("--out-glb");
        else if (std::strcmp(argv[i], "--quality") == 0) quality = need("--quality");
        else if (std::strcmp(argv[i], "--seed") == 0) seed = (uint64_t) std::strtoull(need("--seed"), nullptr, 10);
        else if (std::strcmp(argv[i], "--texture-steps") == 0) texture_steps = std::atoi(need("--texture-steps"));
        else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            std::fprintf(stderr,
                "usage: %s --models-dir DIR --mesh M.t2mesh --input IMAGE --out-mesh OUT\n"
                "       [--grid M.t2grid] [--out-glb OUT.glb] [--quality 512|1024]\n"
                "       [--quantization q8|f16]  model weight precision (default: q8)\n"
                "       [--seed N] [--texture-steps N]\n"
                "  omit --grid to run mesh->QEF->shape_enc (arbitrary mesh path)\n", argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "unknown arg: %s\n", argv[i]);
            return 2;
        }
    }
    if (mesh_path.empty() || input.empty() || out_mesh.empty()) {
        std::fprintf(stderr, "required: --models-dir --mesh --input --out-mesh\n");
        return 2;
    }

    std::vector<float> verts, normals;
    std::vector<int> tris;
    if (!read_t2mesh(mesh_path.c_str(), verts, normals, tris)) {
        std::fprintf(stderr, "read mesh %s failed\n", mesh_path.c_str());
        return 1;
    }

    const int pt = pipe_type(quality.c_str());
    const int grid_res = grid_res_from_quality(quality.c_str());
    const float * grid_feats = nullptr;
    const int * grid_coords = nullptr;
    int grid_nvox = 0;
    t2_grid_sidecar grid;

    if (!grid_path.empty()) {
        if (!read_t2grid(grid_path.c_str(), grid)) {
            std::fprintf(stderr, "read grid %s failed\n", grid_path.c_str());
            return 1;
        }
        grid_feats = grid.feats.data();
        grid_coords = grid.coords.data();
        grid_nvox = (int) (grid.coords.size() / 3);
        std::fprintf(stderr, "texturing mesh %s + grid %s (quality=%s, nvox=%d)...\n",
                     mesh_path.c_str(), grid_path.c_str(), quality.c_str(), grid_nvox);
    } else {
        std::fprintf(stderr, "texturing mesh %s via QEF @ %d^3 (quality=%s, %zu tris)...\n",
                     mesh_path.c_str(), grid_res, quality.c_str(), tris.size() / 3);
    }

    std::vector<uint8_t> img;
    {
        FILE * f = std::fopen(input.c_str(), "rb");
        if (!f) { std::fprintf(stderr, "read %s failed\n", input.c_str()); return 1; }
        std::fseek(f, 0, SEEK_END);
        long n = std::ftell(f);
        std::fseek(f, 0, SEEK_SET);
        img.resize((size_t) n);
        std::fread(img.data(), 1, img.size(), f);
        std::fclose(f);
    }

    auto p = [&](const char * name) -> std::string {
        std::string s = models_dir + "/" + name;
        bool keep_f16 = (quant == "f16")
            || (std::string(name).find("shape_dec") != std::string::npos)
            || (std::string(name).find("tex_dec") != std::string::npos)
            || (std::string(name).find("shape_enc") != std::string::npos);
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
        p("shape_enc_f16.gguf").c_str(),
        p("tex_dec_f16.gguf").c_str(),
        p("tex_slat_flow_512_f16.gguf").c_str(),
        p("tex_slat_flow_1024_f16.gguf").c_str(),
        0, err, (int) sizeof err);
    if (!pipe) {
        std::fprintf(stderr, "t2_pipeline_load failed: %s\n", err[0] ? err : "(no message)");
        return 1;
    }

    const int effective_res = grid_path.empty() ? grid_res : grid.grid_res;
    t2_mesh_result * mesh = t2_texture_mesh(
        pipe, verts.data(), (int) (verts.size() / 3),
        tris.data(), (int) (tris.size() / 3),
        grid_feats, grid_nvox, grid_coords, effective_res, pt,
        img.data(), (int) img.size(), T2_BACKGROUND_AUTO,
        seed, texture_steps, on_progress, nullptr, err, (int) sizeof err);
    t2_pipeline_free(pipe);
    if (!mesh) {
        std::fprintf(stderr, "t2_texture_mesh failed: %s\n", err[0] ? err : "(no message)");
        return 1;
    }

    std::fprintf(stderr, "textured: %d verts, pbr=%d\n",
                 t2_mesh_n_verts(mesh), t2_mesh_has_pbr(mesh));
    if (!write_mesh_bin(out_mesh.c_str(), mesh)) {
        std::fprintf(stderr, "write %s failed\n", out_mesh.c_str());
        t2_mesh_free(mesh);
        return 1;
    }
    std::fprintf(stderr, "wrote %s\n", out_mesh.c_str());

    if (!out_glb.empty()) {
        int glen = 0;
        uint8_t * glb = t2_bake_glb(
            t2_mesh_verts(mesh), t2_mesh_n_verts(mesh),
            t2_mesh_tris(mesh), t2_mesh_n_tris(mesh),
            t2_mesh_pbr(mesh), 2048, 2, &glen, err, (int) sizeof err);
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
