// QEF mesh->dual-grid parity vs dumps/reference_texture.gguf enc_* tensors.
// usage: test_mesh_to_dual_grid <T2MESH01.bin> <reference_texture.gguf> [grid_size]
// exits 77 when inputs are missing.

#include "../examples/mesh_to_dual_grid.h"
#include "parity.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

static bool file_exists(const std::string & p) {
    std::ifstream f(p);
    return f.good();
}

static bool load_t2mesh(const char * path, std::vector<float> & verts, std::vector<int32_t> & tris) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    char magic[8];
    uint32_t nv = 0, nt = 0;
    if (!f.read(magic, 8) || std::memcmp(magic, "T2MESH01", 8) != 0) return false;
    if (!f.read(reinterpret_cast<char *>(&nv), 4)) return false;
    if (!f.read(reinterpret_cast<char *>(&nt), 4)) return false;
    verts.resize((size_t) nv * 3);
    if (!f.read(reinterpret_cast<char *>(verts.data()), verts.size() * sizeof(float))) return false;
    f.seekg((std::streamoff) nv * 3 * sizeof(float), std::ios::cur); // skip normals
    tris.resize((size_t) nt * 3);
    return (bool) f.read(reinterpret_cast<char *>(tris.data()), tris.size() * sizeof(int32_t));
}

static uint64_t vkey(int32_t x, int32_t y, int32_t z) {
    return ((uint64_t) (uint32_t) x << 42) | ((uint64_t) (uint32_t) y << 21) | (uint64_t) (uint32_t) z;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <T2MESH01.bin> <reference_texture.gguf> [grid_size]\n", argv[0]);
        return 2;
    }
    const std::string mesh_path = argv[1];
    const std::string ref_path = argv[2];
    const int grid_size = (argc >= 4) ? std::atoi(argv[3]) : 512;
    if (!file_exists(mesh_path) || !file_exists(ref_path)) {
        std::fprintf(stderr, "missing input file(s), skipping\n");
        return 77;
    }

    std::vector<float> verts;
    std::vector<int32_t> tris;
    if (!load_t2mesh(mesh_path.c_str(), verts, tris)) {
        std::fprintf(stderr, "failed to read %s\n", mesh_path.c_str());
        return 1;
    }

    t2_parity::baseline ref;
    if (!ref.open(ref_path)) {
        std::fprintf(stderr, "failed to open %s\n", ref_path.c_str());
        return 1;
    }
    std::vector<float> enc_vert_ref, enc_inter_ref, enc_coords4;
    if (!ref.load("enc_vert", enc_vert_ref) || !ref.load("enc_inter", enc_inter_ref) ||
        !ref.load("enc_coords", enc_coords4)) {
        std::fprintf(stderr, "reference missing enc_vert/enc_inter/enc_coords\n");
        return 1;
    }
    const int Nref = (int) (enc_coords4.size() / 4);

    mtdg::Result got = mtdg::mesh_to_flexible_dual_grid(
        verts.data(), (int) (verts.size() / 3), tris.data(), (int) (tris.size() / 3),
        grid_size, 1.f, 0.2f, 1e-2f);
    const int Ngot = (int) (got.coords.size() / 3);
    std::printf("mesh %zu verts / %zu tris -> %d voxels (ref %d) @ R=%d\n",
                verts.size() / 3, tris.size() / 3, Ngot, Nref, grid_size);
    if (Ngot == 0) {
        std::fprintf(stderr, "QEF returned empty grid\n");
        return 1;
    }

    std::vector<float> enc_vert(Ngot * 3), enc_inter(Ngot * 3);
    for (int v = 0; v < Ngot; ++v) {
        for (int c = 0; c < 3; ++c) {
            enc_vert[(size_t) v * 3 + c] =
                got.dual_verts[(size_t) v * 3 + c] * (float) grid_size -
                (float) got.coords[(size_t) v * 3 + c];
            enc_inter[(size_t) v * 3 + c] = (float) got.intersected[(size_t) v * 3 + c];
        }
    }

    if (Ngot != Nref) {
        std::printf("voxel count mismatch got=%d ref=%d -> FAIL\n", Ngot, Nref);
        return 1;
    }

    std::unordered_map<uint64_t, int> gm;
    gm.reserve((size_t) Ngot * 2);
    for (int v = 0; v < Ngot; ++v) {
        gm[vkey(got.coords[(size_t) v * 3], got.coords[(size_t) v * 3 + 1], got.coords[(size_t) v * 3 + 2])] = v;
    }

    std::vector<float> enc_vert_aligned((size_t) Nref * 3), enc_inter_aligned((size_t) Nref * 3);
    int miss = 0;
    for (int r = 0; r < Nref; ++r) {
        const int32_t cx = (int32_t) enc_coords4[(size_t) r * 4 + 1];
        const int32_t cy = (int32_t) enc_coords4[(size_t) r * 4 + 2];
        const int32_t cz = (int32_t) enc_coords4[(size_t) r * 4 + 3];
        auto it = gm.find(vkey(cx, cy, cz));
        if (it == gm.end()) { ++miss; continue; }
        const int gi = it->second;
        std::memcpy(enc_vert_aligned.data() + (size_t) r * 3, enc_vert.data() + (size_t) gi * 3, 3 * sizeof(float));
        std::memcpy(enc_inter_aligned.data() + (size_t) r * 3, enc_inter.data() + (size_t) gi * 3, 3 * sizeof(float));
    }
    if (miss) {
        std::printf("%d/%d ref voxels absent in got -> FAIL\n", miss, Nref);
        return 1;
    }

    int n_fail = 0;
    t2_parity::compare_stats st_v, st_i;
    t2_parity::compare(enc_vert_aligned, enc_vert_ref, "enc_vert", 1e-4f, 1e-3f, &st_v);
    t2_parity::compare(enc_inter_aligned, enc_inter_ref, "enc_inter", 0.f, 0.f, &st_i);
    if (st_v.max_abs > 5e-3f) { std::printf("enc_vert max_abs %.4g > 5e-3 -> FAIL\n", st_v.max_abs); ++n_fail; }
    if (st_i.max_abs > 0.5f) { std::printf("enc_inter max_abs %.4g > 0 -> FAIL\n", st_i.max_abs); ++n_fail; }
    return n_fail ? 1 : 0;
}
