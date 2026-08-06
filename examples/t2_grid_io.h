#pragma once
// T2GRID01 sidecar: dual-grid decode output for standalone PBR texturing.
// Written by t2_generate --save-grid; consumed by t2_texture.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

struct t2_grid_sidecar {
    int grid_res = 0;
    std::vector<float> feats;     // 7 * nvox
    std::vector<int32_t> coords;  // 3 * nvox
};

inline bool read_t2mesh(const char * path,
                        std::vector<float> & verts,
                        std::vector<float> & normals,
                        std::vector<int> & tris) {
    FILE * f = std::fopen(path, "rb");
    if (!f) return false;
    char magic[8] = {};
    uint32_t nv = 0, nt = 0;
    if (std::fread(magic, 1, 8, f) != 8 ||
        std::fread(&nv, 4, 1, f) != 1 ||
        std::fread(&nt, 4, 1, f) != 1) {
        std::fclose(f);
        return false;
    }
    if (std::memcmp(magic, "T2MESH01", 8) != 0 &&
        std::memcmp(magic, "T2MESH03", 8) != 0) {
        std::fclose(f);
        return false;
    }
    const bool has_pbr = std::memcmp(magic, "T2MESH03", 8) == 0;
    verts.resize((size_t) nv * 3);
    normals.resize((size_t) nv * 3);
    tris.resize((size_t) nt * 3);
    if (std::fread(verts.data(), sizeof(float), (size_t) nv * 3, f) != (size_t) nv * 3 ||
        std::fread(normals.data(), sizeof(float), (size_t) nv * 3, f) != (size_t) nv * 3) {
        std::fclose(f);
        return false;
    }
    if (has_pbr) {
        std::vector<float> pbr((size_t) nv * 6);
        if (std::fread(pbr.data(), sizeof(float), pbr.size(), f) != pbr.size()) {
            std::fclose(f);
            return false;
        }
    }
    if (std::fread(tris.data(), sizeof(int), (size_t) nt * 3, f) != (size_t) nt * 3) {
        std::fclose(f);
        return false;
    }
    std::fclose(f);
    return true;
}

inline bool write_t2grid(const char * path, const t2_grid_sidecar & g) {
    const uint32_t nvox = (uint32_t) (g.coords.size() / 3);
    FILE * f = std::fopen(path, "wb");
    if (!f) return false;
    const char magic[8] = {'T','2','G','R','I','D','0','1'};
    const uint32_t res = (uint32_t) g.grid_res;
    auto wr = [&](const void * p, size_t n) { return std::fwrite(p, 1, n, f) == n; };
    bool ok = wr(magic, 8) && wr(&res, 4) && wr(&nvox, 4)
           && wr(g.feats.data(), g.feats.size() * sizeof(float))
           && wr(g.coords.data(), g.coords.size() * sizeof(int32_t));
    std::fclose(f);
    return ok;
}

inline bool read_t2grid(const char * path, t2_grid_sidecar & g) {
    FILE * f = std::fopen(path, "rb");
    if (!f) return false;
    char magic[8] = {};
    uint32_t res = 0, nvox = 0;
    if (std::fread(magic, 1, 8, f) != 8 ||
        std::fread(&res, 4, 1, f) != 1 ||
        std::fread(&nvox, 4, 1, f) != 1) {
        std::fclose(f);
        return false;
    }
    if (std::memcmp(magic, "T2GRID01", 8) != 0) {
        std::fclose(f);
        return false;
    }
    g.grid_res = (int) res;
    g.feats.resize((size_t) nvox * 7);
    g.coords.resize((size_t) nvox * 3);
    if (std::fread(g.feats.data(), sizeof(float), g.feats.size(), f) != g.feats.size() ||
        std::fread(g.coords.data(), sizeof(int32_t), g.coords.size(), f) != g.coords.size()) {
        std::fclose(f);
        return false;
    }
    std::fclose(f);
    return true;
}
