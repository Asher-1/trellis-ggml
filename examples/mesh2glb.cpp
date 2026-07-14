// mesh2glb — export a demo mesh (T2MESH01/T2MESH02/T2MESH03 wire format) to a
// portable vertex-coloured GLB. Exercises the CUDA-free component cleanup and
// glTF path offline, with no models or GPU. T2GLB_XATLAS opts into image baking.
//
//   mesh2glb in.bin out.glb [texture_size]
//
// The wire format is what the demo server emits at /api/mesh/{id}:
//   magic[8]  u32 nv  u32 nt  f32[3nv] verts  f32[3nv] normals
//   [T2MESH02: f32[5nv] legacy pbr]
//   [T2MESH03: f32[6nv] pbr incl. alpha] i32[3nt] tris (little-endian)

#include "mesh_export.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

template <class T>
static bool rd(FILE * f, std::vector<T> & v, size_t n) {
    v.resize(n);
    return n == 0 || std::fread(v.data(), sizeof(T), n, f) == n;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s in.bin out.glb [texture_size]\n", argv[0]);
        return 2;
    }
    t2glb::MeshExportOptions opt;
    opt.components = t2glb::ComponentFilter::KeepAll;
    if (argc > 3) opt.texture_size = std::atoi(argv[3]);

    FILE * f = std::fopen(argv[1], "rb");
    if (!f) { std::fprintf(stderr, "open %s failed\n", argv[1]); return 1; }
    char magic[9] = {0};
    uint32_t nv = 0, nt = 0;
    if (std::fread(magic, 1, 8, f) != 8 || std::fread(&nv, 4, 1, f) != 1 || std::fread(&nt, 4, 1, f) != 1) {
        std::fprintf(stderr, "bad header\n"); return 1;
    }
    const bool legacy = std::memcmp(magic, "T2MESH02", 8) == 0;
    const bool textured = legacy || std::memcmp(magic, "T2MESH03", 8) == 0;
    if (!textured && std::memcmp(magic, "T2MESH01", 8) != 0) {
        std::fprintf(stderr, "unknown magic\n"); return 1;
    }
    std::vector<float> verts, normals, pbr; std::vector<int32_t> tris;
    bool ok = rd(f, verts, (size_t) nv * 3) && rd(f, normals, (size_t) nv * 3);
    if (ok && textured) {
        if (legacy) {
            std::vector<float> old;
            ok = rd(f, old, (size_t) nv * 5);
            if (ok) {
                pbr.resize((size_t) nv * 6);
                for (uint32_t i = 0; i < nv; ++i) {
                    std::memcpy(pbr.data() + (size_t) i * 6,
                                old.data() + (size_t) i * 5, 5 * sizeof(float));
                    pbr[(size_t) i * 6 + 5] = 1.0f;
                }
            }
        } else {
            ok = rd(f, pbr, (size_t) nv * 6);
        }
    }
    ok = ok && rd(f, tris, (size_t) nt * 3);
    std::fclose(f);
    if (!ok) { std::fprintf(stderr, "truncated mesh\n"); return 1; }

    std::fprintf(stderr, "in: %s  %u verts  %u tris  %s\n", magic, nv, nt,
                 textured ? "textured" : "geometry-only");
    std::fprintf(stderr, "export: preserving %u input tris ...\n", nt);

    std::vector<uint8_t> glb; std::string err;
    if (!t2glb::mesh_to_glb(verts.data(), (int) nv, tris.data(), (int) nt,
                            textured ? pbr.data() : nullptr, opt, glb, err)) {
        std::fprintf(stderr, "mesh_to_glb: %s\n", err.c_str());
        return 1;
    }

    FILE * o = std::fopen(argv[2], "wb");
    if (!o) { std::fprintf(stderr, "open %s failed\n", argv[2]); return 1; }
    std::fwrite(glb.data(), 1, glb.size(), o);
    std::fclose(o);
    std::fprintf(stderr, "wrote %s (%.2f MB)\n", argv[2], glb.size() / 1048576.0);
    return 0;
}
