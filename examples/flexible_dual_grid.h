// flexible_dual_grid.h — single-header CPU port of TRELLIS.2's flexible dual
// grid mesh extraction (o-voxel/o_voxel/convert/flexible_dual_grid.py, eval
// path). Turns the shape decoder's per-voxel 7-channel output into a triangle
// mesh, replacing the CUDA hashmap kernel with an std::unordered_map.
//
// Per active voxel v at integer coord c (in [0, grid_size)):
//   dual vertex   V_v = (c + offset_v) * voxel_size + aabb0   (unit cube here)
//   offset_v      = (1 + 2*margin) * sigmoid(feat[0:3]) - margin
//   intersected   feat[3:6] > 0, one flag per axis (x, y, z)
//   split_weight  softplus(feat[6])
// For each voxel with an intersected axis, the 4 voxels around that edge
// (offsets below) contribute their dual vertices as a quad; if all 4 exist the
// quad is split into 2 triangles by whichever diagonal gives better-aligned
// face normals (matching the split_weight product tie-break).
#pragma once

#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace fdg {

struct Mesh {
    std::vector<float> verts;  // 3 per vertex
    std::vector<int>   tris;   // 3 indices per triangle
    size_t n_verts() const { return verts.size() / 3; }
    size_t n_tris()  const { return tris.size() / 3; }
};

namespace detail {

inline uint64_t key(int32_t x, int32_t y, int32_t z) {
    return ((uint64_t) (uint32_t) x << 40) |
           ((uint64_t) (uint32_t) y << 20) |
           (uint64_t) (uint32_t) z;
}

// The 4 neighbor-voxel offsets around an edge, per axis (matches
// edge_neighbor_voxel_offset in the reference).
static const int EDGE_OFF[3][4][3] = {
    {{0,0,0},{0,0,1},{0,1,1},{0,1,0}},   // x-axis edge
    {{0,0,0},{1,0,0},{1,0,1},{0,0,1}},   // y-axis edge
    {{0,0,0},{0,1,0},{1,1,0},{1,0,0}},   // z-axis edge
};

inline void cross(const float * a, const float * b, float * o) {
    o[0] = a[1]*b[2] - a[2]*b[1];
    o[1] = a[2]*b[0] - a[0]*b[2];
    o[2] = a[0]*b[1] - a[1]*b[0];
}

} // namespace detail

// feats: [n_voxels * 7], voxel-major (dec output). coords: [n_voxels * 3].
// grid_size = input_res * decoder upscale (e.g. 32 * 16 = 512). margin 0.5.
inline Mesh extract(const float * feats, const int32_t * coords, int n,
                    int grid_size, float margin = 0.5f) {
    using namespace detail;
    Mesh m;
    if (n <= 0) return m;

    const float vs = 1.0f / (float) grid_size;   // voxel size (aabb span 1)
    const float aabb0 = -0.5f;

    // dual vertices + hashmap
    std::vector<float> V((size_t) n * 3);
    std::unordered_map<uint64_t, int> idx;
    idx.reserve((size_t) n * 2);
    for (int v = 0; v < n; ++v) {
        const float * f = feats + (size_t) v * 7;
        for (int a = 0; a < 3; ++a) {
            const float s = 1.0f / (1.0f + std::exp(-f[a]));          // sigmoid
            const float off = (1.0f + 2.0f * margin) * s - margin;
            V[(size_t) v * 3 + a] = ((float) coords[(size_t) v * 3 + a] + off) * vs + aabb0;
        }
        idx[key(coords[(size_t) v * 3], coords[(size_t) v * 3 + 1], coords[(size_t) v * 3 + 2])] = v;
    }
    m.verts = V;

    // quads from intersected edges
    for (int v = 0; v < n; ++v) {
        const float * f = feats + (size_t) v * 7;
        const int32_t cx = coords[(size_t) v * 3];
        const int32_t cy = coords[(size_t) v * 3 + 1];
        const int32_t cz = coords[(size_t) v * 3 + 2];
        for (int axis = 0; axis < 3; ++axis) {
            if (f[3 + axis] <= 0.0f) continue;   // not intersected on this axis
            int q[4];
            bool ok = true;
            for (int i = 0; i < 4; ++i) {
                const int32_t nx = cx + EDGE_OFF[axis][i][0];
                const int32_t ny = cy + EDGE_OFF[axis][i][1];
                const int32_t nz = cz + EDGE_OFF[axis][i][2];
                auto it = idx.find(key(nx, ny, nz));
                if (it == idx.end()) { ok = false; break; }
                q[i] = it->second;
            }
            if (!ok) continue;

            // choose the diagonal split with better-aligned normals
            const float * v0 = &V[(size_t) q[0] * 3];
            const float * v1 = &V[(size_t) q[1] * 3];
            const float * v2 = &V[(size_t) q[2] * 3];
            const float * v3 = &V[(size_t) q[3] * 3];
            auto align = [&](const float * a, const float * b, const float * c,
                             const float * d) {
                // two triangles (a,b,c) and (a,c,d): |n_abc . n_acd|
                float e1[3], e2[3], n0[3], n1[3];
                for (int k = 0; k < 3; ++k) { e1[k] = b[k]-a[k]; e2[k] = c[k]-a[k]; }
                cross(e1, e2, n0);
                for (int k = 0; k < 3; ++k) { e1[k] = c[k]-a[k]; e2[k] = d[k]-a[k]; }
                cross(e1, e2, n1);
                return std::fabs(n0[0]*n1[0] + n0[1]*n1[1] + n0[2]*n1[2]);
            };
            // split 1: (0,1,2)+(0,2,3); split 2: (0,1,3)+(3,1,2)
            const float a0 = align(v0, v1, v2, v3);
            const float a1 = align(v0, v1, v3, v2);   // reordered for split 2
            if (a0 >= a1) {
                m.tris.push_back(q[0]); m.tris.push_back(q[1]); m.tris.push_back(q[2]);
                m.tris.push_back(q[0]); m.tris.push_back(q[2]); m.tris.push_back(q[3]);
            } else {
                m.tris.push_back(q[0]); m.tris.push_back(q[1]); m.tris.push_back(q[3]);
                m.tris.push_back(q[3]); m.tris.push_back(q[1]); m.tris.push_back(q[2]);
            }
        }
    }
    return m;
}

// Per-vertex normals from area-weighted face normals (for shading).
inline std::vector<float> vertex_normals(const Mesh & m) {
    std::vector<float> nrm(m.verts.size(), 0.0f);
    for (size_t t = 0; t < m.tris.size(); t += 3) {
        const int i0 = m.tris[t], i1 = m.tris[t + 1], i2 = m.tris[t + 2];
        const float * a = &m.verts[(size_t) i0 * 3];
        const float * b = &m.verts[(size_t) i1 * 3];
        const float * c = &m.verts[(size_t) i2 * 3];
        float e1[3], e2[3], fn[3];
        for (int k = 0; k < 3; ++k) { e1[k] = b[k]-a[k]; e2[k] = c[k]-a[k]; }
        detail::cross(e1, e2, fn);
        for (int i : {i0, i1, i2})
            for (int k = 0; k < 3; ++k) nrm[(size_t) i * 3 + k] += fn[k];
    }
    for (size_t v = 0; v < m.n_verts(); ++v) {
        float * nn = &nrm[v * 3];
        const float len = std::sqrt(nn[0]*nn[0] + nn[1]*nn[1] + nn[2]*nn[2]);
        if (len > 1e-12f) { nn[0] /= len; nn[1] /= len; nn[2] /= len; }
    }
    return nrm;
}

} // namespace fdg
