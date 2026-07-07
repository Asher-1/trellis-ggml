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

// Per-vertex shading normals, robust to the dual grid's unoriented winding.
//
// The reference flexible_dual_grid mesher emits every quad with a fixed vertex
// order (edge_neighbor_voxel_offset) regardless of which way the surface crosses
// the edge, so ~15-20% of faces are wound opposite to their neighbours. A plain
// area-weighted vector sum (sum of signed face normals) therefore partially
// cancels at those vertices, producing short, noisy normals and salt-and-pepper
// shading. This is faithful to TRELLIS.2 (its mesh is unoriented too) but looks
// grainy under one-sided lighting.
//
// Instead accumulate the area-weighted *structure tensor* A = sum(area * n̂ n̂ᵀ).
// Because n̂ n̂ᵀ == (−n̂)(−n̂)ᵀ it is immune to winding sign, so its dominant
// eigenvector recovers the true surface-normal *direction* even where signed
// normals cancel. We resolve the (arbitrary) sign to agree with the signed sum
// for determinism; final shading is orientation-independent (see the viewer's
// abs(dot) diffuse), so the sign only needs to be stable, not globally correct.
inline std::vector<float> vertex_normals(const Mesh & m) {
    const size_t nv = m.n_verts();
    std::vector<double> A((size_t) nv * 6, 0.0);  // sym 3x3: xx,yy,zz,xy,xz,yz
    std::vector<float>  sgn((size_t) nv * 3, 0.0f); // signed area-weighted sum (sign seed)
    for (size_t t = 0; t < m.tris.size(); t += 3) {
        const int i0 = m.tris[t], i1 = m.tris[t + 1], i2 = m.tris[t + 2];
        const float * a = &m.verts[(size_t) i0 * 3];
        const float * b = &m.verts[(size_t) i1 * 3];
        const float * c = &m.verts[(size_t) i2 * 3];
        float e1[3], e2[3], fn[3];
        for (int k = 0; k < 3; ++k) { e1[k] = b[k]-a[k]; e2[k] = c[k]-a[k]; }
        detail::cross(e1, e2, fn);   // |fn| == 2*area, direction == face normal
        const double area = std::sqrt((double) fn[0]*fn[0] +
                                      (double) fn[1]*fn[1] +
                                      (double) fn[2]*fn[2]);
        if (area <= 1e-20) continue;
        const double inv = 1.0 / area;                       // area * (fn/|fn|)(fn/|fn|)ᵀ
        const double xx = fn[0]*fn[0]*inv, yy = fn[1]*fn[1]*inv, zz = fn[2]*fn[2]*inv;
        const double xy = fn[0]*fn[1]*inv, xz = fn[0]*fn[2]*inv, yz = fn[1]*fn[2]*inv;
        for (int i : {i0, i1, i2}) {
            double * Av = &A[(size_t) i * 6];
            Av[0]+=xx; Av[1]+=yy; Av[2]+=zz; Av[3]+=xy; Av[4]+=xz; Av[5]+=yz;
            float * sv = &sgn[(size_t) i * 3];
            sv[0]+=fn[0]; sv[1]+=fn[1]; sv[2]+=fn[2];
        }
    }
    std::vector<float> nrm((size_t) nv * 3, 0.0f);
    for (size_t v = 0; v < nv; ++v) {
        const double * Av = &A[v * 6];
        // dominant eigenvector of the symmetric structure tensor via power
        // iteration, seeded from the signed sum (a good guess where it survives).
        double x = sgn[v*3], y = sgn[v*3+1], z = sgn[v*3+2];
        double l = std::sqrt(x*x + y*y + z*z);
        if (l < 1e-20) { x = Av[0]; y = Av[3]; z = Av[4]; l = std::sqrt(x*x+y*y+z*z); }
        if (l < 1e-20) { nrm[v*3]=0.0f; nrm[v*3+1]=0.0f; nrm[v*3+2]=1.0f; continue; }
        x/=l; y/=l; z/=l;
        for (int it = 0; it < 8; ++it) {
            const double nx = Av[0]*x + Av[3]*y + Av[4]*z;
            const double ny = Av[3]*x + Av[1]*y + Av[5]*z;
            const double nz = Av[4]*x + Av[5]*y + Av[2]*z;
            const double nl = std::sqrt(nx*nx + ny*ny + nz*nz);
            if (nl < 1e-20) break;
            x = nx/nl; y = ny/nl; z = nz/nl;
        }
        const float * sv = &sgn[v*3];
        const double s = (x*sv[0] + y*sv[1] + z*sv[2]) < 0.0 ? -1.0 : 1.0;
        nrm[v*3]   = (float) (s*x);
        nrm[v*3+1] = (float) (s*y);
        nrm[v*3+2] = (float) (s*z);
    }
    return nrm;
}

} // namespace fdg
