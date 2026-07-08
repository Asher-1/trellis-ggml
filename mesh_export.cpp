#include "mesh_export.h"

#include "xatlas.h"
#include "meshoptimizer.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <unordered_map>

namespace t2glb {
namespace {

// Stage logging to stderr, opt-in via T2GLB_VERBOSE (keeps the library quiet by
// default while giving the CLI / debugging a progress trace).
bool verbose() { static bool v = std::getenv("T2GLB_VERBOSE") != nullptr; return v; }
#define GLBLOG(...) do { if (verbose()) { std::fprintf(stderr, "[glb] " __VA_ARGS__); std::fprintf(stderr, "\n"); } } while (0)

// ─────────────────────────────────────────────────────────────────────────
// QEM edge-collapse decimation (meshoptimizer). Reduces the dense mesh to
// ~target_tris. meshopt_simplify is position-preserving (collapses to existing
// vertices), so the decimated vertices are a subset of the dense ones — the
// nearest-dense-vertex bake below stays exact at vertices. Colours are NOT
// carried through; they are re-sampled from the dense source after unwrapping,
// mirroring the reference's BVH attribute transfer.
// ─────────────────────────────────────────────────────────────────────────
void decimate(const float * verts, int nv, const int32_t * tris, int nt,
              int target_tris, std::vector<float> & dverts, std::vector<int32_t> & dtris) {
    std::vector<unsigned int> idx((size_t) nt * 3);
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = (unsigned int) tris[i];

    const size_t target_idx = (size_t) std::max(target_tris, 1) * 3;
    if (idx.size() > target_idx) {
        float result_error = 0.0f;
        // Pass 1: topology-preserving QEM edge collapse (best quality). target_error
        // 1.0 (max normalised) so it collapses to the budget rather than stopping
        // early on a quality threshold.
        std::vector<unsigned int> simp(idx.size());
        size_t n = meshopt_simplify(simp.data(), idx.data(), idx.size(),
                                    verts, (size_t) nv, sizeof(float) * 3,
                                    target_idx, 1.0f, /*options*/ 0, &result_error);
        // The dual-grid surface has non-manifold edges that pass 1 won't collapse
        // across, so it often stalls well above target. Pass 2 (sloppy grid
        // clustering, topology-agnostic) forces the budget — fine for a baked
        // asset whose detail lives in the texture, not the geometry.
        if (n > target_idx + target_idx / 4 && !std::getenv("T2GLB_NOSLOPPY")) {
            std::vector<unsigned int> sl(n);
            size_t m = meshopt_simplifySloppy(sl.data(), simp.data(), n,
                                              verts, (size_t) nv, sizeof(float) * 3,
                                              /*vertex_lock*/ nullptr, target_idx, 1.0f, &result_error);
            sl.resize(m);
            idx.swap(sl);
        } else {
            simp.resize(n);
            idx.swap(simp);
        }
    }

    // Drop degenerate triangles (a collapse can leave <3 distinct vertices) —
    // xatlas chokes on zero-area faces. Then compact: gather referenced
    // vertices, remap indices.
    std::vector<int> remap((size_t) nv, -1);
    dverts.clear(); dverts.reserve(idx.size());
    dtris.clear(); dtris.reserve(idx.size());
    for (size_t t = 0; t + 2 < idx.size(); t += 3) {
        unsigned int a = idx[t], b = idx[t+1], c = idx[t+2];
        if (a == b || b == c || a == c) continue;
        for (unsigned int v : {a, b, c}) {
            if (remap[v] < 0) {
                remap[v] = (int)(dverts.size() / 3);
                dverts.push_back(verts[3*v+0]); dverts.push_back(verts[3*v+1]); dverts.push_back(verts[3*v+2]);
            }
            dtris.push_back(remap[v]);
        }
    }
}

// Remove triangles belonging to small connected components (islands). The
// dual-grid surface — especially after sloppy clustering — sheds many tiny
// disconnected bits that each force a separate UV chart, making xatlas crawl.
// Mirrors the reference's remove_small_connected_components. Keeps components
// with at least `min_frac` of the triangles; recompacts verts/tris in place.
void remove_small_components(std::vector<float> & v, std::vector<int32_t> & f, float min_frac) {
    const int nv = (int)(v.size() / 3), nt = (int)(f.size() / 3);
    if (nv == 0 || nt == 0) return;
    std::vector<int> parent((size_t) nv), sz((size_t) nv, 1);
    for (int i = 0; i < nv; ++i) parent[i] = i;
    std::function<int(int)> find = [&](int x) { while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; } return x; };
    auto uni = [&](int a, int b) { a = find(a); b = find(b); if (a==b) return; if (sz[a]<sz[b]) std::swap(a,b); parent[b]=a; sz[a]+=sz[b]; };
    for (int t = 0; t < nt; ++t) { uni(f[3*t+0], f[3*t+1]); uni(f[3*t+1], f[3*t+2]); }

    std::unordered_map<int, int> tris_in;   // component root -> triangle count
    for (int t = 0; t < nt; ++t) tris_in[find(f[3*t+0])]++;
    const int keep_min = std::max(1, (int)(nt * min_frac));

    std::vector<int> vremap((size_t) nv, -1);
    std::vector<float> nvts; nvts.reserve(v.size());
    std::vector<int32_t> nfs; nfs.reserve(f.size());
    for (int t = 0; t < nt; ++t) {
        if (tris_in[find(f[3*t+0])] < keep_min) continue;
        for (int k = 0; k < 3; ++k) {
            int vi = f[3*t+k];
            if (vremap[vi] < 0) {
                vremap[vi] = (int)(nvts.size()/3);
                nvts.push_back(v[3*vi+0]); nvts.push_back(v[3*vi+1]); nvts.push_back(v[3*vi+2]);
            }
            nfs.push_back(vremap[vi]);
        }
    }
    if (!nfs.empty()) { v.swap(nvts); f.swap(nfs); }
}

// Taubin (λ|μ) smoothing — regularises the zig-zag sliver triangles left by
// sloppy decimation so per-face normals become coherent. Without this, xatlas
// splits a chart at every normal jump (tens of thousands of one-off charts).
// Taubin's alternating positive/negative passes smooth without the shrinkage of
// plain Laplacian. The texture is unaffected (it samples the dense source).
void taubin_smooth(std::vector<float> & v, const std::vector<int32_t> & f, int iters) {
    const int nv = (int)(v.size() / 3), nt = (int)(f.size() / 3);
    if (nv == 0 || nt == 0) return;
    std::vector<int> deg((size_t) nv, 0);
    // Accumulate neighbour sums per pass; build degree once.
    for (int t = 0; t < nt; ++t)
        for (int k = 0; k < 3; ++k) deg[f[3*t+k]] += 2; // each vertex in 2 edges of its triangle
    const float lambda = 0.5f, mu = -0.53f;
    std::vector<float> acc((size_t) nv * 3);
    auto pass = [&](float w) {
        std::fill(acc.begin(), acc.end(), 0.0f);
        for (int t = 0; t < nt; ++t) {
            int a = f[3*t+0], b = f[3*t+1], c = f[3*t+2];
            for (int e = 0; e < 3; ++e) {
                int i = (e==0?a:e==1?b:c), j = (e==0?b:e==1?c:a);
                acc[3*i+0]+=v[3*j+0]; acc[3*i+1]+=v[3*j+1]; acc[3*i+2]+=v[3*j+2];
                acc[3*j+0]+=v[3*i+0]; acc[3*j+1]+=v[3*i+1]; acc[3*j+2]+=v[3*i+2];
            }
        }
        for (int i = 0; i < nv; ++i) {
            if (deg[i] == 0) continue;
            float inv = 1.0f / deg[i];
            for (int k = 0; k < 3; ++k)
                v[3*i+k] += w * (acc[3*i+k]*inv - v[3*i+k]); // move toward neighbour centroid
        }
    };
    for (int it = 0; it < iters; ++it) { pass(lambda); pass(mu); }
}

// Area-weighted per-vertex normals over a mesh.
void vertex_normals(const std::vector<float> & v, const std::vector<int32_t> & f,
                    std::vector<float> & n) {
    const size_t nv = v.size() / 3;
    n.assign(nv * 3, 0.0f);
    for (size_t t = 0; t < f.size() / 3; ++t) {
        int a = f[3*t+0], b = f[3*t+1], c = f[3*t+2];
        float e1[3] = {v[3*b+0]-v[3*a+0], v[3*b+1]-v[3*a+1], v[3*b+2]-v[3*a+2]};
        float e2[3] = {v[3*c+0]-v[3*a+0], v[3*c+1]-v[3*a+1], v[3*c+2]-v[3*a+2]};
        float fn[3] = {e1[1]*e2[2]-e1[2]*e2[1], e1[2]*e2[0]-e1[0]*e2[2], e1[0]*e2[1]-e1[1]*e2[0]};
        for (int k : {a, b, c}) { n[3*k+0]+=fn[0]; n[3*k+1]+=fn[1]; n[3*k+2]+=fn[2]; }
    }
    for (size_t i = 0; i < nv; ++i) {
        float l = std::sqrt(n[3*i+0]*n[3*i+0]+n[3*i+1]*n[3*i+1]+n[3*i+2]*n[3*i+2]) + 1e-20f;
        n[3*i+0]/=l; n[3*i+1]/=l; n[3*i+2]/=l;
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Uniform spatial hash over the dense vertices for nearest-vertex PBR lookup.
// The dense mesh is voxel-regular, so the nearest dense vertex to any point on
// the decimated surface carries essentially the correct attribute (the same
// role the reference's cuBVH + grid_sample plays).
// ─────────────────────────────────────────────────────────────────────────
struct VertexGrid {
    const float * v = nullptr;
    int nv = 0;
    float inv_cell = 0.0f;
    float org[3] = {0,0,0};
    int dim[3] = {1,1,1};
    std::unordered_map<int64_t, std::vector<int>> cells;

    int64_t key(int x, int y, int z) const {
        return (int64_t)(x & 0x1FFFFF) | ((int64_t)(y & 0x1FFFFF) << 21)
             | ((int64_t)(z & 0x1FFFFF) << 42);
    }
    void cell_of(const float * p, int & cx, int & cy, int & cz) const {
        cx = (int)((p[0]-org[0])*inv_cell); cy = (int)((p[1]-org[1])*inv_cell);
        cz = (int)((p[2]-org[2])*inv_cell);
    }

    void build(const float * verts, int n) {
        v = verts; nv = n;
        float lo[3] = {1e30f,1e30f,1e30f}, hi[3] = {-1e30f,-1e30f,-1e30f};
        for (int i = 0; i < n; ++i)
            for (int k = 0; k < 3; ++k) {
                lo[k] = std::min(lo[k], verts[3*i+k]); hi[k] = std::max(hi[k], verts[3*i+k]);
            }
        float ext = std::max({hi[0]-lo[0], hi[1]-lo[1], hi[2]-lo[2], 1e-6f});
        // ~1 vertex per cell for a surface of ~n verts: cell ≈ ext / n^(1/2).
        float cells_per_axis = std::max(16.0f, std::sqrt((float) std::max(n, 1)));
        cells_per_axis = std::min(cells_per_axis, 2000.0f);
        float cell = ext / cells_per_axis;
        inv_cell = 1.0f / cell;
        for (int k = 0; k < 3; ++k) { org[k] = lo[k]; dim[k] = (int)((hi[k]-lo[k])*inv_cell)+2; }
        cells.reserve((size_t) n);
        for (int i = 0; i < n; ++i) {
            int cx, cy, cz; cell_of(&verts[3*i], cx, cy, cz);
            cells[key(cx, cy, cz)].push_back(i);
        }
    }

    // Nearest dense vertex to p; grows the search ring until a candidate is
    // found, then one extra ring so the true nearest isn't missed at a boundary.
    int nearest(const float * p) const {
        int cx, cy, cz; cell_of(p, cx, cy, cz);
        int best = -1; float bestd = 1e30f;
        int maxr = std::max({dim[0], dim[1], dim[2]});
        int found_r = -1;
        for (int r = 0; r <= maxr; ++r) {
            // Scan only the shell at radius r (interior already scanned).
            for (int dz = -r; dz <= r; ++dz)
            for (int dy = -r; dy <= r; ++dy)
            for (int dx = -r; dx <= r; ++dx) {
                if (std::max({std::abs(dx),std::abs(dy),std::abs(dz)}) != r) continue;
                auto it = cells.find(key(cx+dx, cy+dy, cz+dz));
                if (it == cells.end()) continue;
                for (int idx : it->second) {
                    float ddx=v[3*idx+0]-p[0], ddy=v[3*idx+1]-p[1], ddz=v[3*idx+2]-p[2];
                    float d = ddx*ddx+ddy*ddy+ddz*ddz;
                    if (d < bestd) { bestd = d; best = idx; }
                }
            }
            if (best >= 0 && found_r < 0) found_r = r;
            if (found_r >= 0 && r >= found_r + 1) break; // one extra ring, then stop
        }
        return best;
    }
};

// ─────────────────────────────────────────────────────────────────────────
// PNG encode to an in-memory byte vector.
// ─────────────────────────────────────────────────────────────────────────
void png_sink(void * ctx, void * data, int size) {
    auto * v = (std::vector<uint8_t> *) ctx;
    auto * p = (uint8_t *) data;
    v->insert(v->end(), p, p + size);
}
bool encode_png(int w, int h, int comp, const uint8_t * pix, std::vector<uint8_t> & out) {
    out.clear();
    return stbi_write_png_to_func(png_sink, &out, w, h, comp, pix, w * comp) != 0;
}

// ─────────────────────────────────────────────────────────────────────────
// glTF 2.0 binary (GLB) assembly. One buffer holding, in order: POSITION,
// NORMAL, TEXCOORD_0, indices, baseColor PNG, metallicRoughness PNG.
// ─────────────────────────────────────────────────────────────────────────
void put_u32(std::vector<uint8_t> & b, uint32_t v) {
    b.push_back(v & 0xFF); b.push_back((v>>8)&0xFF); b.push_back((v>>16)&0xFF); b.push_back((v>>24)&0xFF);
}
void pad4(std::vector<uint8_t> & b, uint8_t fill) { while (b.size() % 4) b.push_back(fill); }

void write_glb(const std::vector<float> & pos, const std::vector<float> & nrm,
               const std::vector<float> & uv, const std::vector<uint32_t> & idx,
               const std::vector<uint8_t> & basecolor_png,
               const std::vector<uint8_t> & metalrough_png,
               std::vector<uint8_t> & out) {
    const uint32_t nv = (uint32_t)(pos.size() / 3);
    const uint32_t ni = (uint32_t) idx.size();

    // Assemble the BIN buffer, tracking 4-aligned bufferView offsets.
    std::vector<uint8_t> bin;
    auto view = [&](const void * data, size_t bytes, uint32_t & off, uint32_t & len) {
        pad4(bin, 0);
        off = (uint32_t) bin.size();
        len = (uint32_t) bytes;
        const uint8_t * p = (const uint8_t *) data;
        bin.insert(bin.end(), p, p + bytes);
    };
    uint32_t off_pos, len_pos, off_nrm, len_nrm, off_uv, len_uv, off_idx, len_idx;
    uint32_t off_bc, len_bc, off_mr, len_mr;
    view(pos.data(), pos.size()*4, off_pos, len_pos);
    view(nrm.data(), nrm.size()*4, off_nrm, len_nrm);
    view(uv.data(),  uv.size()*4,  off_uv,  len_uv);
    view(idx.data(), idx.size()*4, off_idx, len_idx);
    view(basecolor_png.data(), basecolor_png.size(), off_bc, len_bc);
    view(metalrough_png.data(), metalrough_png.size(), off_mr, len_mr);
    pad4(bin, 0);

    float pmin[3] = {1e30f,1e30f,1e30f}, pmax[3] = {-1e30f,-1e30f,-1e30f};
    for (uint32_t i = 0; i < nv; ++i)
        for (int k = 0; k < 3; ++k) { pmin[k]=std::min(pmin[k],pos[3*i+k]); pmax[k]=std::max(pmax[k],pos[3*i+k]); }

    char buf[2048];
    std::string j = "{\"asset\":{\"version\":\"2.0\",\"generator\":\"trellis2cpp\"},";
    j += "\"scene\":0,\"scenes\":[{\"nodes\":[0]}],\"nodes\":[{\"mesh\":0}],";
    j += "\"meshes\":[{\"primitives\":[{\"attributes\":{\"POSITION\":0,\"NORMAL\":1,\"TEXCOORD_0\":2},\"indices\":3,\"material\":0}]}],";
    j += "\"materials\":[{\"pbrMetallicRoughness\":{\"baseColorTexture\":{\"index\":0},\"metallicRoughnessTexture\":{\"index\":1},\"metallicFactor\":1.0,\"roughnessFactor\":1.0},\"doubleSided\":true}],";
    j += "\"textures\":[{\"source\":0,\"sampler\":0},{\"source\":1,\"sampler\":0}],";
    j += "\"images\":[{\"bufferView\":4,\"mimeType\":\"image/png\"},{\"bufferView\":5,\"mimeType\":\"image/png\"}],";
    // minFilter LINEAR (no mipmaps): the grid atlas packs unrelated triangles in
    // adjacent cells, so mip downsampling would bleed colour across cell seams.
    j += "\"samplers\":[{\"magFilter\":9729,\"minFilter\":9729,\"wrapS\":33071,\"wrapT\":33071}],";
    snprintf(buf, sizeof buf,
        "\"accessors\":[{\"bufferView\":0,\"componentType\":5126,\"count\":%u,\"type\":\"VEC3\",\"min\":[%.8g,%.8g,%.8g],\"max\":[%.8g,%.8g,%.8g]},"
        "{\"bufferView\":1,\"componentType\":5126,\"count\":%u,\"type\":\"VEC3\"},"
        "{\"bufferView\":2,\"componentType\":5126,\"count\":%u,\"type\":\"VEC2\"},"
        "{\"bufferView\":3,\"componentType\":5125,\"count\":%u,\"type\":\"SCALAR\"}],",
        nv, pmin[0],pmin[1],pmin[2], pmax[0],pmax[1],pmax[2], nv, nv, ni);
    j += buf;
    snprintf(buf, sizeof buf,
        "\"bufferViews\":[{\"buffer\":0,\"byteOffset\":%u,\"byteLength\":%u,\"target\":34962},"
        "{\"buffer\":0,\"byteOffset\":%u,\"byteLength\":%u,\"target\":34962},"
        "{\"buffer\":0,\"byteOffset\":%u,\"byteLength\":%u,\"target\":34962},"
        "{\"buffer\":0,\"byteOffset\":%u,\"byteLength\":%u,\"target\":34963},"
        "{\"buffer\":0,\"byteOffset\":%u,\"byteLength\":%u},"
        "{\"buffer\":0,\"byteOffset\":%u,\"byteLength\":%u}],",
        off_pos,len_pos, off_nrm,len_nrm, off_uv,len_uv, off_idx,len_idx, off_bc,len_bc, off_mr,len_mr);
    j += buf;
    snprintf(buf, sizeof buf, "\"buffers\":[{\"byteLength\":%u}]}", (uint32_t) bin.size());
    j += buf;

    std::vector<uint8_t> json(j.begin(), j.end());
    while (json.size() % 4) json.push_back(0x20); // pad JSON chunk with spaces

    const uint32_t total = 12 + 8 + (uint32_t) json.size() + 8 + (uint32_t) bin.size();
    out.clear();
    out.reserve(total);
    put_u32(out, 0x46546C67); put_u32(out, 2); put_u32(out, total);          // header
    put_u32(out, (uint32_t) json.size()); put_u32(out, 0x4E4F534A);          // JSON chunk
    out.insert(out.end(), json.begin(), json.end());
    put_u32(out, (uint32_t) bin.size()); put_u32(out, 0x004E4942);           // BIN chunk
    out.insert(out.end(), bin.begin(), bin.end());
}

// Chart-based unwrap via xatlas → per-output-vertex position/normal/uv (texel
// space) + index buffer + atlas dims. Proper editable charts, but only fast on
// clean, manifold meshes. Opt-in (T2GLB_XATLAS); the grid atlas is the default.
bool xatlas_unwrap(const std::vector<float> & dverts, const std::vector<float> & dnrm, int dnv,
                   const std::vector<int32_t> & dtris, int dnt, const MeshExportOptions & opt, int TS,
                   std::vector<float> & opos, std::vector<float> & onrm, std::vector<float> & ouv,
                   std::vector<uint32_t> & oidx, int & AW, int & AH, std::string & err) {
    xatlas::Atlas * atlas = xatlas::Create();
    xatlas::MeshDecl md{};
    md.vertexCount = (uint32_t) dnv;
    md.vertexPositionData = dverts.data();
    md.vertexPositionStride = sizeof(float) * 3;
    md.indexCount = (uint32_t)(dnt * 3);
    md.indexData = dtris.data();
    md.indexFormat = xatlas::IndexFormat::UInt32;
    if (xatlas::AddMesh(atlas, md) != xatlas::AddMeshError::Success) {
        xatlas::Destroy(atlas); err = "xatlas AddMesh failed"; return false;
    }
    GLBLOG("xatlas computing charts ...");
    xatlas::ChartOptions co{};
    co.normalDeviationWeight = 0.5f; co.roundnessWeight = 0.0f; co.straightnessWeight = 1.0f;
    co.normalSeamWeight = 1.0f; co.textureSeamWeight = 0.0f; co.maxCost = 8.0f; co.maxIterations = 1;
    xatlas::ComputeCharts(atlas, co);

    xatlas::PackOptions po{};
    po.padding = (uint32_t) opt.padding;
    po.bilinear = true;
    po.createImage = false;
    uint32_t res = (uint32_t) TS;
    for (int attempt = 0; attempt < 4; ++attempt) {
        po.resolution = res;
        xatlas::PackCharts(atlas, po);
        if (atlas->atlasCount <= 1) break;
        res = (uint32_t)(res * 1.5f);
    }
    GLBLOG("atlas %ux%u, %u pages, %u charts", atlas->width, atlas->height, atlas->atlasCount, atlas->chartCount);
    if (atlas->meshCount != 1 || atlas->atlasCount != 1 || atlas->width == 0 || atlas->height == 0) {
        xatlas::Destroy(atlas); err = "xatlas packing failed (charts did not fit one atlas)"; return false;
    }
    AW = (int) atlas->width; AH = (int) atlas->height;
    const xatlas::Mesh & om = atlas->meshes[0];
    const uint32_t onv = om.vertexCount;
    opos.resize(onv*3); onrm.resize(onv*3); ouv.resize(onv*2);
    for (uint32_t i = 0; i < onv; ++i) {
        uint32_t xr = om.vertexArray[i].xref;
        if (xr >= (uint32_t) dnv) xr = 0;
        opos[3*i+0]=dverts[3*xr+0]; opos[3*i+1]=dverts[3*xr+1]; opos[3*i+2]=dverts[3*xr+2];
        onrm[3*i+0]=dnrm[3*xr+0];   onrm[3*i+1]=dnrm[3*xr+1];   onrm[3*i+2]=dnrm[3*xr+2];
        ouv[2*i+0]=om.vertexArray[i].uv[0]; ouv[2*i+1]=om.vertexArray[i].uv[1];
    }
    oidx.assign(om.indexArray, om.indexArray + om.indexCount);
    xatlas::Destroy(atlas);
    return true;
}

std::mutex g_bake_mu; // serialize bakes (bounds peak RAM; keeps stb/xatlas tidy)

} // namespace

bool mesh_to_glb(const float * verts, int nv,
                 const int32_t * tris, int nt,
                 const float * pbr,
                 const MeshExportOptions & opt,
                 std::vector<uint8_t> & out,
                 std::string & err) {
    if (nv <= 0 || nt <= 0) { err = "empty mesh"; return false; }
    const int TS = opt.texture_size;
    if (TS < 16 || TS > 8192) { err = "bad texture_size"; return false; }

    std::lock_guard<std::mutex> lock(g_bake_mu);

    // 1) decimate ----------------------------------------------------------
    GLBLOG("input %d verts %d tris; decimating to %d tris ...", nv, nt, opt.target_tris);
    std::vector<float> dverts; std::vector<int32_t> dtris;
    decimate(verts, nv, tris, nt, std::max(opt.target_tris, 64), dverts, dtris);
    GLBLOG("decimated -> %d verts %d tris", (int)(dverts.size()/3), (int)(dtris.size()/3));

    // Drop tiny islands so xatlas isn't swamped with one-off charts.
    remove_small_components(dverts, dtris, 0.0005f);
    const int dnv = (int)(dverts.size()/3), dnt = (int)(dtris.size()/3);
    if (dnv < 3 || dnt < 1) { err = "decimation produced degenerate mesh"; return false; }
    GLBLOG("after component filter -> %d verts %d tris", dnv, dnt);

    // Regularise sliver geometry so xatlas can grow coherent charts.
    if (!std::getenv("T2GLB_NOSMOOTH")) taubin_smooth(dverts, dtris, 3);

    std::vector<float> dnrm; vertex_normals(dverts, dtris, dnrm);

    // 2) UV atlas ----------------------------------------------------------
    // Output-vertex streams: opos/onrm (geometry) and ouv (atlas-texel space),
    // plus the triangle index buffer oidx and atlas dims AW×AH.
    std::vector<float> opos, onrm, ouv;
    std::vector<uint32_t> oidx;
    int AW = TS, AH = TS;

    if (std::getenv("T2GLB_XATLAS")) {
        // Chart-based unwrap (proper, editable UVs). Only fast on clean, manifold
        // meshes — machine-generated dual-grid geometry shatters it into tens of
        // thousands of charts. Available for clean inputs / experimentation.
        if (!xatlas_unwrap(dverts, dnrm, dnv, dtris, dnt, opt, TS, opos, onrm, ouv, oidx, AW, AH, err))
            return false;
    } else {
        // Deterministic per-triangle grid atlas: topology-agnostic, O(n), exactly
        // TS×TS, never chokes. Two triangles per square cell (its right-triangle
        // halves). The bake samples the dense source, so per-texel colour is
        // faithful regardless of the decimated geometry's UV layout.
        const int ncell = (dnt + 1) / 2;
        const int gdim  = std::max(1, (int) std::ceil(std::sqrt((double) ncell)));
        const float cell = (float) TS / gdim;
        const float m = 0.75f;   // texel margin: gutter for dilation between cells
        opos.resize((size_t) dnt * 9); onrm.resize((size_t) dnt * 9);
        ouv.resize((size_t) dnt * 6);  oidx.resize((size_t) dnt * 3);
        for (int t = 0; t < dnt; ++t) {
            const int cidx = t / 2, half = t & 1;
            const int cx = cidx % gdim, cy = cidx / gdim;
            const float x0 = cx*cell + m, y0 = cy*cell + m;
            const float x1 = (cx+1)*cell - m, y1 = (cy+1)*cell - m;
            float uv[3][2];
            if (half == 0) { uv[0][0]=x0;uv[0][1]=y0; uv[1][0]=x1;uv[1][1]=y0; uv[2][0]=x0;uv[2][1]=y1; }
            else           { uv[0][0]=x1;uv[0][1]=y1; uv[1][0]=x0;uv[1][1]=y1; uv[2][0]=x1;uv[2][1]=y0; }
            const int vs[3] = { dtris[3*t+0], dtris[3*t+1], dtris[3*t+2] };
            for (int k = 0; k < 3; ++k) {
                const int o = 3*t + k, s = vs[k];
                opos[3*o+0]=dverts[3*s+0]; opos[3*o+1]=dverts[3*s+1]; opos[3*o+2]=dverts[3*s+2];
                onrm[3*o+0]=dnrm[3*s+0];   onrm[3*o+1]=dnrm[3*s+1];   onrm[3*o+2]=dnrm[3*s+2];
                ouv[2*o+0]=uv[k][0]; ouv[2*o+1]=uv[k][1];
                oidx[o]=(uint32_t) o;
            }
        }
        GLBLOG("grid atlas %dx%d, %d cells, %d tris", AW, AH, ncell, dnt);
    }
    const uint32_t onv = (uint32_t)(opos.size()/3);
    const uint32_t ntri = (uint32_t)(oidx.size()/3);

    // 3) bake --------------------------------------------------------------
    GLBLOG("building vertex grid + rasterizing %u tris ...", ntri);
    VertexGrid grid;
    if (pbr) grid.build(verts, nv);

    const int NP = AW * AH;
    std::vector<float> bc(NP*3, 0.0f), met(NP, 0.0f), rou(NP, 0.0f);
    std::vector<uint8_t> mask(NP, 0);
    // Defaults for the untextured path / unfilled charts.
    auto set_default = [&](int pix) { bc[3*pix+0]=bc[3*pix+1]=bc[3*pix+2]=0.7f; met[pix]=0.0f; rou[pix]=0.6f; };

    for (uint32_t t = 0; t < ntri; ++t) {
        uint32_t ia = oidx[3*t+0], ib = oidx[3*t+1], ic = oidx[3*t+2];
        float ax=ouv[2*ia+0], ay=ouv[2*ia+1];
        float bx=ouv[2*ib+0], by=ouv[2*ib+1];
        float cx=ouv[2*ic+0], cy=ouv[2*ic+1];
        float area = (bx-ax)*(cy-ay) - (by-ay)*(cx-ax);
        if (std::fabs(area) < 1e-9f) continue;
        float inv_area = 1.0f / area;
        int x0 = std::max(0, (int)std::floor(std::min({ax,bx,cx}) - 1));
        int x1 = std::min(AW-1, (int)std::ceil (std::max({ax,bx,cx}) + 1));
        int y0 = std::max(0, (int)std::floor(std::min({ay,by,cy}) - 1));
        int y1 = std::min(AH-1, (int)std::ceil (std::max({ay,by,cy}) + 1));
        for (int py = y0; py <= y1; ++py) {
            float sy = py + 0.5f;
            for (int px = x0; px <= x1; ++px) {
                float sx = px + 0.5f;
                float w0 = ((bx-sx)*(cy-sy) - (by-sy)*(cx-sx)) * inv_area;
                float w1 = ((cx-sx)*(ay-sy) - (cy-sy)*(ax-sx)) * inv_area;
                float w2 = 1.0f - w0 - w1;
                const float e = -0.001f;
                if (w0 < e || w1 < e || w2 < e) continue;
                int pix = py*AW + px;
                if (!pbr) { set_default(pix); mask[pix]=1; continue; }
                float p[3] = {
                    w0*opos[3*ia+0]+w1*opos[3*ib+0]+w2*opos[3*ic+0],
                    w0*opos[3*ia+1]+w1*opos[3*ib+1]+w2*opos[3*ic+1],
                    w0*opos[3*ia+2]+w1*opos[3*ib+2]+w2*opos[3*ic+2] };
                int nvtx = grid.nearest(p);
                if (nvtx < 0) { set_default(pix); mask[pix]=1; continue; }
                const float * s = pbr + (size_t) nvtx * 5;
                bc[3*pix+0]=s[0]; bc[3*pix+1]=s[1]; bc[3*pix+2]=s[2];
                met[pix]=s[3]; rou[pix]=s[4];
                mask[pix]=1;
            }
        }
    }

    // 4) inpaint (edge-pad dilation) — fill the gutter so bilinear/mip sampling
    // doesn't bleed empty texels across chart seams.
    {
        std::vector<uint8_t> m = mask;
        for (int pass = 0; pass < opt.dilate; ++pass) {
            std::vector<uint8_t> nm = m;
            for (int py = 0; py < AH; ++py)
            for (int px = 0; px < AW; ++px) {
                int pix = py*AW+px;
                if (m[pix]) continue;
                float acc[5]={0,0,0,0,0}; int cnt=0;
                for (int dy=-1; dy<=1; ++dy)
                for (int dx=-1; dx<=1; ++dx) {
                    int qx=px+dx, qy=py+dy;
                    if (qx<0||qx>=AW||qy<0||qy>=AH) continue;
                    int q=qy*AW+qx;
                    if (!m[q]) continue;
                    acc[0]+=bc[3*q+0]; acc[1]+=bc[3*q+1]; acc[2]+=bc[3*q+2]; acc[3]+=met[q]; acc[4]+=rou[q]; ++cnt;
                }
                if (cnt) {
                    bc[3*pix+0]=acc[0]/cnt; bc[3*pix+1]=acc[1]/cnt; bc[3*pix+2]=acc[2]/cnt;
                    met[pix]=acc[3]/cnt; rou[pix]=acc[4]/cnt; nm[pix]=1;
                }
            }
            m.swap(nm);
        }
    }

    // 5) pack channel images. baseColor is an sRGB texture (values written as-is,
    // matching the reference's base_color*255 → baseColorTexture). glTF packs
    // metallic→B, roughness→G (R unused).
    auto to8 = [](float f){ int v=(int)std::lround(f*255.0f); return (uint8_t) std::min(255,std::max(0,v)); };
    std::vector<uint8_t> bc8(NP*3), mr8(NP*3);
    for (int i = 0; i < NP; ++i) {
        bc8[3*i+0]=to8(bc[3*i+0]); bc8[3*i+1]=to8(bc[3*i+1]); bc8[3*i+2]=to8(bc[3*i+2]);
        mr8[3*i+0]=0; mr8[3*i+1]=to8(rou[i]); mr8[3*i+2]=to8(met[i]);
    }
    GLBLOG("inpaint + PNG encode ...");
    std::vector<uint8_t> bc_png, mr_png;
    if (!encode_png(AW, AH, 3, bc8.data(), bc_png) || !encode_png(AW, AH, 3, mr8.data(), mr_png)) {
        err = "PNG encode failed"; return false;
    }

    // 6) final vertex streams with the Trellis→glTF (Y-up) axis convention:
    // (x,y,z) → (x, z, -y); UVs normalised (top-left origin, no V flip).
    std::vector<float> gpos(onv*3), gnrm(onv*3), guv(onv*2);
    for (uint32_t i = 0; i < onv; ++i) {
        gpos[3*i+0]= opos[3*i+0]; gpos[3*i+1]= opos[3*i+2]; gpos[3*i+2]=-opos[3*i+1];
        gnrm[3*i+0]= onrm[3*i+0]; gnrm[3*i+1]= onrm[3*i+2]; gnrm[3*i+2]=-onrm[3*i+1];
        guv[2*i+0]= ouv[2*i+0]/(float)AW; guv[2*i+1]= ouv[2*i+1]/(float)AH;
    }
    write_glb(gpos, gnrm, guv, oidx, bc_png, mr_png, out);
    GLBLOG("GLB %zu bytes (%u verts, %u tris)", out.size(), onv, ntri);
    return true;
}

} // namespace t2glb
