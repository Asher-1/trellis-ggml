#include "trellis2.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <vector>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <random>
#include <unordered_map>

/*****************************************************************************
** Helpers
*****************************************************************************/

namespace {

inline void set_error(std::string * error, const std::string & msg) {
    if (error) *error = msg;
}

// Read a little-endian uint32 from a byte cursor, advancing it.
inline bool read_u32_le(const uint8_t *& p, const uint8_t * end, uint32_t & out) {
    if (p + 4 > end) return false;
    out = (uint32_t) p[0]
        | ((uint32_t) p[1] << 8)
        | ((uint32_t) p[2] << 16)
        | ((uint32_t) p[3] << 24);
    p += 4;
    return true;
}

} // namespace

/*****************************************************************************
** Version
*****************************************************************************/

const char * trellis2_version(void) {
    return TRELLIS2_VERSION;
}

/*****************************************************************************
** .dinodata loader
**
** Binary layout (little-endian), produced by dump_dinodata.py:
**   magic   : 8 bytes  "DINOCOND"
**   version : uint32
**   dtype   : uint32   (0 = float32)   -- only float32 is supported here
**   ndim    : uint32
**   shape   : ndim * uint32            (C-contiguous in this order)
**   payload : prod(shape) * float32    (little-endian)
*****************************************************************************/

bool trellis2_load_dinodata(const std::string & path,
                            trellis2_dino_cond & out,
                            std::string * error) {
    out = trellis2_dino_cond{};

    std::ifstream f(path, std::ios::binary);
    if (!f) {
        set_error(error, "cannot open file: " + path);
        return false;
    }

    // Slurp the whole file — these are a few MB, well within memory.
    std::vector<uint8_t> buf((std::istreambuf_iterator<char>(f)),
                              std::istreambuf_iterator<char>());
    if (buf.size() < 8 + 12) {
        set_error(error, "file too small to contain a .dinodata header");
        return false;
    }

    const uint8_t * p   = buf.data();
    const uint8_t * end = buf.data() + buf.size();

    static const char MAGIC[8] = {'D','I','N','O','C','O','N','D'};
    if (std::memcmp(p, MAGIC, 8) != 0) {
        set_error(error, "bad magic (expected 'DINOCOND')");
        return false;
    }
    p += 8;

    uint32_t version = 0, dtype = 0, ndim = 0;
    if (!read_u32_le(p, end, version) ||
        !read_u32_le(p, end, dtype)   ||
        !read_u32_le(p, end, ndim)) {
        set_error(error, "truncated header");
        return false;
    }

    if (dtype != 0) {
        set_error(error, "unsupported dtype " + std::to_string(dtype) +
                         " (only 0=float32 is supported)");
        return false;
    }
    if (ndim == 0 || ndim > 8) {
        set_error(error, "implausible ndim " + std::to_string(ndim));
        return false;
    }

    std::vector<int64_t> shape(ndim);
    int64_t total = 1;
    for (uint32_t i = 0; i < ndim; ++i) {
        uint32_t dim = 0;
        if (!read_u32_le(p, end, dim)) {
            set_error(error, "truncated shape");
            return false;
        }
        shape[i] = (int64_t) dim;
        total   *= (int64_t) dim;
    }

    const size_t want_bytes = (size_t) total * sizeof(float);
    const size_t have_bytes = (size_t) (end - p);
    if (have_bytes < want_bytes) {
        set_error(error, "payload truncated: have " + std::to_string(have_bytes) +
                         " bytes, need " + std::to_string(want_bytes));
        return false;
    }

    out.shape          = std::move(shape);
    out.format_version = version;
    out.data.resize((size_t) total);
    // Little-endian float32 on the host (all targets we build for are LE).
    std::memcpy(out.data.data(), p, want_bytes);

    return true;
}

/*****************************************************************************
** Fingerprints
*****************************************************************************/

trellis2_dino_fingerprint
trellis2_dino_fingerprints(const trellis2_dino_cond & cond) {
    trellis2_dino_fingerprint fp;
    fp.count = cond.data.size();
    if (cond.data.empty()) {
        return fp;
    }

    float  vmin = std::numeric_limits<float>::infinity();
    float  vmax = -std::numeric_limits<float>::infinity();
    double sum  = 0.0;
    double sumsq = 0.0;
    for (float v : cond.data) {
        if (v < vmin) vmin = v;
        if (v > vmax) vmax = v;
        sum   += (double) v;
        sumsq += (double) v * (double) v;
    }

    fp.vmin = vmin;
    fp.vmax = vmax;
    fp.sum  = sum;
    fp.mean = sum / (double) fp.count;
    fp.l2   = std::sqrt(sumsq);
    return fp;
}

/*****************************************************************************
** Sparse-structure flow DiT (stage 1) — GGUF loader
*****************************************************************************/

struct trellis2_ss_flow_model {
    gguf_context * gguf = nullptr;
    ggml_context * ctx  = nullptr;
    trellis2_ss_flow_hparams hp;
    bool has_data = false; // true if weight payloads were read (load_tensors)

    // Compute backend (auto-selected: GPU if available, else CPU) and the
    // buffer holding the weights on that backend. Only set when has_data.
    ggml_backend_t        backend     = nullptr;
    ggml_backend_buffer_t weights_buf = nullptr;
    std::string           backend_name;

    // name -> tensor (into ctx); built once at load for O(1) graph wiring later.
    std::unordered_map<std::string, ggml_tensor *> tensors;
};

namespace {

// Pick the best available compute backend: the first GPU device exposed by the
// ggml backend registry (CUDA / Metal / Vulkan / ...), falling back to CPU.
// Mirrors sam3.cpp's "use a GPU backend automatically if one is available".
ggml_backend_t init_best_backend(std::string & name_out) {
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            ggml_backend_t b = ggml_backend_dev_init(dev, nullptr);
            if (b) {
                const char * d = ggml_backend_dev_description(dev);
                name_out = d ? d : ggml_backend_dev_name(dev);
                return b;
            }
        }
    }
    name_out = "CPU";
    return ggml_backend_cpu_init();
}

// KV readers with defaults (return the default if the key is absent).
uint32_t kv_u32(const gguf_context * g, const char * key, uint32_t def) {
    const int64_t id = gguf_find_key(g, key);
    return id < 0 ? def : gguf_get_val_u32(g, id);
}
float kv_f32(const gguf_context * g, const char * key, float def) {
    const int64_t id = gguf_find_key(g, key);
    return id < 0 ? def : gguf_get_val_f32(g, id);
}
bool kv_bool(const gguf_context * g, const char * key, bool def) {
    const int64_t id = gguf_find_key(g, key);
    return id < 0 ? def : gguf_get_val_bool(g, id);
}
const char * kv_str(const gguf_context * g, const char * key, const char * def) {
    const int64_t id = gguf_find_key(g, key);
    return id < 0 ? def : gguf_get_val_str(g, id);
}

} // namespace

trellis2_ss_flow_model *
trellis2_ss_flow_load(const std::string & path, bool load_tensors, std::string * error) {
    auto * m = new trellis2_ss_flow_model();

    // Always parse metadata only; the weights are then allocated on the chosen
    // backend and the payloads streamed in from the file (so the GPU can use
    // them directly). This is the standard llama.cpp / stable-diffusion.cpp path.
    gguf_init_params params;
    params.no_alloc = true;
    params.ctx      = &m->ctx;

    m->gguf = gguf_init_from_file(path.c_str(), params);
    if (!m->gguf) {
        set_error(error, "gguf_init_from_file failed (not a GGUF file?): " + path);
        delete m;
        return nullptr;
    }

    // Sanity-check the architecture tag.
    const char * arch = kv_str(m->gguf, "general.architecture", "");
    if (std::strcmp(arch, "trellis2-ss-flow") != 0) {
        set_error(error, std::string("unexpected architecture '") + arch +
                         "' (expected 'trellis2-ss-flow')");
        trellis2_ss_flow_free(m);
        return nullptr;
    }

    trellis2_ss_flow_hparams & hp = m->hp;
    const char * P = "trellis2.ss_flow.";
    auto K = [&](const char * suffix) { return std::string(P) + suffix; };

    hp.resolution        = (int32_t) kv_u32 (m->gguf, K("resolution").c_str(),     0);
    hp.in_channels       = (int32_t) kv_u32 (m->gguf, K("in_channels").c_str(),    0);
    hp.out_channels      = (int32_t) kv_u32 (m->gguf, K("out_channels").c_str(),   0);
    hp.model_channels    = (int32_t) kv_u32 (m->gguf, K("model_channels").c_str(), 0);
    hp.cond_channels     = (int32_t) kv_u32 (m->gguf, K("cond_channels").c_str(),  0);
    hp.num_blocks        = (int32_t) kv_u32 (m->gguf, K("num_blocks").c_str(),     0);
    hp.num_heads         = (int32_t) kv_u32 (m->gguf, K("num_heads").c_str(),      0);
    hp.mlp_ratio         =           kv_f32 (m->gguf, K("mlp_ratio").c_str(),      0.0f);
    hp.share_mod         =           kv_bool(m->gguf, K("share_mod").c_str(),         false) ? 1 : 0;
    hp.qk_rms_norm       =           kv_bool(m->gguf, K("qk_rms_norm").c_str(),       false) ? 1 : 0;
    hp.qk_rms_norm_cross =           kv_bool(m->gguf, K("qk_rms_norm_cross").c_str(), false) ? 1 : 0;
    hp.rope_freq_min     =           kv_f32 (m->gguf, K("rope_freq_min").c_str(),  1.0f);
    hp.rope_freq_base    =           kv_f32 (m->gguf, K("rope_freq_base").c_str(), 10000.0f);
    hp.file_type         = (int32_t) kv_u32 (m->gguf, "general.file_type", 0);
    std::snprintf(hp.pe_mode, sizeof(hp.pe_mode), "%s",
                  kv_str(m->gguf, K("pe_mode").c_str(), "rope"));

    // Build name -> tensor map.
    for (ggml_tensor * t = ggml_get_first_tensor(m->ctx); t != nullptr;
         t = ggml_get_next_tensor(m->ctx, t)) {
        m->tensors[t->name] = t;
    }

    if (load_tensors) {
        // Allocate all weights on the auto-selected backend, then stream the
        // payloads from the file into that buffer.
        m->backend = init_best_backend(m->backend_name);
        m->weights_buf = ggml_backend_alloc_ctx_tensors(m->ctx, m->backend);
        if (!m->weights_buf) {
            set_error(error, "failed to allocate weights on backend " + m->backend_name);
            trellis2_ss_flow_free(m);
            return nullptr;
        }

        std::ifstream fin(path, std::ios::binary);
        if (!fin) {
            set_error(error, "cannot reopen file for weight data: " + path);
            trellis2_ss_flow_free(m);
            return nullptr;
        }
        const size_t data_off = gguf_get_data_offset(m->gguf);
        const int64_t nt = gguf_get_n_tensors(m->gguf);
        std::vector<uint8_t> buf;
        for (int64_t i = 0; i < nt; ++i) {
            const char * name = gguf_get_tensor_name(m->gguf, i);
            ggml_tensor * t = m->tensors[name];
            const size_t nb  = ggml_nbytes(t);
            const size_t off = data_off + gguf_get_tensor_offset(m->gguf, i);
            buf.resize(nb);
            fin.seekg((std::streamoff) off, std::ios::beg);
            if (!fin.read(reinterpret_cast<char *>(buf.data()), (std::streamsize) nb)) {
                set_error(error, std::string("failed reading weight '") + name + "' from file");
                trellis2_ss_flow_free(m);
                return nullptr;
            }
            ggml_backend_tensor_set(t, buf.data(), 0, nb);
        }
        m->has_data = true;
    }

    return m;
}

void trellis2_ss_flow_free(trellis2_ss_flow_model * m) {
    if (!m) return;
    if (m->weights_buf) ggml_backend_buffer_free(m->weights_buf);
    if (m->backend)     ggml_backend_free(m->backend);
    if (m->gguf)        gguf_free(m->gguf);
    if (m->ctx)         ggml_free(m->ctx);
    delete m;
}

const char * trellis2_ss_flow_backend_name(const trellis2_ss_flow_model * m) {
    return (m && !m->backend_name.empty()) ? m->backend_name.c_str() : "none";
}

const trellis2_ss_flow_hparams &
trellis2_ss_flow_hparams_of(const trellis2_ss_flow_model * m) {
    return m->hp;
}

int trellis2_ss_flow_n_tensors(const trellis2_ss_flow_model * m) {
    return m ? (int) gguf_get_n_tensors(m->gguf) : 0;
}

bool trellis2_ss_flow_get_tensor_info(const trellis2_ss_flow_model * m,
                                      int i, trellis2_tensor_info & out) {
    if (!m || i < 0 || i >= (int) gguf_get_n_tensors(m->gguf)) return false;
    const char * name = gguf_get_tensor_name(m->gguf, i);
    out.name = name;

    ggml_tensor * t = ggml_get_tensor(m->ctx, name);
    if (!t) return false;
    out.n_dims    = ggml_n_dims(t);
    for (int d = 0; d < 4; ++d) out.ne[d] = t->ne[d];
    out.ggml_type = (int) t->type;
    out.type_name = ggml_type_name(t->type);
    out.n_bytes   = ggml_nbytes(t);
    return true;
}

bool trellis2_ss_flow_has_tensor(const trellis2_ss_flow_model * m,
                                 const std::string & name) {
    return m && m->tensors.find(name) != m->tensors.end();
}

/*****************************************************************************
** Sparse-structure flow DiT — forward pass (CPU backend)
**
** Mirrors SparseStructureFlowModel.forward + ModulatedTransformerCrossBlock:
**   h = input_layer(x)                                  # [C, N]
**   t_emb = adaLN(SiLU stack)(timestep_embedding(t))    # [6C] shared modulation
**   for each of num_blocks cross-blocks:
**     (shift/scale/gate)_{msa,mlp} = modulation_b + t_emb
**     h += gate_msa * self_attn( modulate(LN0(h)) )     # RoPE + QK-RMSNorm
**     h += cross_attn( LN1_affine(h), cond )            # QK-RMSNorm, no RoPE
**     h += gate_mlp * mlp( modulate(LN2(h)) )           # GELU-tanh FFN
**   out = out_layer(LayerNorm(h))                       # [out_channels, N]
*****************************************************************************/

namespace {

// Sinusoidal timestep embedding (cos|sin), matching TimestepEmbedder.
std::vector<float> timestep_embedding(float t, int dim) {
    std::vector<float> e((size_t) dim, 0.0f);
    const int half = dim / 2;
    for (int i = 0; i < half; ++i) {
        const float freq = std::exp(-std::log(10000.0f) * (float) i / (float) half);
        const float arg  = t * freq;
        e[i]        = std::cos(arg);
        e[half + i] = std::sin(arg);
    }
    return e;  // dim is even here (256) so no padding needed
}

// Precompute the interleaved 3D-RoPE cos/sin tables for an R^3 grid.
// Layout matches q reshaped to [head_dim, n_heads, N]: cos/sin are [head_dim, 1, N]
// with cos[n*head_dim + 2p] == cos[n*head_dim + 2p+1] == cos(theta_p(n)).
void rope_tables(int res, int head_dim, float freq_min, float freq_base,
                 std::vector<float> & cos_t, std::vector<float> & sin_t) {
    const int dim      = 3;                         // 3 spatial axes
    const int freq_dim = head_dim / 2 / dim;        // 21 for head_dim 128
    const int N        = res * res * res;

    std::vector<float> freqs((size_t) freq_dim);
    for (int mi = 0; mi < freq_dim; ++mi) {
        freqs[mi] = freq_min / std::pow(freq_base, (float) mi / (float) freq_dim);
    }

    cos_t.assign((size_t) head_dim * N, 1.0f);
    sin_t.assign((size_t) head_dim * N, 0.0f);
    const int pairs = head_dim / 2;                 // 64
    for (int n = 0; n < N; ++n) {
        const int coord[3] = { n / (res * res), (n / res) % res, n % res };
        for (int p = 0; p < pairs; ++p) {
            float theta = 0.0f;                     // p == 63 -> pad (theta 0)
            if (p < dim * freq_dim) {               // p in 0..62
                theta = (float) coord[p / freq_dim] * freqs[p % freq_dim];
            }
            const size_t base = (size_t) n * head_dim + (size_t) 2 * p;
            cos_t[base] = cos_t[base + 1] = std::cos(theta);
            sin_t[base] = sin_t[base + 1] = std::sin(theta);
        }
    }
}

} // namespace

bool trellis2_ss_flow_forward(trellis2_ss_flow_model * m,
                              const float * x, float t,
                              const float * cond, int cond_tokens, int cond_channels,
                              float * out, std::string * error) {
    if (!m)            { set_error(error, "null model");        return false; }
    if (!m->has_data)  { set_error(error, "model loaded metadata-only; reload with load_tensors=true"); return false; }

    const trellis2_ss_flow_hparams & hp = m->hp;
    if (std::strcmp(hp.pe_mode, "rope") != 0) { set_error(error, "only pe_mode=rope is implemented"); return false; }
    if (!hp.share_mod)                         { set_error(error, "only share_mod=true is implemented"); return false; }
    if (cond_channels != hp.cond_channels)     { set_error(error, "cond_channels mismatch"); return false; }

    const int   C   = hp.model_channels;     // 1536
    const int   R   = hp.resolution;         // 16
    const int   N   = R * R * R;             // 4096 tokens
    const int   H   = hp.num_heads;          // 12
    const int   hd  = hp.head_dim();         // 128
    const int   Lkv = cond_tokens;           // 1029
    const float attn_scale = 1.0f / std::sqrt((float) hd);

    std::string missing;
    auto W = [&](const std::string & n) -> ggml_tensor * {
        auto it = m->tensors.find(n);
        if (it == m->tensors.end()) { if (missing.empty()) missing = n; return nullptr; }
        return it->second;
    };

    // ── compute graph context (metadata only; gallocr allocates data) ────────
    const size_t mem = ggml_tensor_overhead() * 32768 + ggml_graph_overhead_custom(32768, false);
    ggml_init_params ip{ mem, nullptr, /*no_alloc*/ true };
    ggml_context * ctx = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 32768, false);

    // ── input leaves ─────────────────────────────────────────────────────────
    ggml_tensor * x_t   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, hp.in_channels); // channel-major [N, Cin]
    ggml_tensor * temb  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 256);
    ggml_tensor * cos_t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hd, 1, N);
    ggml_tensor * sin_t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hd, 1, N);
    ggml_tensor * cnd   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cond_channels, Lkv); // [ctx_ch, Lkv]
    ggml_set_input(x_t);  ggml_set_name(x_t, "x");
    ggml_set_input(temb); ggml_set_name(temb, "temb");
    ggml_set_input(cos_t);
    ggml_set_input(sin_t);
    ggml_set_input(cnd);

    auto lin = [&](ggml_tensor * in, const std::string & pfx) -> ggml_tensor * {
        ggml_tensor * y = ggml_mul_mat(ctx, W(pfx + ".weight"), in);
        ggml_tensor * b = W(pfx + ".bias");
        if (b) y = ggml_add(ctx, y, b);
        return y;
    };
    // h * (1 + scale) + shift, broadcasting the [C] vectors over tokens.
    auto modulate = [&](ggml_tensor * h, ggml_tensor * scale, ggml_tensor * shift) {
        return ggml_add(ctx, ggml_add(ctx, ggml_mul(ctx, h, scale), h), shift);
    };
    // interleaved RoPE on a [hd, H, N] tensor using the cos/sin tables.
    auto rope = [&](ggml_tensor * q3) -> ggml_tensor * {
        ggml_tensor * q4 = ggml_reshape_4d(ctx, q3, 2, hd / 2, H, N);
        ggml_tensor * q0 = ggml_cont(ctx, ggml_view_4d(ctx, q4, 1, hd / 2, H, N,
                                                       q4->nb[1], q4->nb[2], q4->nb[3], 0));
        ggml_tensor * q1 = ggml_cont(ctx, ggml_view_4d(ctx, q4, 1, hd / 2, H, N,
                                                       q4->nb[1], q4->nb[2], q4->nb[3], q4->nb[0]));
        ggml_tensor * swap = ggml_concat(ctx, ggml_neg(ctx, q1), q0, 0);   // [2,hd/2,H,N]
        swap = ggml_reshape_3d(ctx, swap, hd, H, N);
        return ggml_add(ctx, ggml_mul(ctx, q3, cos_t), ggml_mul(ctx, swap, sin_t));
    };
    // QK-RMSNorm: F.normalize(x)*gamma*sqrt(hd) == rms_norm(x)*gamma (sqrt cancels).
    auto qk_norm = [&](ggml_tensor * v3, const std::string & gname) {
        return ggml_mul(ctx, ggml_rms_norm(ctx, v3, 1e-12f), W(gname));
    };
    // scaled-dot-product attention; q3/k3/v3 are [hd, H, Lq]/[hd, H, Lk].
    auto sdpa = [&](ggml_tensor * q3, ggml_tensor * k3, ggml_tensor * v3) -> ggml_tensor * {
        ggml_tensor * qp = ggml_cont(ctx, ggml_permute(ctx, q3, 0, 2, 1, 3)); // [hd, Lq, H]
        ggml_tensor * kp = ggml_cont(ctx, ggml_permute(ctx, k3, 0, 2, 1, 3)); // [hd, Lk, H]
        ggml_tensor * vp = ggml_cont(ctx, ggml_permute(ctx, v3, 0, 2, 1, 3)); // [hd, Lk, H]
        ggml_tensor * sc = ggml_mul_mat(ctx, kp, qp);                         // [Lk, Lq, H]
        sc = ggml_soft_max_ext(ctx, sc, nullptr, attn_scale, 0.0f);
        ggml_tensor * vt = ggml_cont(ctx, ggml_permute(ctx, vp, 1, 0, 2, 3)); // [Lk, hd, H]
        ggml_tensor * o  = ggml_mul_mat(ctx, vt, sc);                         // [hd, Lq, H]
        o = ggml_cont(ctx, ggml_permute(ctx, o, 0, 2, 1, 3));                 // [hd, H, Lq]
        return ggml_reshape_2d(ctx, o, C, o->ne[2]);                          // [C, Lq]
    };

    const size_t es = sizeof(float);

    // ── stem: input projection (+ no additive PE in rope mode) ───────────────
    ggml_tensor * h = ggml_cont(ctx, ggml_transpose(ctx, x_t));   // [Cin, N]
    h = lin(h, "input_layer");                                    // [C, N]

    // ── shared modulation from the timestep ──────────────────────────────────
    ggml_tensor * te = lin(temb, "t_embedder.mlp.0");
    te = ggml_silu(ctx, te);
    te = lin(te, "t_embedder.mlp.2");                             // [C]
    ggml_tensor * tmod = lin(ggml_silu(ctx, te), "adaLN_modulation.1"); // [6C]

    ggml_tensor * cond_h = cnd;                                   // [ctx_ch, Lkv]

    for (int b = 0; b < hp.num_blocks; ++b) {
        const std::string blk = "blocks." + std::to_string(b);
        ggml_tensor * mods = ggml_add(ctx, W(blk + ".modulation"), tmod); // [6C]
        auto chunk = [&](int idx) {
            return ggml_view_1d(ctx, mods, C, (size_t) idx * C * es);
        };
        ggml_tensor * shift_msa = chunk(0), * scale_msa = chunk(1), * gate_msa = chunk(2);
        ggml_tensor * shift_mlp = chunk(3), * scale_mlp = chunk(4), * gate_mlp = chunk(5);

        // self-attention (norm1 affine-free, modulated; RoPE + QK-RMSNorm)
        ggml_tensor * hn = modulate(ggml_norm(ctx, h, 1e-6f), scale_msa, shift_msa);
        ggml_tensor * qkv = lin(hn, blk + ".self_attn.to_qkv");           // [3C, N]
        ggml_tensor * q = ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_view_2d(ctx, qkv, C, N, qkv->nb[1], 0)),       hd, H, N);
        ggml_tensor * k = ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_view_2d(ctx, qkv, C, N, qkv->nb[1], (size_t)C*es)),   hd, H, N);
        ggml_tensor * v = ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_view_2d(ctx, qkv, C, N, qkv->nb[1], (size_t)2*C*es)), hd, H, N);
        q = rope(qk_norm(q, blk + ".self_attn.q_rms_norm.gamma"));
        k = rope(qk_norm(k, blk + ".self_attn.k_rms_norm.gamma"));
        ggml_tensor * sa = lin(sdpa(q, k, v), blk + ".self_attn.to_out");
        h = ggml_add(ctx, h, ggml_mul(ctx, sa, gate_msa));

        // cross-attention (norm2 affine; QK-RMSNorm, no RoPE, no gate)
        ggml_tensor * h2 = ggml_norm(ctx, h, 1e-6f);
        h2 = ggml_add(ctx, ggml_mul(ctx, h2, W(blk + ".norm2.weight")), W(blk + ".norm2.bias"));
        ggml_tensor * cq = ggml_reshape_3d(ctx, lin(h2, blk + ".cross_attn.to_q"), hd, H, N);
        cq = qk_norm(cq, blk + ".cross_attn.q_rms_norm.gamma");
        ggml_tensor * kv = lin(cond_h, blk + ".cross_attn.to_kv");        // [2C, Lkv]
        ggml_tensor * ck = ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_view_2d(ctx, kv, C, Lkv, kv->nb[1], 0)),         hd, H, Lkv);
        ggml_tensor * cv = ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_view_2d(ctx, kv, C, Lkv, kv->nb[1], (size_t)C*es)), hd, H, Lkv);
        ck = qk_norm(ck, blk + ".cross_attn.k_rms_norm.gamma");
        ggml_tensor * ca = lin(sdpa(cq, ck, cv), blk + ".cross_attn.to_out");
        h = ggml_add(ctx, h, ca);

        // feed-forward (norm3 affine-free, modulated; GELU-tanh)
        ggml_tensor * hm = modulate(ggml_norm(ctx, h, 1e-6f), scale_mlp, shift_mlp);
        hm = lin(hm, blk + ".mlp.mlp.0");
        hm = ggml_gelu(ctx, hm);
        hm = lin(hm, blk + ".mlp.mlp.2");
        h = ggml_add(ctx, h, ggml_mul(ctx, hm, gate_mlp));
    }

    // ── head: affine-free LayerNorm (eps 1e-5) + output projection ────────────
    h = ggml_norm(ctx, h, 1e-5f);
    h = lin(h, "out_layer");                                      // [out_channels, N]
    ggml_tensor * y = ggml_cont(ctx, ggml_transpose(ctx, h));     // [N, out_channels], channel-major
    ggml_set_output(y);

    if (!missing.empty()) {
        set_error(error, "missing tensor: " + missing);
        ggml_free(ctx);
        return false;
    }

    ggml_build_forward_expand(gf, y);

    // ── allocate + run on the model's backend (GPU if available, else CPU) ────
    ggml_backend_t backend = m->backend;
    ggml_gallocr_t alloc   = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(alloc, gf)) {
        set_error(error, "ggml_gallocr_alloc_graph failed");
        ggml_gallocr_free(alloc); ggml_free(ctx);
        return false;
    }

    std::vector<float> emb = timestep_embedding(t, 256);
    std::vector<float> cosv, sinv;
    rope_tables(R, hd, hp.rope_freq_min, hp.rope_freq_base, cosv, sinv);

    ggml_backend_tensor_set(x_t,   x,           0, (size_t) hp.in_channels * N * es);
    ggml_backend_tensor_set(temb,  emb.data(),  0, emb.size() * es);
    ggml_backend_tensor_set(cos_t, cosv.data(), 0, cosv.size() * es);
    ggml_backend_tensor_set(sin_t, sinv.data(), 0, sinv.size() * es);
    ggml_backend_tensor_set(cnd,   cond,        0, (size_t) cond_channels * Lkv * es);

    const ggml_status st = ggml_backend_graph_compute(backend, gf);
    bool ok = (st == GGML_STATUS_SUCCESS);
    if (ok) {
        ggml_backend_tensor_get(y, out, 0, (size_t) hp.out_channels * N * es);
    } else {
        set_error(error, "graph compute failed");
    }

    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    return ok;
}

/*****************************************************************************
** Flow-Euler sampler (classifier-free guidance + interval + rescale)
**
** Mirrors FlowEulerGuidanceIntervalSampler.sample. All the per-step flow
** arithmetic is elementwise on the latent and runs on the host; only the
** velocity prediction (1 or 2 forwards per step) uses the GPU graph above.
*****************************************************************************/

namespace {

// x_0 estimate from a velocity prediction: (1-s)x_t - (s + (1-s)t) pred.
inline void pred_to_xstart(const std::vector<float> & x_t, double t, double sm,
                           const std::vector<float> & pred, std::vector<float> & x0) {
    const double a = 1.0 - sm;
    const double b = sm + a * t;
    for (size_t i = 0; i < x_t.size(); ++i) x0[i] = (float) (a * x_t[i] - b * pred[i]);
}

// inverse of pred_to_xstart.
inline void xstart_to_pred(const std::vector<float> & x_t, double t, double sm,
                           const std::vector<float> & x0, std::vector<float> & pred) {
    const double a = 1.0 - sm;
    const double b = sm + a * t;
    for (size_t i = 0; i < x_t.size(); ++i) pred[i] = (float) ((a * x_t[i] - x0[i]) / b);
}

// unbiased std over a whole buffer (matches torch .std(), correction=1).
double unbiased_std(const std::vector<float> & v) {
    const size_t n = v.size();
    if (n < 2) return 0.0;
    double sum = 0.0;
    for (float x : v) sum += x;
    const double mean = sum / (double) n;
    double ss = 0.0;
    for (float x : v) { const double d = (double) x - mean; ss += d * d; }
    return std::sqrt(ss / (double) (n - 1));
}

} // namespace

bool trellis2_ss_flow_sample(trellis2_ss_flow_model * m,
                             const float * cond, int cond_tokens, int cond_channels,
                             const trellis2_ss_sampler_params * params_in,
                             const float * noise,
                             float * out_latent, std::string * error) {
    if (!m)           { set_error(error, "null model"); return false; }
    if (!m->has_data) { set_error(error, "model loaded metadata-only; reload with load_tensors=true"); return false; }

    trellis2_ss_sampler_params P;
    if (params_in) P = *params_in;

    const trellis2_ss_flow_hparams & hp = m->hp;
    const int    R   = hp.resolution;
    const size_t N   = (size_t) R * R * R;
    const size_t n   = (size_t) hp.in_channels * N;
    const double sm  = P.sigma_min;

    // ── initial noise ─────────────────────────────────────────────────────────
    std::vector<float> x_t(n);
    if (noise) {
        std::memcpy(x_t.data(), noise, n * sizeof(float));
    } else {
        std::mt19937_64 rng(P.seed);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        for (size_t i = 0; i < n; ++i) x_t[i] = nd(rng);
    }

    // ── timestep schedule: linspace(1,0,steps+1) warped by rescale_t ─────────
    std::vector<double> ts((size_t) P.steps + 1);
    for (int i = 0; i <= P.steps; ++i) {
        const double lin = 1.0 - (double) i / (double) P.steps;            // 1 -> 0
        ts[i] = P.rescale_t * lin / (1.0 + (P.rescale_t - 1.0) * lin);
    }

    const std::vector<float> zero_cond((size_t) cond_tokens * cond_channels, 0.0f);
    std::vector<float> pred_pos(n), pred_neg(n), pred_v(n), x0_pos(n), x0_cfg(n);

    auto fwd = [&](double t, const float * c, std::vector<float> & dst) -> bool {
        return trellis2_ss_flow_forward(m, x_t.data(), (float) (1000.0 * t),
                                        c, cond_tokens, cond_channels, dst.data(), error);
    };

    for (int i = 0; i < P.steps; ++i) {
        const double t = ts[i], t_prev = ts[i + 1];
        const bool in_interval = (t >= P.guidance_interval_min && t <= P.guidance_interval_max);
        const float gs = in_interval ? P.guidance_strength : 1.0f;

        if (gs == 1.0f) {
            if (!fwd(t, cond, pred_v)) return false;
        } else if (gs == 0.0f) {
            if (!fwd(t, zero_cond.data(), pred_v)) return false;
        } else {
            if (!fwd(t, cond, pred_pos)) return false;
            if (!fwd(t, zero_cond.data(), pred_neg)) return false;
            for (size_t k = 0; k < n; ++k) pred_v[k] = gs * pred_pos[k] + (1.0f - gs) * pred_neg[k];

            if (P.guidance_rescale > 0.0f) {
                pred_to_xstart(x_t, t, sm, pred_pos, x0_pos);
                pred_to_xstart(x_t, t, sm, pred_v,   x0_cfg);
                const double std_pos = unbiased_std(x0_pos);
                const double std_cfg = unbiased_std(x0_cfg);
                const double ratio = (std_cfg != 0.0) ? std_pos / std_cfg : 1.0;
                const float  gr = P.guidance_rescale;
                for (size_t k = 0; k < n; ++k) {
                    const double rescaled = x0_cfg[k] * ratio;
                    x0_cfg[k] = (float) (gr * rescaled + (1.0 - gr) * x0_cfg[k]);
                }
                xstart_to_pred(x_t, t, sm, x0_cfg, pred_v);
            }
        }

        // Euler step: x_{t-1} = x_t - (t - t_prev) * v
        const double dt = t - t_prev;
        for (size_t k = 0; k < n; ++k) x_t[k] = (float) (x_t[k] - dt * pred_v[k]);

        if (P.verbose) {
            std::fprintf(stderr, "\r[ss sample] step %2d/%d  t=%.4f->%.4f  %s   ",
                         i + 1, P.steps, t, t_prev, in_interval ? "cfg" : "uncond");
            std::fflush(stderr);
        }
    }
    if (P.verbose) std::fprintf(stderr, "\n");

    std::memcpy(out_latent, x_t.data(), n * sizeof(float));
    return true;
}

/*****************************************************************************
** Sparse-structure decoder (stage 1): SparseStructureDecoder
**
**   h = input_layer(z_s)                         # Conv3d latent->channels[0]
**   h = middle_block(h)                          # num_res_blocks_middle ResBlocks
**   for level i in 0..n_levels-1:
**       h = ResBlock x num_res_blocks            # at channels[i]
**       if i < n_levels-1: h = Upsample(h)       # Conv3d (C->C'*8) + pixel_shuffle_3d
**   logits = out_layer(h)                        # ChannelLayerNorm + SiLU + Conv3d->out
**
** ResBlock3d: x + conv2(silu(norm2(conv1(silu(norm1(x)))))), all skips Identity
** here (in==out at every block). norm is a per-voxel LayerNorm over channels.
** Two pixel-shuffle upsamples take 16^3 -> 32^3 -> 64^3.
*****************************************************************************/

struct trellis2_ss_dec_model {
    gguf_context * gguf = nullptr;
    ggml_context * ctx  = nullptr;
    trellis2_ss_dec_hparams hp;
    bool has_data = false;

    ggml_backend_t        backend     = nullptr;
    ggml_backend_buffer_t weights_buf = nullptr;
    std::string           backend_name;

    std::unordered_map<std::string, ggml_tensor *> tensors;
};

trellis2_ss_dec_model *
trellis2_ss_dec_load(const std::string & path, bool load_tensors, std::string * error) {
    auto * m = new trellis2_ss_dec_model();

    gguf_init_params params;
    params.no_alloc = true;
    params.ctx      = &m->ctx;

    m->gguf = gguf_init_from_file(path.c_str(), params);
    if (!m->gguf) {
        set_error(error, "gguf_init_from_file failed (not a GGUF file?): " + path);
        delete m;
        return nullptr;
    }

    const char * arch = kv_str(m->gguf, "general.architecture", "");
    if (std::strcmp(arch, "trellis2-ss-dec") != 0) {
        set_error(error, std::string("unexpected architecture '") + arch +
                         "' (expected 'trellis2-ss-dec')");
        trellis2_ss_dec_free(m);
        return nullptr;
    }

    trellis2_ss_dec_hparams & hp = m->hp;
    const char * P = "trellis2.ss_dec.";
    auto K = [&](const char * suffix) { return std::string(P) + suffix; };

    hp.out_channels          = (int32_t) kv_u32(m->gguf, K("out_channels").c_str(),          1);
    hp.latent_channels       = (int32_t) kv_u32(m->gguf, K("latent_channels").c_str(),       8);
    hp.num_res_blocks        = (int32_t) kv_u32(m->gguf, K("num_res_blocks").c_str(),        2);
    hp.num_res_blocks_middle = (int32_t) kv_u32(m->gguf, K("num_res_blocks_middle").c_str(), 2);
    hp.n_levels              = (int32_t) kv_u32(m->gguf, K("n_levels").c_str(),              3);
    hp.norm_eps              =           kv_f32(m->gguf, K("norm_eps").c_str(),           1e-5f);
    hp.file_type             = (int32_t) kv_u32(m->gguf, "general.file_type", 0);
    std::snprintf(hp.norm_type, sizeof(hp.norm_type), "%s",
                  kv_str(m->gguf, K("norm_type").c_str(), "layer"));
    if (hp.n_levels > 8) hp.n_levels = 8;
    for (int i = 0; i < hp.n_levels; ++i) {
        hp.channels[i] = (int32_t) kv_u32(m->gguf, K(("channels." + std::to_string(i)).c_str()).c_str(), 0);
    }

    for (ggml_tensor * t = ggml_get_first_tensor(m->ctx); t != nullptr;
         t = ggml_get_next_tensor(m->ctx, t)) {
        m->tensors[t->name] = t;
    }

    if (load_tensors) {
        m->backend = init_best_backend(m->backend_name);
        m->weights_buf = ggml_backend_alloc_ctx_tensors(m->ctx, m->backend);
        if (!m->weights_buf) {
            set_error(error, "failed to allocate weights on backend " + m->backend_name);
            trellis2_ss_dec_free(m);
            return nullptr;
        }
        std::ifstream fin(path, std::ios::binary);
        if (!fin) {
            set_error(error, "cannot reopen file for weight data: " + path);
            trellis2_ss_dec_free(m);
            return nullptr;
        }
        const size_t data_off = gguf_get_data_offset(m->gguf);
        const int64_t nt = gguf_get_n_tensors(m->gguf);
        std::vector<uint8_t> buf;
        for (int64_t i = 0; i < nt; ++i) {
            const char * name = gguf_get_tensor_name(m->gguf, i);
            ggml_tensor * t = m->tensors[name];
            const size_t nb  = ggml_nbytes(t);
            const size_t off = data_off + gguf_get_tensor_offset(m->gguf, i);
            buf.resize(nb);
            fin.seekg((std::streamoff) off, std::ios::beg);
            if (!fin.read(reinterpret_cast<char *>(buf.data()), (std::streamsize) nb)) {
                set_error(error, std::string("failed reading weight '") + name + "' from file");
                trellis2_ss_dec_free(m);
                return nullptr;
            }
            ggml_backend_tensor_set(t, buf.data(), 0, nb);
        }
        m->has_data = true;
    }

    return m;
}

void trellis2_ss_dec_free(trellis2_ss_dec_model * m) {
    if (!m) return;
    if (m->weights_buf) ggml_backend_buffer_free(m->weights_buf);
    if (m->backend)     ggml_backend_free(m->backend);
    if (m->gguf)        gguf_free(m->gguf);
    if (m->ctx)         ggml_free(m->ctx);
    delete m;
}

const char * trellis2_ss_dec_backend_name(const trellis2_ss_dec_model * m) {
    return (m && !m->backend_name.empty()) ? m->backend_name.c_str() : "none";
}

const trellis2_ss_dec_hparams &
trellis2_ss_dec_hparams_of(const trellis2_ss_dec_model * m) { return m->hp; }

int trellis2_ss_dec_n_tensors(const trellis2_ss_dec_model * m) {
    return m ? (int) gguf_get_n_tensors(m->gguf) : 0;
}

bool trellis2_ss_dec_get_tensor_info(const trellis2_ss_dec_model * m,
                                     int i, trellis2_tensor_info & out) {
    if (!m || i < 0 || i >= (int) gguf_get_n_tensors(m->gguf)) return false;
    const char * name = gguf_get_tensor_name(m->gguf, i);
    out.name = name;
    ggml_tensor * t = ggml_get_tensor(m->ctx, name);
    if (!t) return false;
    out.n_dims = ggml_n_dims(t);
    for (int d = 0; d < 4; ++d) out.ne[d] = t->ne[d];
    out.ggml_type = (int) t->type;
    out.type_name = ggml_type_name(t->type);
    out.n_bytes   = ggml_nbytes(t);
    return true;
}

bool trellis2_ss_dec_decode(trellis2_ss_dec_model * m,
                            const float * latent, float * out, std::string * error) {
    if (!m)           { set_error(error, "null model"); return false; }
    if (!m->has_data) { set_error(error, "model loaded metadata-only; reload with load_tensors=true"); return false; }

    const trellis2_ss_dec_hparams & hp = m->hp;
    const int    R    = hp.res_in();                  // 16
    const int    Cin  = hp.latent_channels;           // 8
    const float  eps  = hp.norm_eps;                  // 1e-5
    const size_t es   = sizeof(float);

    std::string missing;
    auto W = [&](const std::string & n) -> ggml_tensor * {
        auto it = m->tensors.find(n);
        if (it == m->tensors.end()) { if (missing.empty()) missing = n; return nullptr; }
        return it->second;
    };

    const size_t mem = ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom(8192, false);
    ggml_init_params ip{ mem, nullptr, /*no_alloc*/ true };
    ggml_context * ctx = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8192, false);

    // input leaf: z_s as ggml [R,R,R,Cin] (ne0=k, ne1=j, ne2=i, ne3=channel) —
    // identical layout to channel-major latent[c*R^3 + i*R^2 + j*R + k].
    ggml_tensor * x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, R, R, R, Cin);
    ggml_set_input(x);
    ggml_set_name(x, "z_s");

    // Conv3d (stride 1, pad 1) + per-output-channel bias. ic/oc passed explicitly.
    auto conv = [&](ggml_tensor * in, const std::string & pfx, int ic, int oc) -> ggml_tensor * {
        ggml_tensor * w = W(pfx + ".weight");
        ggml_tensor * b = W(pfx + ".bias");
        if (!w) return in;
        ggml_tensor * y = ggml_conv_3d_direct(ctx, w, in, 1,1,1, 1,1,1, 1,1,1, ic, 1, oc);
        if (b) y = ggml_add(ctx, y, ggml_reshape_4d(ctx, b, 1, 1, 1, oc));
        return y;
    };
    // ChannelLayerNorm32: per-voxel LayerNorm over the channel axis (with affine).
    auto clnorm = [&](ggml_tensor * in, const std::string & pfx) -> ggml_tensor * {
        ggml_tensor * p = ggml_cont(ctx, ggml_permute(ctx, in, 1, 2, 3, 0)); // [C,W,H,D]
        p = ggml_norm(ctx, p, eps);
        p = ggml_mul(ctx, p, W(pfx + ".weight"));
        p = ggml_add(ctx, p, W(pfx + ".bias"));
        return ggml_cont(ctx, ggml_permute(ctx, p, 3, 0, 1, 2));             // [W,H,D,C]
    };
    auto resblock = [&](ggml_tensor * in, const std::string & pfx, int C) -> ggml_tensor * {
        ggml_tensor * h = clnorm(in, pfx + ".norm1");
        h = ggml_silu(ctx, h);
        h = conv(h, pfx + ".conv1", C, C);
        h = clnorm(h, pfx + ".norm2");
        h = ggml_silu(ctx, h);
        h = conv(h, pfx + ".conv2", C, C);
        return ggml_add(ctx, h, in);                                          // skip = Identity
    };
    // pixel_shuffle_3d(scale 2): [A,A,A, Cout*8] -> [2A,2A,2A, Cout]. Each scale
    // bit (LSB->axis0, mid->axis1, MSB->axis2, matching torch's H/W/D pairing)
    // is peeled out of the channel and interleaved into its spatial axis.
    auto pshuf = [&](ggml_tensor * t, int A0, int A1, int A2, int Co) -> ggml_tensor * {
        // peel s2 (channel LSB) into axis0
        t = ggml_reshape_4d(ctx, t, A0, A1 * A2, 2, Co * 4);
        t = ggml_cont(ctx, ggml_permute(ctx, t, 1, 2, 0, 3));   // [2, A0, A1*A2, Co*4]
        t = ggml_reshape_4d(ctx, t, 2 * A0, A1, A2, Co * 4);
        // peel s1 into axis1
        t = ggml_cont(ctx, ggml_permute(ctx, t, 1, 0, 2, 3));   // [A1, 2A0, A2, Co*4]
        t = ggml_reshape_4d(ctx, t, A1, 2 * A0 * A2, 2, Co * 2);
        t = ggml_cont(ctx, ggml_permute(ctx, t, 1, 2, 0, 3));   // [2, A1, 2A0*A2, Co*2]
        t = ggml_reshape_4d(ctx, t, 2 * A1, 2 * A0, A2, Co * 2);
        t = ggml_cont(ctx, ggml_permute(ctx, t, 1, 0, 2, 3));   // [2A0, 2A1, A2, Co*2]
        // peel s0 (channel MSB) into axis2
        t = ggml_cont(ctx, ggml_permute(ctx, t, 1, 2, 0, 3));   // [A2, 2A0, 2A1, Co*2]
        t = ggml_reshape_4d(ctx, t, A2, 2 * A0 * 2 * A1, 2, Co);
        t = ggml_cont(ctx, ggml_permute(ctx, t, 1, 2, 0, 3));   // [2, A2, 2A0*2A1, Co]
        t = ggml_reshape_4d(ctx, t, 2 * A2, 2 * A0, 2 * A1, Co);
        t = ggml_cont(ctx, ggml_permute(ctx, t, 2, 0, 1, 3));   // [2A0, 2A1, 2A2, Co]
        return t;
    };

    // ── forward ───────────────────────────────────────────────────────────────
    ggml_tensor * h = conv(x, "input_layer", Cin, hp.channels[0]);

    for (int i = 0; i < hp.num_res_blocks_middle; ++i) {
        h = resblock(h, "middle_block." + std::to_string(i), hp.channels[0]);
    }

    int blk = 0;
    int cur_res = R;
    for (int lvl = 0; lvl < hp.n_levels; ++lvl) {
        const int C = hp.channels[lvl];
        for (int r = 0; r < hp.num_res_blocks; ++r) {
            h = resblock(h, "blocks." + std::to_string(blk++), C);
        }
        if (lvl < hp.n_levels - 1) {
            const int Co = hp.channels[lvl + 1];
            h = conv(h, "blocks." + std::to_string(blk++) + ".conv", C, Co * 8);
            h = pshuf(h, cur_res, cur_res, cur_res, Co);
            cur_res *= 2;
        }
    }

    h = clnorm(h, "out_layer.0");
    h = ggml_silu(ctx, h);
    h = conv(h, "out_layer.2", hp.channels[hp.n_levels - 1], hp.out_channels); // [Rout,Rout,Rout,Oc]
    ggml_set_output(h);

    if (!missing.empty()) {
        set_error(error, "missing tensor: " + missing);
        ggml_free(ctx);
        return false;
    }

    ggml_build_forward_expand(gf, h);

    ggml_backend_t backend = m->backend;
    ggml_gallocr_t alloc   = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(alloc, gf)) {
        set_error(error, "ggml_gallocr_alloc_graph failed");
        ggml_gallocr_free(alloc); ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_set(x, latent, 0, (size_t) Cin * R * R * R * es);

    const ggml_status st = ggml_backend_graph_compute(backend, gf);
    bool ok = (st == GGML_STATUS_SUCCESS);
    if (ok) {
        const size_t Rout = (size_t) hp.res_out();
        ggml_backend_tensor_get(h, out, 0, (size_t) hp.out_channels * Rout * Rout * Rout * es);
    } else {
        set_error(error, "graph compute failed");
    }

    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    return ok;
}
