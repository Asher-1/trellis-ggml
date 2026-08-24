# TRELLIS.2-4B Model Card

This directory holds all model files required for the TRELLIS.2 image-to-3D
pipeline. Two kinds of files live here:

- **GGUF runtime models** (`*.gguf`, top level) — actually loaded by the ggml
  inference engine, converted from upstream weights;
- **Source weights & configs** (subdirectories) — upstream HuggingFace
  safetensors weights and pipeline configs, used only for conversion/verification,
  never loaded at inference time.

All GGUF files are **local build artifacts** and are excluded from git via
`.gitignore` (`/models/`, `*.gguf`). For benchmarks and parity verification,
see [docs/BENCHMARK.md](../docs/BENCHMARK.md) and
[docs/VERIFICATION.md](../docs/VERIFICATION.md).

---

## 1. Pipeline Overview

```
image (RGB/RGBA)
  → [RMBG-2.0]           rmbg_f16/f32/q8.gguf            Background removal (optional, enabled via --rmbg)
  → [Preprocess]          (no model, pure C++)
  → [DINOv3]             dino_f16/q8.gguf                image → 1029×1024 conditioning tokens
  → [SS-flow DiT]        ss_flow_f16/q8.gguf             sparse structure flow → z_s (8ch)
  → [SS decoder]         ss_dec_f16/q8.gguf              3D-conv → 64³ occupancy → 32³ scaffold
  → [shape-SLAT DiT]     slat_flow[_1024]_f16/q8.gguf    sparse shape flow (512/1024 res)
  → [shape VAE decoder]  shape_dec_f16.gguf              sparse ConvNeXt U-Net → dual grid
  ├→ [meshing]            (no model, marching cubes)
  └→ [shape VAE enc]     shape_enc_f16.gguf              reconstructed dual grid → shape SLat
     → [tex-SLAT DiT]     tex_slat_flow_512/1024_*.gguf   texture flow
     → [texture decoder] tex_dec_f16.gguf                sparse 6-channel PBR volume
  → [material sampling]   (no model, C++)
```

The three DiT flows (SS-flow / shape-SLAT / tex-SLAT) share the same 1.3B DiT
architecture (`trellis2-slat-flow`); they differ only in channel counts and
condition concatenation.

---

## 2. Model Inventory

### 2.1 GGUF runtime models (`models/*.gguf`)

| Model file | Size | Params | GGUF arch | Source weights |
|---------|------|--------|-----------|--------|
| `dino_f16.gguf` | 579 MB | 303.1M | trellis2-dino | `dinov3-vitl16/` |
| `dino_q8.gguf` | 309 MB | 303.1M | trellis2-dino | same |
| `rmbg_f16.gguf` | 421 MB | 220.7M | rmbg (swin_v1_l) | RMBG-2.0-GGML converter |
| `rmbg_f32.gguf` | 842 MB | 220.7M | rmbg (swin_v1_l) | same |
| `rmbg_q8.gguf` | 247 MB | 220.7M | rmbg (swin_v1_l) | same |
| `ss_flow_f16.gguf` | 2494 MB | 1.29B | trellis2-ss-flow | `ss_flow_img_dit_1_3B_64_bf16` |
| `ss_flow_q8.gguf` | 1353 MB | 1.29B | trellis2-ss-flow | same |
| `ss_dec_f16.gguf` | 141 MB | 73.7M | trellis2-ss-dec | `ss_dec_conv3d_16l8_fp16` |
| `ss_dec_q8.gguf` | 141 MB | 73.7M | trellis2-ss-dec | same |
| `slat_flow_f16.gguf` | 2494 MB | 1.29B | trellis2-slat-flow | `slat_flow_img2shape_dit_1_3B_512_bf16` |
| `slat_flow_q8.gguf` | 1353 MB | 1.29B | trellis2-slat-flow | same |
| `slat_flow_1024_f16.gguf` | 2508 MB | 1.29B | trellis2-slat-flow | `slat_flow_img2shape_dit_1_3B_1024_bf16` |
| `slat_flow_1024_q8.gguf` | 1353 MB | 1.29B | trellis2-slat-flow | same |
| `shape_dec_f16.gguf` | 905 MB | 474.2M | trellis2-shape-dec | `shape_dec_next_dc_f16c32_fp16` |
| `shape_enc_f16.gguf` | 676 MB | 354.4M | trellis2-shape-enc | `shape_enc_next_dc_f16c32_fp16` |
| `tex_dec_f16.gguf` | 905 MB | 474.2M | trellis2-tex-dec | `tex_dec_next_dc_f16c32_fp16` |
| `tex_slat_flow_512_f16.gguf` | 2494 MB | 1.29B | trellis2-slat-flow | `slat_flow_imgshape2tex_dit_1_3B_512_bf16` |
| `tex_slat_flow_512_q8.gguf` | 1353 MB | 1.29B | trellis2-slat-flow | same |
| `tex_slat_flow_1024_f16.gguf` | 2494 MB | 1.29B | trellis2-slat-flow | `slat_flow_imgshape2tex_dit_1_3B_1024_bf16` |
| `tex_slat_flow_1024_q8.gguf` | 1353 MB | 1.29B | trellis2-slat-flow | same |

> Params are approximated by summing GGUF tensor shapes. File sizes are the
> decimal byte counts from `ls -l`.

### 2.2 Source weight directories

| Directory | Content | Size | Purpose |
|------|------|------|------|
| `TRELLIS.2-4B/ckpts/` | 8 upstream safetensors (5 DiT + 3 VAE) + json | ~16 GB | main pipeline weight source |
| `TRELLIS.2-4B/pipeline.json` | geometry pipeline config (model map, samplers, normalization) | — | conversion reference |
| `TRELLIS.2-4B/texturing_pipeline.json` | texturing pipeline config | — | conversion reference |
| `TRELLIS-image-large/ckpts/` | `ss_dec_conv3d_16l8_fp16.{json,safetensors}` | 141 MB | SS decoder weight source |
| `dinov3-vitl16/` | DINOv3 ViT-L/16 weights + preprocessor config | 771 MB | image encoder weight source |

---

## 3. Per-Model Details

### 3.1 DINOv3 ViT-L/16 image encoder — `dino_f16.gguf` / `dino_q8.gguf`

- **Source**: `dinov3-vitl16-pretrain-lvd1689m` (Meta DINOv3, HF format under `dinov3-vitl16/`).
- **Architecture** (GGUF `trellis2-dino`, 415 tensors): ViT-L/16, hidden=1024,
  24 layers, 16 heads, intermediate=4096, patch=16, 4 register tokens, RoPE
  (θ=100), no key bias; preprocessing mean=(0.485, 0.456, 0.406),
  std=(0.229, 0.224, 0.225).
- **Role**: encodes the preprocessed 512×512 image into `[1, 1029, 1024]`
  conditioning tokens (1 CLS + 4 register + 1024 patch, taken from the last
  layer with affine LN removed) — the **visual condition** for all three DiT
  flows (SS-flow / shape-SLAT / tex-SLAT).
- **Precision variants**: f16 (579 MB) / q8 (309 MB). Q8 quality loss is tiny;
  recommended default.
- **Benchmark** (RTX 3060, CUDA): ~1s.
- **Note**: conditioning tokens are concatenated with F32 tensors downstream,
  so token-related weights stay f32.

### 3.2 RMBG-2.0 background removal — `rmbg_f16.gguf` / `rmbg_f32.gguf` / `rmbg_q8.gguf`

- **Source**: RMBG-2.0 (BiRefNet family), produced by the
  `third_party/RMBG-2.0-GGML` converter.
- **Architecture** (GGUF `rmbg`, 742 tensors): backbone = Swin-Transformer-Large
  (`swin_v1_l`), 1024×1024 input, outputs a feathered alpha segmentation map.
- **Role**: removes complex backgrounds so the pipeline reconstructs only the
  subject. Not required for solid-color backgrounds; only enabled when
  `--rmbg MODEL.gguf` is passed explicitly — an **optional** preprocess stage.
- **Precision variants**:
  | Variant | Size | CUDA latency | Vulkan latency |
  |------|------|-----------|-------------|
  | f32 | 842 MB | 644.5 ms | 1293.4 ms |
  | **f16 (recommended)** | 421 MB | 655.2 ms | 1278.5 ms |
  | q8 | 247 MB | 648.7 ms | 1278.5 ms |
  - f16 vs f32 max alpha diff 1.1e-4; **f16 is the deployment default**.
  - **Q8 not recommended**: saves only 16.9 MiB vs f16, no speedup, and full Q8
    exceeds the 2e-3 alpha accuracy gate.
- **Note**: under heavy load use `--rmbg-device cpu` to leave VRAM for the main model.

### 3.3 SS-flow DiT (sparse structure flow) — `ss_flow_f16.gguf` / `ss_flow_q8.gguf`

- **Source**: `ss_flow_img_dit_1_3B_64_bf16` (TRELLIS.2-4B main repo).
- **Architecture** (GGUF `trellis2-ss-flow`, 640 tensors): resolution=16
  (16³=4096 tokens), in/out=8, model_channels=1536, cond_channels=1024,
  30 layers, 12 heads, mlp_ratio=5.33, pe_mode=rope, share_mod,
  qk_rms_norm (incl. cross).
- **Role**: flow model of the sparse structure stage. 12-step CFG flow-Euler
  sampling → sparse structure latent `z_s` (8ch), which determines the
  **voxel scaffold** of the object.
- **Benchmark** (RTX 3060, CUDA): ~19.8s — matches PyTorch CUDA 19.76s
  (diff <0.2%, proving ggml matmul performance parity).
- **Precision variants**: f16 (2494 MB) / q8 (1353 MB). Q8 is essentially
  lossless in speed; recommended.

### 3.4 SS decoder (sparse structure decoder) — `ss_dec_f16.gguf` / `ss_dec_q8.gguf`

- **Source**: `ss_dec_conv3d_16l8_fp16` (from the TRELLIS-image-large repo,
  stored under `TRELLIS-image-large/ckpts/`).
- **Architecture** (GGUF `trellis2-ss-dec`, 74 tensors): latent_channels=8,
  out_channels=1, 3 levels (channels 512/128/32), 2+2 res blocks, layer norm.
- **Role**: decodes `z_s` into **64³ occupancy logits** → 32³ voxel scaffold,
  the sparse voxel backbone for shape-SLAT.
- **Benchmark**: small model, runs in milliseconds.
- **Note**: the Q8 variant is the same size as f16 (141 MB) — 3D conv kernels
  (ne[0]=3) do not satisfy the ggml alignment constraint and stay at original
  precision in practice.

### 3.5 Shape-SLAT DiT (shape sparse flow) — `slat_flow_f16/q8.gguf` (512) and `slat_flow_1024_f16/q8.gguf` (1024)

- **Source**: `slat_flow_img2shape_dit_1_3B_512_bf16` / `..._1024_bf16`.
- **Architecture** (GGUF `trellis2-slat-flow`, 640 tensors): in/out=32,
  model_channels=1536, cond_channels=1024, 30 layers, 12 heads, mlp_ratio=5.33,
  RoPE, share_mod, qk_rms_norm; embeds shape SLat normalization mean/std
  (32 channels).
  - **512 variant**: resolution=32, produces 512³ grids (~1M vertices, default).
  - **1024 variant**: resolution=64, 1024 cascade, ~5M vertices high-res grids.
- **Role**: shape flow sampling (12-step CFG) over the 32³ sparse scaffold,
  yielding the shape SLat latent (32 channels).
- **Benchmark** (RTX 3060, CUDA): 512 variant sampling ~7s (PyTorch 17.37s,
  2.5x faster).
- **Note**: the 1024 HR tokens (~49k) only fit in VRAM via flash attention,
  and require DINOv3 encoding at 1024 resolution.

### 3.6 Shape VAE decoder — `shape_dec_f16.gguf`

- **Source**: `shape_dec_next_dc_f16c32_fp16`.
- **Architecture** (GGUF `trellis2-shape-dec`, 292 tensors): latent_channels=32,
  out_channels=7, 5 levels (channels 1024/512/256/128/64, blocks 4/16/8/4/0),
  sparse ConvNeXt U-Net, 16× up.
- **Role**: decodes shape SLat into a 7-channel dual grid (occupancy +
  features) and outputs subdivision guidance. **Performance-critical
  bottleneck** — ggml CUDA takes only 9.8s vs 198s on PyTorch CPU (20x faster).
- **Note**: **f16 only** — sparse subdivision and mesh geometry are not robust
  to Q8 weight rounding; must stay f16 (not a CUDA limitation).

### 3.7 Shape VAE encoder — `shape_enc_f16.gguf`

- **Source**: `shape_enc_next_dc_f16c32_fp16`.
- **Architecture** (GGUF `trellis2-shape-enc`, 284 tensors): in_channels=6,
  latent_channels=32, 5 levels (channels 64/128/256/512/1024, blocks 0/4/8/16/4),
  sparse U-Net (mirror of the decoder).
- **Role**: encodes the reconstructed/validated dual grid back into shape SLat
  + subdivision guide, used as the shape condition for the texturing stage.
- **Note**: **f16 only** (precision-sensitive, same as shape_dec).

### 3.8 Tex-SLAT DiT (texture sparse flow) — `tex_slat_flow_512_f16/q8.gguf` and `tex_slat_flow_1024_f16/q8.gguf`

- **Source**: `slat_flow_imgshape2tex_dit_1_3B_512_bf16` / `..._1024_bf16`.
- **Architecture** (GGUF `trellis2-slat-flow`, 640 tensors): in_channels=64,
  out_channels=32, concat_cond_channels=32 (shape SLat concatenated into the
  channel dim), model_channels=1536, cond_channels=1024, 30 layers, 12 heads,
  mlp_ratio=5.33, RoPE, share_mod, qk_rms_norm; embeds texture SLat
  normalization mean/std and concat normalization mean/std (32 channels each).
  - **512 variant**: resolution=32; **1024 variant**: resolution=64.
- **Role**: samples the texture SLat latent (32 channels) conditioned on the
  shape SLat + DINO tokens.
- **Note**: the 1024 HR variant also relies on flash attention.

### 3.9 Texture decoder — `tex_dec_f16.gguf`

- **Source**: `tex_dec_next_dc_f16c32_fp16`.
- **Architecture** (GGUF `trellis2-tex-dec`, 284 tensors): latent_channels=32,
  out_channels=6, 5 levels (channels 1024/512/256/128/64), sparse U-Net.
- **Role**: replays subdivision from the texture SLat and decodes into a sparse
  **6-channel PBR volume** (base color / metallic / roughness / normal / etc.).
- **Note**: **f16 only** (no Q8, precision-sensitive).

---

## 4. Precision & Quantization Guide

### 4.1 Variant overview

| Model | f16 | q8 | Recommended |
|------|-----|-----|------|
| dino | ✅ | ✅ | **q8** (309 MB, tiny quality loss) |
| rmbg | ✅ | ✅ (not recommended) | **f16** (q8 saves nothing, breaks alpha gate) |
| ss_flow | ✅ | ✅ | **q8** |
| ss_dec | ✅ | ⚠️ (same size) | f16/q8 both fine (q8 is not actually quantized) |
| slat_flow / slat_flow_1024 | ✅ | ✅ | **q8** |
| tex_slat_flow_512 / 1024 | ✅ | ✅ | **q8** |
| shape_dec / shape_enc / tex_dec | ✅ | ❌ | **f16 only** (precision-sensitive) |

### 4.2 End-to-end results (RTX 3060 12GB)

| Config | Model size | 12GB GPU | Inference time | Vertices |
|------|---------|----------|----------|--------|
| All f16 | ~16 GB | ❌ OOM | — | — |
| **Q8 combo (recommended)** | ~4.3 GB | ✅ | **129.0s** (quality=512) | 1,824,032 (−3.2%) |
| All f16 (reference) | ~16 GB | ❌ | 142.7s | 1,884,862 |

Q8 is ~10% faster than F16 (halved memory bandwidth), 73% smaller, with <3.5%
geometric quality loss.

### 4.3 Parity verification

Each GGUF has been verified tap-by-tap against the PyTorch reference (see
[docs/VERIFICATION.md](../docs/VERIFICATION.md)):

| Model | Test | Result |
|------|------|------|
| `dino_*` | `test_dino` (40 taps) | rel-L2 ≤ 7e-7 |
| `ss_flow_*` | `test_ss_flow_forward` / `test_ss_sample` | 2.4e-4 / 5.7e-3 |
| `ss_dec_*` | `test_ss_dec` | rel-L2 2e-5 |
| `slat_flow_*` | `test_slat` | 2.9e-4 (CPU) / 8e-4 (GPU) |
| `shape_dec` | `test_slat` (full 5-level chain) | rel-L2 ≤ 6e-7 |
| `slat_flow_1024` | `test_cascade` | rel-L2 ~3e-4 (CPU) |
| `tex_*` | texture/material regression | pass |

---

## 5. Acquisition & Regeneration

```bash
# One-click download of prebuilt f16 GGUFs (into ggufs/)
scripts/download_ggufs.sh

# Download safetensors source weights (TRELLIS.2-4B / TRELLIS-image-large / dinov3-vitl16)
scripts/download_models.sh

# Convert all GGUFs from safetensors
scripts/convert_all.sh

# Quantize to Q8 (skip precision-sensitive decoders)
python3 quantize_to_q8.py --batch models/ models/ --skip shape_dec --skip tex_dec --skip shape_enc

# RMBG model conversion (from the RMBG-2.0-GGML submodule)
third_party/RMBG-2.0-GGML/scripts/convert_rmbg_to_gguf.py
```

> **Note**: use real files under `models/`, not symlinks pointing outside the
> repo — external paths break inference once moved or deleted. When reusing
> models from another repo, `cp` them in instead of `ln -s`.
