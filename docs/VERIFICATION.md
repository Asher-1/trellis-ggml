# Verification

Every ported stage is validated numerically against the PyTorch reference, per
component, not just end-to-end — the depth-anything.cpp approach. Reference
activations are dumped from `microsoft/TRELLIS.2` (via `scripts/refgen.sh`
inside the CUDA container) into GGUF files under `dumps/`, and the C++ side is
compared tap-by-tap with `tests/parity.hpp` (gate: `|got-ref| <= atol +
rtol*|ref|`, reported as max-abs / rel-L2 / per-row).

## One-time setup

```sh
docker build -f docker/Dockerfile.ref -t trellis2-ref docker   # PyTorch reference env
scripts/download_models.sh                                     # HF checkpoints -> models/
docker run --rm -v "$PWD":/work -w /work trellis2-ref bash -c '
  python convert_dino_to_gguf.py        --output models/dino_f32.gguf       --ftype 0
  python convert_ss_flow_to_gguf.py     --model models/TRELLIS.2-4B/ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors --output models/ss_flow_f32.gguf --ftype 0
  python convert_ss_dec_to_gguf.py      --model models/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16.safetensors --output models/ss_dec_f32.gguf --ftype 0
  python convert_slat_flow_to_gguf.py   --model models/TRELLIS.2-4B/ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors --pipeline-json models/TRELLIS.2-4B/pipeline.json --output models/slat_flow_f32.gguf --ftype 0
  python convert_shape_dec_to_gguf.py   --output models/shape_dec_f32.gguf  --ftype 0'
scripts/refgen.sh                                              # dump reference activations
```

## Run

```sh
cmake -B build -DTRELLIS2_BUILD_TESTS=ON && cmake --build build -j
ctest --test-dir build -LE model        # fast, no assets (marching cubes, preprocess)
ctest --test-dir build                  # full parity (needs models/ + dumps/)
```

## Parity table (f32 GGUF vs true-fp32 reference)

| stage | test | tap coverage | result |
|---|---|---|---|
| image preprocess (alpha crop, premultiply, Lanczos-512) | `test_preprocess` | full 512×512 RGB | **byte-exact** (0/786432 differ) |
| DINOv3 ViT-L/16 encoder | `test_dino` | 40 taps: embeddings, RoPE, per-layer output + first/last-layer detail (norm/attn/layerscale/mlp), final affine-free LN | **PASS**, rel-L2 ≤ 7e-7 all taps |
| SS-flow DiT forward | `test_ss_flow_forward` | full output | **PASS**, rel-L2 2.4e-4 |
| SS-flow Euler sampler (12-step CFG) | `test_ss_sample` | z_s latent | **PASS**, rel-L2 5.7e-3, sign 99.85% (CPU) |
| SS decoder (dense 3D-conv → 64³ occupancy) | `test_ss_dec` | occupancy logits | **PASS**, rel-L2 2e-5 |
| shape-SLAT flow forward | `test_slat` | full output | **PASS**, rel-L2 2.9e-4 (CPU) / 2.5e-6 (GPU, TF32 disabled) |
| shape-SLAT VAE decoder (sparse ConvNeXt U-Net, 4 levels, 16× up) | `test_slat` | per-level features + subdivision logits + final 7-ch output, all 5 levels | **PASS**, rel-L2 ≤ 6e-7 (levels 0–3 exact; final set within 0.0001%) |
| integrated subdivision guide | `test_slat` | all decoder levels; final guide coordinates equal decoded shape coordinates | **PASS** |
| standalone shape encoder → texture flow → texture decoder | `test_texture` | shape latent, flow forward/sampler, guided 6-channel PBR decode | parity-gated; sampler backend drift uses the documented loose gate |
| sparse PBR surface sampling | `test_pbr_sampling` | dense trilinear interpolation + sparse-boundary normalization | **PASS** |
| GLB PBR/alpha export | `test_mesh_export` | direct vertex RGBA, retained metallic/roughness, glTF alpha mode | **PASS** |
| dual-grid mesh extraction | `test_marching_cubes` (invariants) + visual | watertight-manifold, Euler characteristic, winding | **PASS** |
| **1024 cascade** — decoder upsample(×4) → 512³ coords | `test_cascade` | full coord set + quantized 64³ HR scaffold | **PASS**, set match to 0.0001% (1 voxel of 995k) |
| **1024 cascade** — HR (1024-model) flow forward | `test_cascade` | full output | **PASS**, rel-L2 ~3e-4 (CPU); ~1e-2 on GPU flash (K/V→F16; see notes) |
| **1024 cascade** — final 1024³ decode (3.97M voxels) | `test_cascade` | per-level features + subdivision + 7-ch output | **PASS**, rel-L2 ≤ 2e-2, set within 0.0001% |

Notes:
- **Flash attention is opt-out for a reason.** `sdpa_auto()` runs the exact
  F32 materialized softmax by default: one score matrix when
  `Lq*Lk*H*sizeof(float) <= 12 GiB`, otherwise a Q-dim-chunked exact path
  (mathematically identical — each softmax row is independent) that bounds
  memory even for the 49,152-token HR cascade attention (whose full score
  matrix would be ~108 GiB). `TRELLIS2_SDPA_EXACT` forces the unchunked path,
  `TRELLIS2_SDPA_FLASH` opts back into `ggml_flash_attn_ext`.
  Exact is the fp32 reference (CPU SS-flow 2.4e-4, SLAT 2.9e-4). Flash is
  not bit-faithful anywhere: the CUDA F16-MMA kernel rounds F32 K/V to fp16
  (`ggml_cuda_flash_attn_ext` does this unconditionally) costing ~3e-3
  rel-L2 per forward at 10k+ tokens, and even CPU flash's tiling order drifts
  ~1e-4 — both gaps are amplified by the 12-step CFG sampler into different
  voxel sets (on CUDA all the way to collapsing the HR shape decode's
  subdivision predictions, "no children at level 0"). That is why every stage
  runs exact by default; see the backend fixes below.
- **Backend precision fixes (2026-08).** Three traps made CUDA/Vulkan drift
  outside the sampler's noise band:
  1. **Vulkan fp16 accumulation** — `ggml_vk_get_mul_mat_mat_pipeline` picked
     an fp16-accumulate pipeline for F16×F32 matmuls. Fixed by forcing
     `GGML_PREC_F32` on every `lin()` in trellis2.cpp: SS-flow single-step
     rel-L2 7.5e-3 → 5.5e-4; coarse q8 mesh Δverts +58% → +0.8%.
  2. **CUDA flash-attn K/V→F16** — see above; the exact budget restored the
     fp32 reference path for all non-HR attention (CUDA f32 single-step
     5.6e-3 → 8e-4).
  3. **cuBLAS TF32** — cuBLAS runs FP32 GEMMs in TF32 tensor-op math unless
     told otherwise; trellis2 sets `NVIDIA_TF32_OVERRIDE=0` at backend init.
     `test_slat` single step 8.2e-4 → 2.5e-6.
  4. **CUDA f16 GEMM kernels** — ggml-cuda dispatches small-batch F16×F32
     matmuls to `mmvf`/`mmf` kernels that accumulate in half and ignores
     `GGML_PREC_F32` there; `patches/ggml-trellis.patch` makes PREC_F32
     bypass them and fall through to the FP32 cuBLAS path.
  5. **Q8 + chaos** — Q8_0 error on anything feeding the samplers (dino cond
     ~7e-3 rel-L2, flow weights) is amplified into completely different voxel
     sets (orientation flips); `t2_generate` therefore quantizes only the
     post-sampling `ss_dec` and keeps dino/flow/decoders at F16 even in q8
     mode.
- **TF32 in the reference.** PyTorch's default CUDA matmul/attention uses TF32
  (≈10-bit mantissa) and reduced-precision flash SDPA, which shows up as ~1e-3
  relative error versus true fp32. `scripts/ref_common.py` disables it so the
  golden dumps are real fp32; otherwise a correct port looks like it has a
  0.08-rel-L2 bug (this exact trap cost a debugging session — see the
  flow-forward gate).
- **Sampler drift.** The 12-step Euler + CFG-rescale loop chaotically amplifies
  per-step fp differences between backends; it validates tightly on CPU and
  drifts to ~0.1 rel-L2 on GPU. The decoder gate therefore decodes the
  *reference* SLAT so decoder parity is independent of sampler trajectory.
- **Subdivision boundary.** A handful of level-3 subdivision logits sit within
  fp-noise of zero; the `>0` threshold can flip them, so the final active-voxel
  set differs by ~4 voxels out of ~4 million (0.0001%) run-to-run and
  hardware-to-hardware. This is inherent to a hard threshold, not a port bug.

## GPU

`-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120` (Blackwell / RTX 50-series). The
flow DiTs and DINOv3 run on CUDA; the 3D-conv decoders (SS decoder CONV_3D and
the sparse-conv gather-GEMM) run on CPU because the bundled ggml has no CUDA
CONV_3D kernel — they are a small fraction of total inference time. GPU f32
matmul is fp16-class, so tap parity on CUDA is ~1e-3 (deterministic, not device
noise); CPU is the tight-tolerance reference backend.

## Q8 Quantization

`quantize_to_q8.py` converts F16/F32 models to Q8_0 (~50% size reduction). The
Q8_0 implementation exactly matches ggml's `quantize_row_q8_0_ref()`.

### End-to-end mesh parity (quality=512, steps=12, seed=42, 2026-08-25)

| Backend × precision | Vertices | Triangles | Δverts vs CPU same-precision |
|--------------------|----------|-----------|------------------------------|
| CPU q8  | 1,806,350 | 3,858,700 | reference |
| CUDA q8 | 1,758,271 | 3,678,790 | −2.7% |
| Vulkan q8 | 1,780,085 | 3,759,418 | −1.5% |
| CPU f16  | 1,880,963 | 3,985,350 | reference |
| CUDA f16 | 1,796,584 | 3,759,754 | −4.5% |
| Vulkan f16 | 1,906,072 | 4,036,634 | +1.3% |

### End-to-end mesh parity (coarse, steps=12, seed=42)

| Backend × precision | Vertices | Δverts vs CPU q8 |
|--------------------|----------|------------------|
| CPU q8  | 82,437 | reference |
| CUDA q8 | 82,934 | +0.6% (was +11% before the fixes) |
| Vulkan q8 | 83,093 | +0.8% (was +58% before the fixes) |
| CPU f16  | 83,436 | +1.2% |
| CUDA f16 | 83,400 | +1.2% |
| Vulkan f16 | 83,574 | +1.4% |

Note: the 512 numbers include the shape decoder's sparse-subdivision hard
threshold (`logit > 0`), so per-voxel feature diffs (~1e-3) can flip thousands
of boundary voxels; the aggregate geometry stays within the percentages above.
PBR texture mean/std across backends agree within <12% per channel.

### CUDA backend constraints

- `ggml_cuda_cpy` supports Q8_0 → Q8_0 copies (part of
  `patches/ggml-trellis.patch`). End-to-end Q8 inference runs fully on CUDA.
- Precision-sensitive decoders (`shape_dec`, `tex_dec`, `shape_enc`) are kept at
  F16 by policy (`quantize_to_q8.py --skip`), not because of a CUDA limitation
  — **experiment-confirmed 2026-08-30** (CUDA+Vulkan e2e, T.png 512/seed0/12):
  q8 weight noise (rel-L2 ~5-7e-3 vs f16, ~25× f16's own) is fatal to all
  three. shape_dec q8 flips every level-0 subdivision logit negative
  ("no children at level 0"); tex_dec q8 collapses into a saturated PBR
  material; shape_enc q8's condition error is chaotically amplified by the
  12-step texture sampler into the same collapse. Note the FlexiDualGrid
  [Co,27,Ci] conv layout is technically Q8-blockable (Ci % 32 = 0, view
  offsets stay block-aligned) — the blocker is purely numerical.
- Token embeddings (`cls_token`, `register_tokens`) must stay F32 because they
  are concatenated with F32 computed tensors in the DINO encoder graph.
- The only model q8 mode actually loads quantized is `ss_dec` — and its dense
  conv3d weight layout [3,3,3,N] fails the dims[0] % 32 check, so
  `ss_dec_q8.gguf` is a byte clone of the f16 file (only `general.file_type`
  differs). q8 mode currently saves no geometric-model bytes (only rmbg).

See `docs/BENCHMARK.md` §5 for full Q8 benchmark data and comparison charts.
