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
  python convert_dino_to_gguf.py        --output ggufs/dino_f32.gguf       --ftype 0
  python convert_ss_flow_to_gguf.py     --model models/TRELLIS.2-4B/ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors --output ggufs/ss_flow_f32.gguf --ftype 0
  python convert_ss_dec_to_gguf.py      --model models/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16.safetensors --output ggufs/ss_dec_f32.gguf --ftype 0
  python convert_slat_flow_to_gguf.py   --model models/TRELLIS.2-4B/ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors --pipeline-json models/TRELLIS.2-4B/pipeline.json --output ggufs/slat_flow_f32.gguf --ftype 0
  python convert_shape_dec_to_gguf.py   --output ggufs/shape_dec_f32.gguf  --ftype 0'
scripts/refgen.sh                                              # dump reference activations
```

## Run

```sh
cmake -B build -DTRELLIS2_BUILD_TESTS=ON && cmake --build build -j
ctest --test-dir build -LE model        # fast, no assets (marching cubes, preprocess)
ctest --test-dir build                  # full parity (needs ggufs/ + dumps/)
```

## Parity table (f32 GGUF vs true-fp32 reference)

| stage | test | tap coverage | result |
|---|---|---|---|
| image preprocess (alpha crop, premultiply, Lanczos-512) | `test_preprocess` | full 512×512 RGB | **byte-exact** (0/786432 differ) |
| DINOv3 ViT-L/16 encoder | `test_dino` | 40 taps: embeddings, RoPE, per-layer output + first/last-layer detail (norm/attn/layerscale/mlp), final affine-free LN | **PASS**, rel-L2 ≤ 7e-7 all taps |
| SS-flow DiT forward | `test_ss_flow_forward` | full output | **PASS**, rel-L2 2.4e-4 |
| SS-flow Euler sampler (12-step CFG) | `test_ss_sample` | z_s latent | **PASS**, rel-L2 5.7e-3, sign 99.85% (CPU) |
| SS decoder (dense 3D-conv → 64³ occupancy) | `test_ss_dec` | occupancy logits | **PASS**, rel-L2 2e-5 |
| shape-SLAT flow forward | `test_slat` | full output | **PASS**, rel-L2 2.9e-4 (CPU) / 8e-4 (GPU) |
| shape-SLAT VAE decoder (sparse ConvNeXt U-Net, 4 levels, 16× up) | `test_slat` | per-level features + subdivision logits + final 7-ch output, all 5 levels | **PASS**, rel-L2 ≤ 6e-7 (levels 0–3 exact; final set within 0.0001%) |
| dual-grid mesh extraction | `test_marching_cubes` (invariants) + visual | watertight-manifold, Euler characteristic, winding | **PASS** |
| **1024 cascade** — decoder upsample(×4) → 512³ coords | `test_cascade` | full coord set + quantized 64³ HR scaffold | **PASS**, set match to 0.0001% (1 voxel of 995k) |
| **1024 cascade** — HR (1024-model) flow forward | `test_cascade` | full output | **PASS**, rel-L2 ~3e-4 (CPU); ~1e-2 on GPU flash |
| **1024 cascade** — final 1024³ decode (3.97M voxels) | `test_cascade` | per-level features + subdivision + 7-ch output | **PASS**, rel-L2 ≤ 2e-2, set within 0.0001% |

Notes:
- **Hybrid attention.** `sdpa_auto()` uses exact materialized-softmax attention
  while the `[L_k, L_q, heads]` score matrix stays under 1 GiB (everything in
  the 512 tier — SS-flow 4096 tokens, SLAT ≤~2300 voxels — so the tap parity
  above is the exact path), and switches to `ggml_flash_attn_ext` above that.
  Flash is bit-faithful to full softmax on CPU (SS-flow 2.4e-4, SLAT 2.9e-4 —
  identical to exact) but incurs ~3e-3 rel-L2 on the CUDA F16-MMA kernel. It is
  what makes the HR cascade fit: benchmarked OK at the full 49,152-token cap on
  16 GB, versus a 108 GiB exact self-attention matrix.
- **TF32 matters.** PyTorch's default CUDA matmul/attention uses TF32 (≈10-bit
  mantissa) and reduced-precision flash SDPA, which shows up as ~1e-3 relative
  error versus true fp32. `scripts/ref_common.py` disables it so the golden
  dumps are real fp32; otherwise a correct port looks like it has a 0.08-rel-L2
  bug (this exact trap cost a debugging session — see the flow-forward gate).
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
