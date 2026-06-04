# trellis2.cpp

A C++/[ggml](https://github.com/ggml-org/ggml) implementation of the
**TRELLIS.2** image-to-3D pipeline (stage 1 first: sparse-structure flow).

Modeled structurally after [sam3.cpp](https://github.com/rms80/sam3.cpp):
single-file library (`trellis2.h` / `trellis2.cpp`), bundled ggml as a
submodule (Metal on by default on Apple), DLL-export decoration, and a
CMake build with example executables.

## Status

Early scaffolding. Implemented so far:

- **`.dinodata` loader** — reads the DINOv3 conditioning tensor that the
  sparse-structure flow DiT consumes as cross-attention K/V. This is the full
  DINOv3 ViT-L/16 token sequence (`[1, 1029, 1024]` = 1 CLS + 4 register +
  1024 patch) of the **last transformer layer** with an **affine-free
  LayerNorm** (NOT HF's `last_hidden_state`). `neg_cond = zeros_like(cond)`,
  so it is never stored. Files are produced by `trellis2-shiv/dump_dinodata.py`.

- **SS-flow DiT weights** — `convert_ss_flow_to_gguf.py` converts the stage-1
  `ss_flow_img_dit_1_3B_64_bf16` checkpoint to GGUF; `trellis2_ss_flow_load()`
  reads it back through ggml (hparams from `trellis2.ss_flow.*` KV metadata,
  weights keyed by their original checkpoint names).

- **SS-flow DiT forward pass** — `trellis2_ss_flow_forward()` builds the full
  ggml graph: input projection, sinusoidal timestep + shared adaLN-Zero
  modulation, 30 cross-blocks (self-attention with 3D interleaved RoPE +
  QK-RMSNorm, cross-attention to the DINOv3 tokens, GELU-tanh FFN), and the
  final LayerNorm + output projection. Runs on an **auto-selected backend** —
  the first GPU exposed by ggml (CUDA / Metal / Vulkan / ...), falling back to
  CPU, like sam3.cpp. Validated against a PyTorch f32 reference to **<1e-3
  relative L2** on CPU, Metal (f32), and Metal (f16) (see *Validation* below).

- **Stage-1 sampler** — `trellis2_ss_flow_sample()` runs the full flow-Euler
  loop with classifier-free guidance (interval [0.6,1.0], strength 7.5, rescale
  0.7, rescale_t 5.0, 12 steps; `neg_cond = zeros`) to turn a DINOv3 cond into
  the sparse-structure latent z_s. Validated against the real
  `FlowEulerGuidanceIntervalSampler`: **rel L2 5.7e-3, 99.85% sign agreement**
  (the SS decoder thresholds z_s at 0). Run it:

  ```sh
  ./build/examples/ss_sample ss_flow_dit_f16.gguf /path/img.dinodata out.latent
  # -> z_s [8,16,16,16], occupancy(>0) ~50%
  ```

- **Stage-1 SS decoder** — `trellis2_ss_dec_decode()` runs the
  `SparseStructureDecoder` (a dense 3D-conv ResNet) that turns the z_s latent
  `[8,16³]` into an occupancy logit grid `[1,64³]`, upsampling 16→32→64 with two
  `pixel_shuffle_3d` blocks. The coarse voxel scaffold is `logit > 0`. Runs fully
  on the GPU (ggml `conv_3d_direct`, channel-LayerNorm, in-graph pixel-shuffle).
  Validated against the real PyTorch decoder to **rel L2 5e-7 (f32) / 2e-5 (f16),
  100% sign agreement** on a sampled z_s. Run it:

  ```sh
  ./build/examples/ss_decode ss_dec_f16.gguf out.latent out.occ
  # -> logits [1,64,64,64], occupied(>0) grid (the coarse voxel scaffold)
  ```

- **Occupancy → mesh** — `ss_mesh` decodes a z_s latent and exports the
  `{logit = 0}` isosurface as a watertight OBJ via a self-contained marching
  cubes (`examples/marching_cubes.h`, the tetrahedral / Freudenthal variant — no
  256-row table, provably manifold). Chain it after sampling to *see* the result:

  ```sh
  ./build/examples/ss_sample ss_flow_dit_f16.gguf /path/img.dinodata z_s.latent
  ./build/examples/ss_mesh   ss_dec_f16.gguf z_s.latent shape.obj --normalize
  # -> watertight shape.obj in the centered unit cube; open in any 3D viewer
  ```

## Validate the forward pass

```sh
# 1. lossless f32 weights for an exact comparison
python convert_ss_flow_to_gguf.py --output ss_flow_dit_f32.gguf --ftype 0

# 2. PyTorch f32 reference forward -> tests/ss_flow_ref.bin
python tests/ref_ss_flow.py --dinodata /path/MushroomBoy.dinodata

# 3. build + run the C++ comparison
cmake -B build -DTRELLIS2_BUILD_TESTS=ON && cmake --build build -j
./build/tests/test_ss_flow_forward ss_flow_dit_f32.gguf tests/ss_flow_ref.bin
# -> rel L2 err ~2.8e-4, RESULT: PASS
```

## Validate the SS decoder

```sh
# 1. lossless f32 decoder weights
python convert_ss_dec_to_gguf.py --output ss_dec_f32.gguf --ftype 0

# 2. PyTorch f32 reference decode of a sampled z_s -> tests/ss_dec_ref.bin
./build/examples/ss_sample ss_flow_dit_f16.gguf /path/img.dinodata z_s.latent
python tests/ref_ss_dec.py --latent z_s.latent

# 3. build + run the C++ comparison
./build/tests/test_ss_dec ss_dec_f32.gguf tests/ss_dec_ref.bin
# -> rel L2 err ~5e-7, RESULT: PASS
```

## Convert the stage-1 weights

```sh
# needs safetensors + torch + numpy (e.g. the trellis2-shiv venv)
python convert_ss_flow_to_gguf.py --output ss_flow_dit_f16.gguf --ftype 1   # DiT
python convert_ss_dec_to_gguf.py  --output ss_dec_f16.gguf      --ftype 1   # decoder
```

`--model` / `--config` default to the `microsoft/TRELLIS.2-4B` HF cache
snapshot. `--ftype`: `0` = f32 (lossless upcast from bf16), `1` = f16
(default — big 2-D weight matrices only; norms/gammas/modulation stay f32),
`2` = bf16 (lossless, needs bf16-capable ggml). The f16 file is ~2.6 GB.

Inspect it (validates that ggml can read every tensor):

```sh
./build/examples/ss_flow_info ss_flow_dit_f16.gguf          # metadata only
./build/examples/ss_flow_info ss_flow_dit_f16.gguf --load   # + read all weights
```

## Build

```sh
git clone --recursive <this-repo> trellis2cpp
cd trellis2cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

If you already cloned without `--recursive`:

```sh
git submodule update --init --recursive
```

## Try it

```sh
./build/examples/dino_info /path/to/MushroomBoy.dinodata
```

Prints the shape, token breakdown, and fingerprints (min/max/mean/sum/l2).
`min`/`max`/`count` match the matching `<stem>.dino.txt` JSON sidecar exactly
(they are true element values); `sum`/`l2` agree to float32 precision — the C++
side reduces in `double` and is slightly more accurate than numpy's float32
reduction.

## Layout

| path           | what                                                   |
|----------------|--------------------------------------------------------|
| `trellis2.h`   | public API (DLL-decorated, versioned)                  |
| `trellis2.cpp` | implementation                                         |
| `convert_ss_flow_to_gguf.py` | stage-1 DiT checkpoint → GGUF converter  |
| `convert_ss_dec_to_gguf.py`  | stage-1 decoder checkpoint → GGUF converter |
| `examples/`    | CLI tools (`dino_info`, `ss_flow_info`, `ss_sample`, `ss_decode`, `ss_mesh`) |
| `examples/marching_cubes.h` | single-file isosurface → OBJ extractor      |
| `ggml/`        | submodule, pinned to the same commit as sam3.cpp       |
| `stb/`         | `stb_image.h` / `stb_image_write.h` for image I/O      |

## License

MIT. See [LICENSE](LICENSE).
