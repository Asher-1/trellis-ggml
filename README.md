# trellis2.cpp

A C++/[ggml](https://github.com/ggml-org/ggml) implementation of the
**TRELLIS.2** image-to-3D pipeline: an image goes in, a 3D mesh with per-vertex
PBR textures comes out, with all inference in C++/ggml (no PyTorch at runtime).
The demo can also bake the result into a portable, image-textured **GLB** —
CUDA-free UV atlas + PBR bake, no reference container required.

Modeled structurally after [sam3.cpp](https://github.com/rms80/sam3.cpp):
single-file library (`trellis2.h` / `trellis2.cpp`), bundled ggml as a
submodule (Metal on by default on Apple), DLL-export decoration, and a
CMake build with example executables. A flat C ABI (`trellis2_capi.h`) drives
a Go demo server with a browser mesh viewer.

## Quick start (demo)

```sh
git submodule update --init --depth 1                 # ggml
scripts/download_models.sh                            # HF checkpoints -> models/ (~7 GB)
docker build -f docker/Dockerfile.ref  -t trellis2-ref  docker   # convert weights / gen refs
docker build -f docker/Dockerfile.demo -t trellis2-demo docker   # CUDA runtime + Go
# convert every checkpoint to GGUF (f16 for the demo, f32 for validation)
docker run --rm -v "$PWD":/work -w /work trellis2-ref bash scripts/convert_all.sh
# build the CUDA shared lib + Go server, then run
docker run --rm -v "$PWD":/work -w /work trellis2-demo bash -c '
  cmake -B build-cuda-shared -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES=120 -DBUILD_SHARED_LIBS=ON && cmake --build build-cuda-shared -j
  cd server && go build -o trellis2-server-linux .'
docker run --rm --device nvidia.com/gpu=all -v "$PWD":/work -w /work/server -p 8742:8742 \
  trellis2-demo ./trellis2-server-linux -lib /work/build-cuda-shared/libtrellis2.so -ggufs /work/ggufs
# open http://localhost:8742 , drop an image (transparent background works best)
```

The browser UI has a **quality** selector: coarse preview (64³ marching cubes),
512³ fine, or **1024³ cascade** (the TRELLIS.2 default — sharpest). Coarse falls
back automatically if the shape-SLAT models are absent (`-coarse`); the 1024
cascade needs the extra 1024 model (`-no-1024` disables it).

On a 16 GB RTX 50-series: the 512 fine path runs image→mesh in ~110 s (~1M-vertex
512³ mesh); the 1024 cascade adds a second 1.3B-model pass and the 1024³ decoder
for a ~5M-vertex mesh (~5 min, ~10 GB VRAM, and a ~14 GB host-RAM spike for the
1024³ sparse-conv decode).

## Pipeline

```
image (RGBA)
  → preprocess          alpha bbox crop, premultiply, PIL-exact Lanczos-512   [C++, byte-exact]
  → DINOv3 ViT-L/16     [1, 1029, 1024] conditioning tokens                   [C++/ggml]
  → SS-flow DiT         1.3B dense DiT, 12-step CFG flow-Euler → z_s          [C++/ggml]
  → SS decoder          dense 3D-conv → 64³ occupancy → 32³ voxel scaffold    [C++/ggml]
  → shape-SLAT DiT      1.3B sparse DiT over active voxels, 12-step CFG       [C++/ggml]
  → shape VAE decoder   sparse ConvNeXt U-Net, 16× up → 512³ dual-grid fields [C++/ggml]
  → flexible dual grid  → triangle mesh                                       [C++]
```

The **1024 cascade** (default in TRELLIS.2) adds a second pass on top: the 512
result's decoder `.upsample(×4)` predicts a denser coordinate scaffold, which is
quantized to 64³ and fed to a second 1.3B shape-SLAT flow (the 1024 model,
conditioned on a 1024-res DINOv3 encode) and the same decoder at 1024³ — a
~5M-vertex mesh. The ~49k-token HR attention only fits in VRAM via flash
attention (`sdpa_auto`); see [docs/VERIFICATION.md](docs/VERIFICATION.md).

Every stage is validated tap-by-tap against the PyTorch reference — see
[docs/VERIFICATION.md](docs/VERIFICATION.md). Highlights: preprocessing is
byte-exact, the DINOv3 encoder matches to rel-L2 ≤ 7e-7 across 40 taps, and the
sparse U-Net decoder is numerically exact through all four conv levels.

## Components

- **Image preprocessing + DINOv3 encoder** — `trellis2_preprocess_rgba()`
  reproduces `pipeline.preprocess_image` (the has-alpha path) with a
  PIL-compatible fixed-point Lanczos-3 resampler (byte-exact vs Pillow).
  `trellis2_dino_encode()` runs the full DINOv3 ViT-L/16 (axial-2D RoPE,
  LayerScale, exact-GELU MLP) and applies the affine-free final LayerNorm the
  flow models expect — the `[1, 1029, 1024]` conditioning that used to come
  from an external `dump_dinodata.py`. `dino_encode` chains them:

  ```sh
  ./build/examples/dino_encode ggufs/dino_f16.gguf image.png cond.dinodata
  ```

- **`.dinodata` loader** — `trellis2_load_dinodata()` still reads/writes the
  precomputed conditioning tensor (1 CLS + 4 register + 1024 patch, last layer,
  affine-free LN; `neg_cond = zeros_like(cond)`), for testing and CLI chaining.

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

- **Occupancy → coarse mesh** — `ss_mesh` decodes a z_s latent and exports the
  `{logit = 0}` isosurface as a watertight OBJ via a self-contained marching
  cubes (`examples/marching_cubes.h`, the tetrahedral / Freudenthal variant — no
  256-row table, provably manifold). This is the fast preview path:

  ```sh
  ./build/examples/ss_sample ss_flow_dit_f16.gguf /path/img.dinodata z_s.latent
  ./build/examples/ss_mesh   ss_dec_f16.gguf z_s.latent shape.obj --normalize
  # -> watertight shape.obj in the centered unit cube; open in any 3D viewer
  ```

- **Shape-SLAT flow + decoder (fine geometry)** — `trellis2_slat_flow_sample()`
  runs the sparse 1.3B DiT over the active voxels of the 32³ scaffold (same
  block structure as the SS-flow DiT, 3D RoPE over each voxel's coords),
  denormalized with `shape_slat_normalization` baked into the GGUF.
  `trellis2_shape_dec_decode()` runs `FlexiDualGridVaeDecoder` — a sparse
  ConvNeXt U-Net whose 3×3×3 submanifold convolutions are expressed as 27
  gather+GEMM steps, with each level's learned subdivision growing the active
  set (32³ → 512³, 16×). `examples/flexible_dual_grid.h` turns the 7-channel
  per-voxel output (dual-vertex offset, per-axis intersection flags, quad split
  weight) into the triangle mesh. This is the real TRELLIS.2 geometry, driven
  end-to-end by the demo server.

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
| `mesh_export.{h,cpp}` | CUDA-free textured-GLB bake: decimate → UV atlas → PBR bake → glTF |
| `examples/`    | CLI tools (`dino_info`, `ss_flow_info`, `ss_sample`, `ss_decode`, `ss_mesh`, `mesh2glb`) |
| `examples/marching_cubes.h` | single-file isosurface → OBJ extractor      |
| `third_party/` | vendored `meshoptimizer` (QEM decimate) + `xatlas` (opt-in charts) |
| `ggml/`        | submodule, pinned to the same commit as sam3.cpp       |
| `stb/`         | `stb_image.h` / `stb_image_write.h` for image I/O      |

## License

MIT. See [LICENSE](LICENSE). Vendored third-party code is also MIT:
[meshoptimizer](https://github.com/zeux/meshoptimizer) (Arseny Kapoulkine) and
[xatlas](https://github.com/jpcy/xatlas) (Jonathan Young) under `third_party/`,
and `stb` (public domain / MIT).
