#!/usr/bin/env bash
# Run official C++ example tools and write artifacts under outputs/demos/.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${BUILD:-$ROOT/build}"
MODELS="${MODELS:-$ROOT/models}"
OUT="${OUT:-$ROOT/outputs/demos}"
BIN="$BUILD/examples"

ROBOTAXI="${ROBOTAXI:-/home/ludahai/develop/data/robotaxi_data/YR-EC15S-29_20260624_025519/frames/camera_4/frame_000100.jpg}"
QUALITY="${QUALITY:-512}"   # 512 fits 12GB VRAM; use 1024 on >=16GB GPUs

mkdir -p "$OUT/stages" "$OUT/full"

need() {
  [ -x "$1" ] || { echo "missing binary: $1 (run cmake --build build)" >&2; exit 1; }
  [ -s "$2" ] || { echo "missing model: $2 (run scripts/download_models_ggufs.sh)" >&2; exit 1; }
}

need "$BIN/dino_encode" "$MODELS/dino_f16.gguf"
need "$BIN/ss_sample"   "$MODELS/ss_flow_f16.gguf"
need "$BIN/ss_decode"   "$MODELS/ss_dec_f16.gguf"
need "$BIN/ss_mesh"     "$MODELS/ss_dec_f16.gguf"
need "$BIN/t2_generate" "$MODELS/shape_dec_f16.gguf"

INPUT="$OUT/input.jpg"
cp -f "$ROBOTAXI" "$INPUT"
echo "input: $INPUT"

echo "== dino_info (on encoded cond) =="
"$BIN/dino_encode" "$MODELS/dino_f16.gguf" "$INPUT" "$OUT/stages/cond.dinodata" --pre "$OUT/stages/preprocessed.png"
"$BIN/dino_info" "$OUT/stages/cond.dinodata" | tee "$OUT/stages/dino_info.txt"

echo "== ss_flow_info =="
"$BIN/ss_flow_info" "$MODELS/ss_flow_f16.gguf" --load | tee "$OUT/stages/ss_flow_info.txt"

echo "== stage chain: ss_sample -> ss_decode -> ss_mesh =="
"$BIN/ss_sample" "$MODELS/ss_flow_f16.gguf" "$OUT/stages/cond.dinodata" "$OUT/stages/z_s.latent"
# SS decoder uses dense CONV_3D — no CUDA kernel in bundled ggml; force CPU (see docs/VERIFICATION.md).
TRELLIS2_DEVICE=cpu "$BIN/ss_decode" "$MODELS/ss_dec_f16.gguf" "$OUT/stages/z_s.latent" "$OUT/stages/occ.bin"
TRELLIS2_DEVICE=cpu "$BIN/ss_mesh" "$MODELS/ss_dec_f16.gguf" "$OUT/stages/z_s.latent" "$OUT/stages/coarse_shape.obj" --normalize

echo "== full pipeline: t2_generate (quality=$QUALITY) =="
PREFIX="trellis-ggml_YR-EC15S-29_20260624_025519_camera_4"
"$BIN/t2_generate" \
  --models-dir "$MODELS" \
  --input "$INPUT" \
  --out-mesh "$OUT/full/${PREFIX}.t2mesh" \
  --out-glb  "$OUT/full/${PREFIX}.glb" \
  --quality "$QUALITY" \
  --seed 42 --steps 12 \
  2>&1 | tee "$OUT/full/${PREFIX}.log"

echo "== mesh2glb (re-export GLB from t2mesh) =="
"$BIN/mesh2glb" "$OUT/full/${PREFIX}.t2mesh" "$OUT/full/${PREFIX}_mesh2glb.glb"

echo "done. artifacts in $OUT"
ls -lh "$OUT/stages" "$OUT/full"
