#!/usr/bin/env bash
# Run all C++ examples against upstream microsoft/TRELLIS.2 assets.
# Primary input: assets/example_image/T.png (official demo image with alpha).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${BUILD:-$ROOT/build}"
MODELS="${MODELS:-$ROOT/models}"
ASSETS="${ASSETS:-$ROOT/assets}"
OUT="${OUT:-$ROOT/outputs/official_assets}"
BIN="$BUILD/examples"
INPUT="${INPUT:-$ASSETS/example_image/T.png}"
QUALITY="${QUALITY:-512}"   # 512 fits 12GB VRAM

mkdir -p "$OUT/stages" "$OUT/full"

need() {
  [ -x "$1" ] || { echo "missing binary: $1 (run cmake --build build)" >&2; exit 1; }
  [ -s "$2" ] || { echo "missing model: $2 (run scripts/download_models_ggufs.sh)" >&2; exit 1; }
}

[ -s "$INPUT" ] || { echo "missing input: $INPUT (run scripts/download_upstream_assets.sh)" >&2; exit 1; }

need "$BIN/dino_encode" "$MODELS/dino_f16.gguf"
need "$BIN/dino_info"   "$MODELS/dino_f16.gguf"
need "$BIN/ss_flow_info" "$MODELS/ss_flow_f16.gguf"
need "$BIN/ss_sample"   "$MODELS/ss_flow_f16.gguf"
need "$BIN/ss_decode"   "$MODELS/ss_dec_f16.gguf"
need "$BIN/ss_mesh"     "$MODELS/ss_dec_f16.gguf"
need "$BIN/t2_generate" "$MODELS/shape_dec_f16.gguf"
need "$BIN/mesh2glb"    "$MODELS/shape_dec_f16.gguf"

echo "input: $INPUT"
echo "output: $OUT"

echo "== [1/8] dino_encode =="
"$BIN/dino_encode" "$MODELS/dino_f16.gguf" "$INPUT" "$OUT/stages/cond.dinodata" \
  --pre "$OUT/stages/preprocessed.png"

echo "== [2/8] dino_info =="
"$BIN/dino_info" "$OUT/stages/cond.dinodata" | tee "$OUT/stages/dino_info.txt"

echo "== [3/8] ss_flow_info =="
"$BIN/ss_flow_info" "$MODELS/ss_flow_f16.gguf" --load | tee "$OUT/stages/ss_flow_info.txt"

echo "== [4/8] ss_sample =="
"$BIN/ss_sample" "$MODELS/ss_flow_f16.gguf" "$OUT/stages/cond.dinodata" "$OUT/stages/z_s.latent"

echo "== [5/8] ss_decode (CPU — CONV_3D has no CUDA kernel) =="
TRELLIS2_DEVICE=cpu "$BIN/ss_decode" "$MODELS/ss_dec_f16.gguf" \
  "$OUT/stages/z_s.latent" "$OUT/stages/occ.bin"

echo "== [6/8] ss_mesh (CPU) =="
TRELLIS2_DEVICE=cpu "$BIN/ss_mesh" "$MODELS/ss_dec_f16.gguf" \
  "$OUT/stages/z_s.latent" "$OUT/stages/coarse_shape.obj" --normalize

PREFIX="trellis_official_T"
echo "== [7/8] t2_generate (quality=$QUALITY) =="
"$BIN/t2_generate" \
  --models-dir "$MODELS" \
  --input "$INPUT" \
  --out-mesh "$OUT/full/${PREFIX}.t2mesh" \
  --out-glb  "$OUT/full/${PREFIX}.glb" \
  --quality "$QUALITY" \
  --seed 42 --steps 12 \
  2>&1 | tee "$OUT/full/${PREFIX}.log"

echo "== [8/8] mesh2glb =="
"$BIN/mesh2glb" "$OUT/full/${PREFIX}.t2mesh" "$OUT/full/${PREFIX}_mesh2glb.glb"

echo ""
echo "all 8 examples passed. artifacts:"
ls -lh "$OUT/stages" "$OUT/full"
