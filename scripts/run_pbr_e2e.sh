#!/usr/bin/env bash
# Full PBR pipeline demo on official assets (image -> textured mesh + GLB + grid sidecar).
set -euo pipefail
set -o pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${BUILD:-$ROOT/build}"
MODELS="${MODELS:-$ROOT/models}"
OUT="${OUT:-$ROOT/outputs/pbr_e2e}"
INPUT="${INPUT:-$ROOT/assets/example_image/T.png}"
QUALITY="${QUALITY:-512}"
# Default inference chain (see .AGENTS.md): f16 weights, seed 0, 12 steps.
QUANT="${QUANT:-f16}"
SEED="${SEED:-0}"

mkdir -p "$OUT"
BIN="$BUILD/examples"

need() {
  [ -x "$1" ] || { echo "missing: $1 (cmake --build build)" >&2; exit 1; }
  [ -s "$2" ] || { echo "missing model: $2" >&2; exit 1; }
}

need "$BIN/t2_generate" "$MODELS/dino_f16.gguf"
need "$BIN/t2_texture" "$MODELS/shape_enc_f16.gguf"

PREFIX="pbr_official_T"
echo "== [1/2] image -> 3D + PBR (quality=$QUALITY) =="
"$BIN/t2_generate" \
  --models-dir "$MODELS" \
  --input "$INPUT" \
  --out-mesh "$OUT/${PREFIX}.t2mesh" \
  --out-glb  "$OUT/${PREFIX}_e2e.glb" \
  --save-grid "$OUT/${PREFIX}.t2grid" \
  --quality "$QUALITY" --quantization "$QUANT" \
  --seed "$SEED" --steps 12 --texture-steps 12 \
  2>&1 | tee "$OUT/${PREFIX}_generate.log"

echo ""
echo "== [2/2] standalone texturing (mesh + grid + image) =="
# Step 1 may leave ~10GB VRAM allocated; retexture encode OOMs on 12GB GPUs unless we use CPU.
TEXTURE_DEVICE="${TEXTURE_DEVICE:-cpu}"
env TRELLIS2_DEVICE="$TEXTURE_DEVICE" \
"$BIN/t2_texture" \
  --models-dir "$MODELS" \
  --mesh "$OUT/${PREFIX}.t2mesh" \
  --grid "$OUT/${PREFIX}.t2grid" \
  --input "$INPUT" \
  --out-mesh "$OUT/${PREFIX}_retexture.t2mesh" \
  --out-glb  "$OUT/${PREFIX}_retexture.glb" \
  --quality "$QUALITY" --quantization "$QUANT" \
  --seed "$SEED" --texture-steps 12 \
  2>&1 | tee "$OUT/${PREFIX}_texture.log"

echo ""
echo "== [3/3] standalone texturing (mesh + image, no grid / QEF path) =="
env TRELLIS2_DEVICE="${TEXTURE_DEVICE:-cpu}" \
"$BIN/t2_texture" \
  --models-dir "$MODELS" \
  --mesh "$OUT/${PREFIX}.t2mesh" \
  --input "$INPUT" \
  --out-mesh "$OUT/${PREFIX}_qef_retexture.t2mesh" \
  --out-glb  "$OUT/${PREFIX}_qef_retexture.glb" \
  --quality "$QUALITY" --quantization "$QUANT" \
  --seed "$SEED" --texture-steps 12 \
  2>&1 | tee "$OUT/${PREFIX}_qef_texture.log"

echo ""
echo "== [4/4] UV atlas GLB (default C++ export; official mesh ~1.8M vert — slow; override T2GLB_DECIMATION) =="
env T2GLB_VERBOSE=1 T2GLB_DECIMATION="${T2GLB_DECIMATION:-50000}" T2GLB_TEXTURE_SIZE="${T2GLB_TEXTURE_SIZE:-512}" \
"$BIN/mesh2glb" "$OUT/${PREFIX}_qef_retexture.t2mesh" "$OUT/${PREFIX}_atlas.glb" \
  2>&1 | tee "$OUT/${PREFIX}_atlas.log"

echo ""
echo "done. artifacts in $OUT"
ls -lh "$OUT"
