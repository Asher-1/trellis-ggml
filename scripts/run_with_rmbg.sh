#!/usr/bin/env bash
# End-to-end with built-in RMBG-2.0 AI background removal.
# The RMBG model is loaded by t2_generate directly — no external preprocessing.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${BUILD:-$ROOT/build}"
MODELS="${MODELS:-$ROOT/models}"
OUT="${OUT:-$ROOT/outputs/rmbg_e2e}"
INPUT="${1:-}"
QUALITY="${QUALITY:-512}"
RMBG_GGUF="${RMBG_GGUF:-$MODELS/rmbg_f16.gguf}"

if [ -z "$INPUT" ] || [ ! -f "$INPUT" ]; then
  echo "usage: $0 <input.jpg|png>" >&2
  exit 2
fi

if [ ! -f "$RMBG_GGUF" ]; then
  echo "error: RMBG model not found at $RMBG_GGUF" >&2
  echo "  set RMBG_GGUF=/path/to/rmbg_f16.gguf or place it in $MODELS/" >&2
  exit 1
fi

mkdir -p "$OUT"
BASE="$(basename "$INPUT" | sed 's/\.[^.]*$//')"

echo "== trellis PBR generate with RMBG-2.0 =="
"$BUILD/examples/t2_generate" \
  --models-dir "$MODELS" \
  --input "$INPUT" \
  --out-mesh "$OUT/${BASE}.t2mesh" \
  --out-glb  "$OUT/${BASE}.glb" \
  --save-grid "$OUT/${BASE}.t2grid" \
  --quality "$QUALITY" \
  --rmbg "$RMBG_GGUF" \
  --seed 42

echo "done: $OUT/${BASE}.glb"
