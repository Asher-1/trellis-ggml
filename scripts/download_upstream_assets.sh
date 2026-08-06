#!/usr/bin/env bash
# Download microsoft/TRELLIS.2/assets into repo root (./assets/).
# Uses GitHub raw URLs (no API token required). Safe to re-run.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${DEST:-$ROOT/assets}"
RAW="https://raw.githubusercontent.com/microsoft/TRELLIS.2/main/assets"

mkdir -p "$DEST"

fetch() {
  local rel="$1"
  local out="$DEST/$rel"
  mkdir -p "$(dirname "$out")"
  if [ -s "$out" ]; then
    echo "skip  $rel"
    return 0
  fi
  echo "fetch $rel"
  curl -fsSL "$RAW/$rel" -o "$out"
}

# Remaining top-level dirs (example_image + app were fetched in bulk elsewhere).
fetch "teaser.webp"
fetch "example_texturing/image.webp"
fetch "example_texturing/the_forgotten_knight.ply"
for h in city courtyard forest interior night studio sunrise sunset; do
  fetch "hdri/${h}.exr"
done

# example_image: list from upstream (71 webp + T.png). Skip if already present.
MANIFEST="$ROOT/scripts/upstream_assets_example_image.txt"
if [ ! -f "$MANIFEST" ]; then
  echo "warning: $MANIFEST missing; only fetching T.png" >&2
  fetch "example_image/T.png"
else
  fetch "example_image/T.png"
  while IFS= read -r name; do
    [ -n "$name" ] || continue
    fetch "example_image/$name"
  done < "$MANIFEST"
fi

count=$(find "$DEST" -type f | wc -l)
size=$(du -sh "$DEST" | cut -f1)
echo "done: $count files under $DEST ($size)"
