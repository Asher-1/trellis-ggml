#!/usr/bin/env bash
# Download all official prebuilt TRELLIS.2 GGUF weights into models/.
# Official LocalAI-io repos publish f16 only (10 files, ~14 GB total).
# f32 validation variants require safetensors + scripts/convert_all.sh (offline).
#
# Usage:
#   scripts/download_models_ggufs.sh
#   MODELS=/path/to/models scripts/download_models_ggufs.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODELS="${MODELS:-$ROOT/models}"
ORG="${GGUF_ORG:-LocalAI-io}"
TOKEN="${HF_TOKEN:-}"
AUTH=()
[ -n "$TOKEN" ] && AUTH=(-H "Authorization: Bearer $TOKEN")

mkdir -p "$MODELS"

fetch() { # fetch <repo> <file>
    local url="https://huggingface.co/$ORG/$1/resolve/main/$2"
    local dest="$MODELS/$2"
    if [ -s "$dest" ]; then echo "have  $dest"; return 0; fi
    echo "fetch $ORG/$1/$2"
    curl -sSL --fail -C - "${AUTH[@]}" -o "$dest.part" "$url"
    mv "$dest.part" "$dest"
}

T2=TRELLIS.2-4B-GGUF
T1=TRELLIS-image-large-GGUF
DINO=dinov3-vitl16-pretrain-lvd1689m-GGUF

fetch "$DINO" dino_f16.gguf
fetch "$T1"   ss_dec_f16.gguf
fetch "$T2"   ss_flow_f16.gguf
fetch "$T2"   slat_flow_f16.gguf
fetch "$T2"   slat_flow_1024_f16.gguf
fetch "$T2"   shape_dec_f16.gguf
fetch "$T2"   shape_enc_f16.gguf
fetch "$T2"   tex_dec_f16.gguf
fetch "$T2"   tex_slat_flow_512_f16.gguf
fetch "$T2"   tex_slat_flow_1024_f16.gguf

echo "all official GGUFs present in $MODELS:"
du -sh "$MODELS"
ls -lh "$MODELS"/*.gguf
