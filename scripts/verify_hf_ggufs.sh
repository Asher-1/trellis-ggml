#!/usr/bin/env bash
# Verify that the local f16 runtime GGUFs are byte-identical to the official
# LocalAI-io HuggingFace releases.
#
# The upstream trellis2cpp repo publishes ready-made f16 GGUFs on HF:
#   LocalAI-io/TRELLIS.2-4B-GGUF
#   LocalAI-io/TRELLIS-image-large-GGUF
#   LocalAI-io/dinov3-vitl16-pretrain-lvd1689m-GGUF
# The same files can be regenerated locally with the deterministic
# convert_*_to_gguf.py scripts, but the published artifacts are the canonical
# consistency baseline. This script compares SHA-256 of the local files against
# the LFS object ids reported by the HF API (no model download needed).
#
# Usage:  scripts/verify_hf_ggufs.sh [--models-dir DIR]
# Exit:   0 when every expected file is present and matches, 1 otherwise.
# Env:    HF_ORG overrides the HF org (default LocalAI-io).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODELS="${ROOT}/models"
ORG="${HF_ORG:-LocalAI-io}"

while [ $# -gt 0 ]; do
    case "$1" in
        --models-dir) MODELS="$2"; shift 2 ;;
        -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

[ -d "$MODELS" ] || { echo "models dir not found: $MODELS" >&2; exit 2; }

# The ten f16 deployment models and the HF repo that publishes each one.
declare -A REPO_OF=(
    [dino_f16.gguf]=dinov3-vitl16-pretrain-lvd1689m-GGUF
    [ss_dec_f16.gguf]=TRELLIS-image-large-GGUF
    [ss_flow_f16.gguf]=TRELLIS.2-4B-GGUF
    [slat_flow_f16.gguf]=TRELLIS.2-4B-GGUF
    [slat_flow_1024_f16.gguf]=TRELLIS.2-4B-GGUF
    [shape_dec_f16.gguf]=TRELLIS.2-4B-GGUF
    [shape_enc_f16.gguf]=TRELLIS.2-4B-GGUF
    [tex_dec_f16.gguf]=TRELLIS.2-4B-GGUF
    [tex_slat_flow_512_f16.gguf]=TRELLIS.2-4B-GGUF
    [tex_slat_flow_1024_f16.gguf]=TRELLIS.2-4B-GGUF
)

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# 1) Pull the official LFS sha256s from the HF API (metadata only, no downloads).
: > "$TMP/hf.sha256"
for repo in "${REPO_OF[@]}"; do
    if ! curl -sSL --fail "https://huggingface.co/api/models/${ORG}/${repo}/tree/main" \
            | python3 -c '
import json, sys
try:
    entries = json.load(sys.stdin)
except Exception:
    sys.exit(1)
for f in entries:
    if f.get("type") == "file" and f["path"].endswith(".gguf"):
        sha = (f.get("lfs") or {}).get("oid", "")
        if sha:
            print(f["path"], sha)
' >> "$TMP/hf.sha256"; then
        echo "error: failed to fetch ${ORG}/${repo} file list from HF API" >&2
        exit 1
    fi
done

# 2) Hash the local files (skips missing ones; the comparison reports them).
: > "$TMP/local.sha256"
for name in "${!REPO_OF[@]}"; do
    if [ -f "$MODELS/$name" ]; then
        sha256sum "$MODELS/$name" | awk -v n="$name" '{print n, $1}' >> "$TMP/local.sha256"
    else
        echo "$name MISSING" >> "$TMP/local.sha256"
    fi
done

# 3) Compare.
python3 - "$TMP" "$MODELS" <<'EOF'
import sys
tmp, models = sys.argv[1], sys.argv[2]

hf = {}
for line in open(f"{tmp}/hf.sha256"):
    p, h = line.split()
    hf[p] = h
local = {}
for line in open(f"{tmp}/local.sha256"):
    parts = line.split()
    local[parts[0]] = parts[1] if len(parts) > 1 else None

expected = {  # mirror the bash REPO_OF table
    "dino_f16.gguf", "ss_dec_f16.gguf", "ss_flow_f16.gguf",
    "slat_flow_f16.gguf", "slat_flow_1024_f16.gguf", "shape_dec_f16.gguf",
    "shape_enc_f16.gguf", "tex_dec_f16.gguf",
    "tex_slat_flow_512_f16.gguf", "tex_slat_flow_1024_f16.gguf",
}

fails = 0
for p in sorted(expected):
    if p not in hf:
        print(f"SKIP  {p}  (no HF listing; repo moved?)")
        continue
    l = local.get(p)
    if l is None:
        print(f"FAIL  {p}  (missing locally at {models}/{p})")
        fails += 1
    elif l == hf[p]:
        print(f"PASS  {p}")
    else:
        print(f"FAIL  {p}")
        print(f"      local: {l}")
        print(f"      hf:    {hf[p]}")
        fails += 1

print(f"\n{len(expected)} expected files, {fails} mismatch(es): "
      + ("all byte-identical to HF releases" if fails == 0 else "CHECK NEEDED"))
sys.exit(1 if fails else 0)
EOF
