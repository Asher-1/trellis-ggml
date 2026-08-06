#!/usr/bin/env python3
"""Convert GGUF f16 models back to safetensors for PyTorch upstream inference.

The GGUF files use the original tensor names from the safetensors checkpoints.
This script reads the GGUF tensors and writes them as f16 safetensors that the
upstream TRELLIS.2 pipeline can load directly.

Usage:
    python scripts/gguf_to_safetensors.py
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import gguf
import torch
from safetensors.torch import save_file

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(REPO, "models")

# Mapping: (gguf_file, output_dir, output_stem)
CONVERSIONS = [
    ("ss_flow_f16.gguf",
     "TRELLIS.2-4B/ckpts", "ss_flow_img_dit_1_3B_64_bf16"),
    ("slat_flow_f16.gguf",
     "TRELLIS.2-4B/ckpts", "slat_flow_img2shape_dit_1_3B_512_bf16"),
    ("slat_flow_1024_f16.gguf",
     "TRELLIS.2-4B/ckpts", "slat_flow_img2shape_dit_1_3B_1024_bf16"),
    ("tex_slat_flow_512_f16.gguf",
     "TRELLIS.2-4B/ckpts", "slat_flow_imgshape2tex_dit_1_3B_512_bf16"),
    ("tex_slat_flow_1024_f16.gguf",
     "TRELLIS.2-4B/ckpts", "slat_flow_imgshape2tex_dit_1_3B_1024_bf16"),
    ("shape_dec_f16.gguf",
     "TRELLIS.2-4B/ckpts", "shape_dec_next_dc_f16c32_fp16"),
    ("shape_enc_f16.gguf",
     "TRELLIS.2-4B/ckpts", "shape_enc_next_dc_f16c32_fp16"),
    ("tex_dec_f16.gguf",
     "TRELLIS.2-4B/ckpts", "tex_dec_next_dc_f16c32_fp16"),
    ("ss_dec_f16.gguf",
     "TRELLIS-image-large/ckpts", "ss_dec_conv3d_16l8_fp16"),
]


def convert_gguf_to_safetensors(gguf_path, output_path):
    """Read GGUF tensors and write as safetensors f16."""
    print(f"  Reading {gguf_path}...")
    reader = gguf.GGUFReader(gguf_path)

    state_dict = {}
    for tensor in reader.tensors:
        name = tensor.name
        # Read tensor data
        data = tensor.data
        # Convert to torch tensor
        if data.dtype == np.float16:
            t = torch.from_numpy(data.copy()).half()
        elif data.dtype == np.float32:
            t = torch.from_numpy(data.copy()).float()
        else:
            # Try to interpret as float
            t = torch.from_numpy(data.astype(np.float32).copy())
        state_dict[name] = t

    print(f"  Writing {output_path} ({len(state_dict)} tensors)...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_file(state_dict, output_path)
    print(f"  Done: {os.path.getsize(output_path) / 1e9:.2f} GB")


def main():
    for gguf_name, out_subdir, out_stem in CONVERSIONS:
        gguf_path = os.path.join(MODELS, gguf_name)
        out_path = os.path.join(MODELS, out_subdir, out_stem + ".safetensors")

        if not os.path.exists(gguf_path):
            print(f"SKIP {gguf_name} (not found)")
            continue

        if os.path.exists(out_path):
            print(f"SKIP {out_stem} (already exists)")
            continue

        print(f"CONVERT {gguf_name} -> {out_stem}.safetensors")
        convert_gguf_to_safetensors(gguf_path, out_path)

    # Also copy JSON configs if they exist in the download
    import shutil
    json_src_dir = os.path.join(MODELS, "TRELLIS.2-4B", "ckpts")
    if os.path.isdir(json_src_dir):
        for f in os.listdir(json_src_dir):
            if f.endswith(".json"):
                print(f"Config already at {json_src_dir}/{f}")
    else:
        print("NOTE: JSON configs not yet downloaded. You may need to download them separately.")

    print("\nAll conversions complete!")


if __name__ == "__main__":
    main()
