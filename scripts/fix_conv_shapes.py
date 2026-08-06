#!/usr/bin/env python3
"""Fix conv weight shapes in converted safetensors.

The GGUF→safetensors conversion flattens 3D conv kernels: [Co, 3, 3, 3, Ci] → [Co, 27, Ci].
This script reshapes them back to the expected 5D format.
"""

import os
import torch
from safetensors.torch import load_file, save_file

MODELS = "models/TRELLIS.2-4B/ckpts"

# Files that have sparse conv weights needing reshape
FILES = [
    f"{MODELS}/shape_dec_next_dc_f16c32_fp16.safetensors",
    f"{MODELS}/tex_dec_next_dc_f16c32_fp16.safetensors",
]

# Also fix the SS decoder
SS_DEC = "models/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16.safetensors"


def fix_conv_weights(state_dict):
    """Reshape [Co, K^3, Ci] → [Co, K, K, K, Ci] for conv weights."""
    fixed = {}
    for name, tensor in state_dict.items():
        if '.conv' in name and name.endswith('.weight') and tensor.dim() == 3:
            Co, K3, Ci = tensor.shape
            if K3 == 27:  # 3x3x3
                K = 3
                new_shape = (Co, K, K, K, Ci)
                fixed[name] = tensor.reshape(new_shape)
                print(f"  Reshaped {name}: {list(tensor.shape)} -> {list(new_shape)}")
            else:
                fixed[name] = tensor
        else:
            fixed[name] = tensor
    return fixed


def main():
    for path in FILES + [SS_DEC]:
        if not os.path.exists(path):
            print(f"SKIP {path} (not found)")
            continue
        print(f"\nFixing {path}...")
        sd = load_file(path)
        fixed = fix_conv_weights(sd)
        save_file(fixed, path)
        print(f"  Saved ({len(fixed)} tensors)")

    print("\nDone!")


if __name__ == "__main__":
    main()
