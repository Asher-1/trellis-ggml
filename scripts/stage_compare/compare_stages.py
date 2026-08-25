#!/usr/bin/env python3
"""Compare t2_stage_probe outputs across backends/quantizations.

Usage: python3 compare_stages.py <dir>
Expects <dir>/{cpu,cuda,vulkan}_{q8,f16}.{cond,latent,occ}.bin
"""
import numpy as np
import sys, os

def load(path):
    return np.fromfile(path, dtype=np.float32)

def stats(got, ref, label):
    n = min(len(got), len(ref))
    g, r = got[:n].astype(np.float64), ref[:n].astype(np.float64)
    d = np.abs(g - r)
    rel_l2 = np.sqrt(np.sum(d**2) / np.sum(r**2)) if np.sum(r**2) > 0 else np.sqrt(np.sum(d**2))
    nbad = np.sum(d > 2e-3 + 2e-3 * np.abs(r))
    print(f"  [{label:<24}] n={n:<9} max|d|={d.max():.4e} mean|d|={d.mean():.4e} "
          f"relL2={rel_l2:.4e} n>2e-3+2e-3|r|={nbad}")
    return rel_l2

def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "/tmp/coarse_test"
    for stage, fname in [("DINO cond", "cond"), ("SS-flow latent", "latent"), ("occupancy", "occ")]:
        print(f"=== {stage} ===")
        data = {}
        for be in ("cpu", "cuda", "vulkan"):
            for q in ("q8", "f16"):
                p = os.path.join(d, f"{be}_{q}.{fname}.bin")
                if os.path.exists(p):
                    data[f"{be}_{q}"] = load(p)
        for be in ("cuda", "vulkan"):
            for q in ("q8", "f16"):
                k = f"{be}_{q}"
                if k in data and f"cpu_{q}" in data:
                    print(f"  {be}-{q} vs cpu-{q}:")
                    stats(data[k], data[f"cpu_{q}"], "same-quant GPU vs CPU")
        if "cpu_q8" in data and "cpu_f16" in data:
            print("  cpu-q8 vs cpu-f16 (quant noise baseline):")
            stats(data["cpu_q8"], data["cpu_f16"], "quant noise")
        for be in ("cuda", "vulkan"):
            k = f"{be}_f16"
            if k in data and "cpu_f16" in data:
                print(f"  {be}-f16 vs cpu-f16 (f16 cross-backend):")
                stats(data[k], data["cpu_f16"], "f16 GPU vs CPU")

if __name__ == "__main__":
    main()
