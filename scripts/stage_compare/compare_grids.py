#!/usr/bin/env python3
"""Compare two .t2grid files voxel-by-voxel for the 7-channel features."""
import numpy as np
import struct
import sys

def load_t2grid(path):
    """Returns (res, nvox, feats_np, coords_np). feats: float32 [nvox, 7], coords: int32 [nvox, 3]."""
    with open(path, 'rb') as f:
        magic = f.read(8)
        assert magic == b'T2GRID01', f"bad magic: {magic}"
        res, nvox = struct.unpack('<II', f.read(8))
        feats = np.frombuffer(f.read(nvox * 7 * 4), dtype=np.float32).reshape(nvox, 7)
        coords = np.frombuffer(f.read(nvox * 3 * 4), dtype=np.int32).reshape(nvox, 3)
    return res, nvox, feats, coords

print("=" * 70)
print("Dual Grid Comparison: CPU Q8 vs CUDA Q8 (exact, no TF32)")
print("=" * 70)

res_cpu, nv_cpu, f_cpu, c_cpu = load_t2grid('/tmp/coarse_test/v512_cpu_q8.t2grid')
res_cuda, nv_cuda, f_cuda, c_cuda = load_t2grid('/tmp/coarse_test/v512_cuda_q8.t2grid')

assert res_cpu == res_cuda, f"res mismatch: {res_cpu} vs {res_cuda}"
print(f"Grid resolution: {res_cpu}")
print(f"CPU voxels:  {nv_cpu:,}")
print(f"CUDA voxels: {nv_cuda:,}")
print(f"Diff: {nv_cuda - nv_cpu:+d} ({100*(nv_cuda/nv_cpu - 1):+.4f}%)")
print()

# Build coordinate -> index maps
cpu_idx = {tuple(c): i for i, c in enumerate(c_cpu)}
cuda_idx = {tuple(c): i for i, c in enumerate(c_cuda)}

common = []
only_cpu = []
only_cuda = []
for i, c in enumerate(c_cpu):
    k = tuple(c)
    if k in cuda_idx:
        common.append((i, cuda_idx[k]))
    else:
        only_cpu.append(i)

for i, c in enumerate(c_cuda):
    k = tuple(c)
    if k not in cpu_idx:
        only_cuda.append(i)

n_common = len(common)
n_only_cpu = len(only_cpu)
n_only_cuda = len(only_cuda)
print(f"Common voxels:       {n_common:,} ({100*n_common/nv_cpu:.2f}% of CPU)")
print(f"CPU-only voxels:     {n_only_cpu:,} ({100*n_only_cpu/nv_cpu:.2f}% of CPU)")
print(f"CUDA-only voxels:    {n_only_cuda:,} ({100*n_only_cuda/nv_cuda:.2f}% of CUDA)")
print()

# ── Compare 7-channel feats on common voxels ────────────────────────────
print("-" * 70)
print("7-channel Feature Analysis on Common Voxels")
print("-" * 70)

common_cpu = np.array([f_cpu[i] for i, _ in common], dtype=np.float32)
common_cuda = np.array([f_cuda[i] for _, i in common], dtype=np.float32)

diff = common_cuda - common_cpu
abs_diff = np.abs(diff)
rel_diff = abs_diff / (np.abs(common_cpu) + 1e-10)

for ch in range(7):
    d = diff[:, ch]
    ad = abs_diff[:, ch]
    rd = rel_diff[:, ch]
    print(f"  ch[{ch}]: mean(diff)={np.mean(d):+.6e}  std(diff)={np.std(d):.6e}  "
          f"max|diff|={np.max(ad):.6e}  L2={np.linalg.norm(d)/np.linalg.norm(common_cpu[:,ch]):.6e}  "
          f"mean(rel)={np.mean(rd):.6e}  max(rel)={np.max(rd):.6e}")
    
total_diff = common_cuda - common_cpu  # [common, 7]
total_norm = np.linalg.norm(total_diff)
total_ref_norm = np.linalg.norm(common_cpu)
print(f"\n  ALL:     relL2={total_norm/total_ref_norm:.6e}")
print(f"  ALL:     max|diff|={np.max(np.abs(total_diff)):.6e}")
print()

# ── Analyze CPU-only voxels (false negatives in CUDA) ──────────────────
print("-" * 70)
print("CPU-Only Voxel Feature Stats (CUDA missed these)")
print("-" * 70)
if n_only_cpu > 0:
    cpu_only_feats = f_cpu[only_cpu]
    for ch in range(7):
        vals = cpu_only_feats[:, ch]
        print(f"  ch[{ch}]: mean={np.mean(vals):+.6f}  std={np.std(vals):.6f}  min={np.min(vals):+.6f}  max={np.max(vals):+.6f}")
    # Channel 6 is typically the occupancy logit
    occ_cpu_only = cpu_only_feats[:, 6]
    print(f"  occupancy(ch6): mean={np.mean(occ_cpu_only):+.6f}  min={np.min(occ_cpu_only):+.6f}  max={np.max(occ_cpu_only):+.6f}")
    print(f"  occupancy close to 0: cnt={np.sum(occ_cpu_only > -0.1):,} / {n_only_cpu:,}")
else:
    print("  (none)")

print()

# ── Analyze CUDA-only voxels (false positives in CUDA) ─────────────────
print("-" * 70)
print("CUDA-Only Voxel Feature Stats (CPU missed these)")
print("-" * 70)
if n_only_cuda > 0:
    cuda_only_feats = f_cuda[only_cuda]
    for ch in range(7):
        vals = cuda_only_feats[:, ch]
        print(f"  ch[{ch}]: mean={np.mean(vals):+.6f}  std={np.std(vals):+.6f}  min={np.min(vals):+.6f}  max={np.max(vals):+.6f}")
    occ_cuda_only = cuda_only_feats[:, 6]
    print(f"  occupancy(ch6): mean={np.mean(occ_cuda_only):+.6f}  min={np.min(occ_cuda_only):+.6f}  max={np.max(occ_cuda_only):+.6f}")
    print(f"  occupancy close to 0: cnt={np.sum(occ_cuda_only > -0.1):,} / {n_only_cuda:,}")
else:
    print("  (none)")

print()

# ── Occupancy threshold analysis ─────────────────────────────────────────
print("-" * 70)
print("Occupancy Channel (ch6) Distribution")
print("-" * 70)
occ_cpu = f_cpu[:, 6]
occ_cuda = f_cuda[:, 6]
print(f"  CPU:   mean={np.mean(occ_cpu):+.4f}  std={np.std(occ_cpu):.4f}  min={np.min(occ_cpu):+.4f}  max={np.max(occ_cpu):+.4f}")
print(f"  CUDA: mean={np.mean(occ_cuda):+.4f}  std={np.std(occ_cuda):.4f}  min={np.min(occ_cuda):+.4f}  max={np.max(occ_cuda):+.4f}")

# Cross-check: for common voxels, what's the occupancy difference?
common_occ_cpu = common_cpu[:, 6]
common_occ_cuda = common_cuda[:, 6]
occ_diff = common_occ_cuda - common_occ_cpu
print(f"  Common voxels: occ diff mean={np.mean(occ_diff):+.6e}  std={np.std(occ_diff):.6e}  max|diff|={np.max(np.abs(occ_diff)):.6e}")

# Near-threshold occupancy comparison
near_th_cpu = np.sum(np.abs(occ_cpu) < 0.5)
near_th_cuda = np.sum(np.abs(occ_cuda) < 0.5)
print(f"  Voxels with |occ|<0.5: CPU={near_th_cpu:,}  CUDA={near_th_cuda:,}")
near_th_cpu_low = np.sum(occ_cpu < 0.5)
near_th_cuda_low = np.sum(occ_cuda < 0.5)
print(f"  Voxels with occ<0.5:   CPU={near_th_cpu_low:,}  CUDA={near_th_cuda_low:,}")

# For CPU-only voxels: how many have occupancy close to threshold?
if n_only_cpu > 0:
    close = np.sum(occ_cpu_only > -0.5)
    print(f"  CPU-only with occ>-0.5: {close:,} / {n_only_cpu:,}")
    far = np.sum(occ_cpu_only <= -0.5)
    print(f"  CPU-only with occ<=-0.5: {far:,} / {n_only_cpu:,}")

print()

# ── Headline metric summary ─────────────────────────────────────────────
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  nvox CPU:  {nv_cpu:,}")
print(f"  nvox CUDA: {nv_cuda:,}")
print(f"  Δnvox:     {nv_cuda - nv_cpu:+d} ({100*(nv_cuda/nv_cpu - 1):+.4f}%)")
print(f"  Common feats relL2: {total_norm/total_ref_norm:.6e}")
print(f"  Shared voxels:      {n_common:,} ({100*n_common/nv_cpu:.2f}%)")
print(f"  CPU-only voxels:    {n_only_cpu:,}")
print(f"  CUDA-only voxels:   {n_only_cuda:,}")