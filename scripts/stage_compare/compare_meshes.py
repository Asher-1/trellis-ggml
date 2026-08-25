#!/usr/bin/env python3
"""Compare two T2MESH01/02/03 files: verts/tris counts, bbox, axis alignment.

Usage: python3 compare_meshes.py a.t2mesh b.t2mesh [label]
Prints per-mesh stats and the diff summary.
"""
import numpy as np
import sys

def load_t2mesh(path):
    with open(path, "rb") as f:
        magic = f.read(8)
        nv, nt = np.frombuffer(f.read(8), dtype=np.uint32)
        verts = np.frombuffer(f.read(int(nv) * 12), dtype=np.float32).reshape(-1, 3)
        normals = np.frombuffer(f.read(int(nv) * 12), dtype=np.float32).reshape(-1, 3)
        pbr = None
        if magic[:7] == b"T2MESH03":
            pbr = np.frombuffer(f.read(int(nv) * 24), dtype=np.float32).reshape(-1, 6)
        tris = np.frombuffer(f.read(int(nt) * 12), dtype=np.int32).reshape(-1, 3)
    return magic, verts, normals, pbr, tris

def stats(verts, tris, pbr, name):
    mi, ma = verts.min(axis=0), verts.max(axis=0)
    diag = ma - mi
    # principal axis = longest bbox axis
    ax = int(np.argmax(diag))
    print(f"  [{name:<10}] nv={len(verts):>9} nt={len(tris):>9} pbr={'Y' if pbr is not None else 'N':1} "
          f"bbox=({mi[0]:.3f},{mi[1]:.3f},{mi[2]:.3f})..({ma[0]:.3f},{ma[1]:.3f},{ma[2]:.3f}) "
          f"diag=({diag[0]:.3f},{diag[1]:.3f},{diag[2]:.3f}) axis={ax}")
    return mi, ma

def main():
    a_path, b_path = sys.argv[1], sys.argv[2]
    la = sys.argv[3] if len(sys.argv) > 3 else "A"
    lb = sys.argv[4] if len(sys.argv) > 4 else "B"
    magic_a, va, na, pa, ta = load_t2mesh(a_path)
    magic_b, vb, nb, pb, tb = load_t2mesh(b_path)
    print(f"magic: {magic_a[:7].decode()} vs {magic_b[:7].decode()}")
    mia, maa = stats(va, ta, pa, la)
    mib, mab = stats(vb, tb, pb, lb)
    dv = abs(len(va) - len(vb)) / len(vb) * 100
    dt = abs(len(ta) - len(tb)) / len(tb) * 100
    print(f"  verts diff: {dv:.2f}%   tris diff: {dt:.2f}%")
    # bbox center & size
    ca, cb = (mia + maa) / 2, (mib + mab) / 2
    sa, sb = maa - mia, mab - mib
    print(f"  bbox center diff: {np.abs(ca - cb).max():.4f}   size diff: {np.abs(sa - sb).max():.4f}")
    print(f"  bbox axis (A): {int(np.argmax(sa))}  (B): {int(np.argmax(sb))}  (z=2 expected)")
    if pa is not None and pb is not None and len(va) == len(vb):
        d = np.abs(pa.astype(np.float64) - pb.astype(np.float64))
        rel = np.sqrt(np.sum(d**2) / np.sum(pb.astype(np.float64)**2))
        print(f"  pbr (same nv): relL2={rel:.4e} max|d|={d.max():.4e} mean|d|={d.mean():.4e}")
    elif pa is not None and pb is not None:
        print(f"  pbr: nv differ ({len(va)} vs {len(vb)}), cannot elementwise-compare")
        # nearest-vertex comparison on a subsample
        rng = np.random.default_rng(0)
        idx = rng.choice(min(len(va), len(vb)), min(20000, len(va), len(vb)), replace=False)
        print(f"  (subsampled nearest-vertex texture comparison skipped for now)")

if __name__ == "__main__":
    main()
