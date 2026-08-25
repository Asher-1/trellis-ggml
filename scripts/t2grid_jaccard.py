#!/usr/bin/env python3
"""Voxel-set Jaccard comparison between two .t2grid sidecars (acceptance gate, see .AGENTS.md).

usage: python3 scripts/t2grid_jaccard.py A.t2grid B.t2grid
exit code 0 if Jaccard >= threshold (--thr, default 0.999), else 1.
"""
import struct, sys, argparse

def load(path):
    with open(path, 'rb') as f:
        data = f.read()
    assert data[:8] == b'T2GRID01', path
    res, nvox = struct.unpack_from('<II', data, 8)
    coords_off = 16 + nvox * 7 * 4  # skip float32 feats (7 per voxel)
    coords = set()
    for i in range(nvox):
        coords.add(struct.unpack_from('<iii', data, coords_off + i * 12))
    return res, coords

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('a'); ap.add_argument('b')
    ap.add_argument('--thr', type=float, default=0.999)
    args = ap.parse_args()
    ra, ca = load(args.a); rb, cb = load(args.b)
    inter, union = len(ca & cb), len(ca | cb)
    j = inter / union if union else 1.0
    print(f"res={ra}x{rb} nvox={len(ca)}/{len(cb)} inter={inter} Jaccard={j:.4f}")
    sys.exit(0 if j >= args.thr else 1)

if __name__ == '__main__':
    main()
