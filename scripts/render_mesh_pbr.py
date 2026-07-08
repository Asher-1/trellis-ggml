#!/usr/bin/env python3
"""Render a T2MESH02 (per-vertex PBR) binary mesh to a PNG for eyeballing the
demo's textured output. Mirrors the viewer's orientation-independent shading."""
import struct, sys, numpy as np
from PIL import Image

path = sys.argv[1]; out = sys.argv[2] if len(sys.argv) > 2 else "mesh_pbr.png"
b = open(path, "rb").read()
magic = b[:8]; nv, nt = struct.unpack("<II", b[8:16]); o = 16
V = np.frombuffer(b, "<f4", 3*nv, o).reshape(-1, 3).astype(np.float64); o += 12*nv
o += 12*nv  # skip normals (recompute from faces)
pbr = None
if magic == b"T2MESH02":
    pbr = np.frombuffer(b, "<f4", 5*nv, o).reshape(-1, 5).astype(np.float64); o += 20*nv
F = np.frombuffer(b, "<i4", 3*nt, o).reshape(-1, 3)
C = pbr[:, 0:3] if pbr is not None else np.full((nv, 3), 0.65)
metal = pbr[:, 3] if pbr is not None else np.zeros(nv)
print(f"{magic} {nv:,} verts  base_color mean {C.mean(0).round(3)}", flush=True)

fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
N = np.zeros_like(V)
for k in range(3): np.add.at(N, F[:, k], fn)
N /= np.linalg.norm(N, axis=1, keepdims=True) + 1e-9

SS = 2; W = H = 720 * SS
c = (V.max(0) + V.min(0)) / 2; Vc = V - c; Vc *= 0.92 / np.abs(Vc).max()
l1 = np.array([0.5, 0.8, 0.6]); l1 /= np.linalg.norm(l1)
l2 = np.array([-0.6, -0.2, -0.7]); l2 /= np.linalg.norm(l2)


def R(el, az):
    a, e = np.radians(az), np.radians(el)
    Ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0, np.sin(e), np.cos(e)]])
    return Rx @ Ry


def render(el, az):
    r = R(el, az); P = Vc @ r.T; Nr = N @ r.T
    sx = ((P[:, 0]*.5+.5)*(W-1)).astype(np.int32); sy = ((.5-P[:, 1]*.5)*(H-1)).astype(np.int32)
    z = P[:, 2]
    d = np.abs(Nr @ l1)*.75 + np.abs(Nr @ l2)*.25
    diff = C * (0.25 + 0.75*d)[:, None] * (1 - 0.7*metal)[:, None]
    sp = np.clip(d, 0, 1)[:, None]**12 * (0.3 + 0.7*metal)[:, None]
    col = np.clip(diff + sp*np.where(metal[:, None] > 0.5, C, 0.16), 0, 1)
    img = np.ones((H, W, 3)); zb = np.full((H, W), -1e9)
    idx = np.argsort(z); sx, sy, z, col = sx[idx], sy[idx], z[idx], col[idx]
    for dx in range(SS+1):
        for dy in range(SS+1):
            xx = sx+dx; yy = sy+dy
            m = (xx >= 0) & (xx < W) & (yy >= 0) & (yy < H)
            img[yy[m], xx[m]] = col[m]
    return (img**(1/2.2)*255).reshape(H//SS, SS, W//SS, SS, 3).mean((1, 3)).astype(np.uint8)


row = np.concatenate([render(12, 25), render(12, 150), render(65, 20)], 1)
Image.fromarray(row).save(out)
print("wrote", out, flush=True)
