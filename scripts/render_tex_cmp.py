#!/usr/bin/env python3
"""Render GLBs with their baked baseColor atlas sampled per-vertex."""
import sys, numpy as np, trimesh
from PIL import Image, ImageDraw

def load(path):
    scene = trimesh.load(path, process=False)
    if isinstance(scene, trimesh.Trimesh):
        mesh = scene
    else:
        mesh = trimesh.util.concatenate([g for g in scene.geometry.values()])
    V = np.asarray(mesh.vertices, np.float64)
    F = np.asarray(mesh.faces, np.int64)
    assert F.max() < len(V), f"{path}: face idx {F.max()} >= nv {len(V)}"
    uv = np.asarray(mesh.visual.uv, np.float64)
    img = np.asarray(mesh.visual.material.baseColorTexture.convert("RGB"), np.float64) / 255.0
    TH, TW = img.shape[:2]
    x = np.clip((uv[:, 0] % 1.0) * (TW - 1), 0, TW - 1).astype(np.int32)
    y = np.clip((uv[:, 1] % 1.0) * (TH - 1), 0, TH - 1).astype(np.int32)
    C = img[y, x] ** 2.2  # linear
    N = np.zeros_like(V)
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    for k in range(3): np.add.at(N, F[:, k], fn)
    N /= np.linalg.norm(N, axis=1, keepdims=True) + 1e-9
    return V, F, N, C

def Rmat(el, az):
    a, e = np.radians(az), np.radians(el)
    Ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0, np.sin(e), np.cos(e)]])
    return Rx @ Ry

def render(V, F, N, C, el, az, W=520, H=520):
    c = (V.max(0) + V.min(0)) / 2
    Vc = V - c; Vc *= 0.88 / np.abs(Vc).max()
    r = Rmat(el, az)
    P = Vc @ r.T; Nr = N @ r.T
    Nr /= np.linalg.norm(Nr, axis=1, keepdims=True) + 1e-9
    sx = np.clip(((P[:, 0]*.5+.5)*(W-1)).astype(np.int32), 0, W-1)
    sy = np.clip(((.5-P[:, 1]*.5)*(H-1)).astype(np.int32), 0, H-1)
    z = P[:, 2]
    LC = [(np.array([.32, .55, .77]), np.array([1., .96, .88]), .95),
          (np.array([-.65, .20, .45]), np.array([.70, .80, 1.]), .5),
          (np.array([.10, -.75, .35]), np.array([.85, .85, .90]), .3)]
    light = np.full((len(V), 3), 0.35)
    for Ld, Lc, w in LC:
        l = Ld / np.linalg.norm(Ld)
        ndl = np.abs(Nr @ l)
        light += Lc[None] * w * ndl[:, None]
    col = np.clip(C * light, 0, 1) ** (1/2.2)
    order = np.argsort(z[F].mean(1))
    im = Image.new("RGB", (W, H), (255, 255, 255))
    dr = ImageDraw.Draw(im)
    fC = (col[F].mean(1) * 255).astype(np.uint8)
    fP = np.stack([sx[F], sy[F]], -1)
    for t in order:
        dr.polygon([tuple(p) for p in fP[t]], fill=tuple(fC[t]))
    return np.asarray(im)

paths = sys.argv[1:-1]
out = sys.argv[-1]
views = [(18, 30), (18, 90), (70, 20)]
rows = []
for p in paths:
    V, F, N, C = load(p)
    tiles = [render(V, F, N, C, el, az) for el, az in views]
    label = p.split("/")[-1]
    tile = np.concatenate(tiles, 1)
    bar = np.full((40, tile.shape[1], 3), 30, np.uint8)
    row = np.concatenate([bar, tile], 0)
    ImageDraw.Draw(Image.fromarray(row)).text((8, 8), label, fill=(255, 255, 255))
    rows.append(row)
canvas = rows[0]
for r in rows[1:]:
    canvas = np.concatenate([canvas, r], 0)
Image.fromarray(canvas).save(out)
print("wrote", out)
