#!/usr/bin/env python3
"""Unified backend comparison renderer.

Reads meshes from GLB or T2MESH files and renders them side-by-side
with identical PBR-like shading (uniform clay color) so that geometry
differences are clearly visible.

Supports:
  - .glb  (trimesh)
  - .t2mesh / .bin  (native T2MESH01/02/03 wire format)

Usage:
    python scripts/render_backend_comparison.py \\
        --inputs outputs/pbr_e2e/pbr_official_T_e2e.glb outputs/pbr_e2e/pbr_official_T_retexture.glb \\
        --labels "ggml CUDA,ggml CUDA (retextured)" \\
        --output docs/backend_comparison.png
"""
import struct, sys, argparse
import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont

# ── t2mesh reader ──────────────────────────────────────────────────────
def read_t2mesh(path):
    """Read T2MESH01/02/03 binary → (verts, normals, tris, pbr_or_none)."""
    with open(path, 'rb') as f:
        magic = f.read(8)
        nv, nt = struct.unpack('<II', f.read(8))
        verts = np.frombuffer(f.read(nv * 3 * 4), dtype='<f4').reshape(nv, 3).astype(np.float64)
        normals = np.frombuffer(f.read(nv * 3 * 4), dtype='<f4').reshape(nv, 3).astype(np.float64)
        pbr = None
        if magic == b'T2MESH02':
            old = np.frombuffer(f.read(nv * 5 * 4), dtype='<f4').reshape(nv, 5)
            pbr = np.zeros((nv, 6), dtype=np.float32)
            pbr[:, :5] = old; pbr[:, 5] = 1.0
        elif magic == b'T2MESH03':
            pbr = np.frombuffer(f.read(nv * 6 * 4), dtype='<f4').reshape(nv, 6)
        tris = np.frombuffer(f.read(nt * 3 * 4), dtype='<i4').reshape(nt, 3).astype(np.int64)
    return verts, normals, tris, pbr

# ── mesh loaders ───────────────────────────────────────────────────────
def load_glb(path):
    scene = trimesh.load(path, process=False)
    mesh = scene if isinstance(scene, trimesh.Trimesh) else list(scene.geometry.values())[0]
    V = np.asarray(mesh.vertices, np.float64)
    F = np.asarray(mesh.faces, np.int64)
    N = np.zeros_like(V)
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    for k in range(3): np.add.at(N, F[:, k], fn)
    N /= np.linalg.norm(N, axis=1, keepdims=True) + 1e-9
    return V, N, F

def load_t2mesh(path):
    V, N, F, _ = read_t2mesh(path)
    # Recompute normals for safety
    N2 = np.zeros_like(V)
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    for k in range(3): np.add.at(N2, F[:, k], fn)
    N2 /= np.linalg.norm(N2, axis=1, keepdims=True) + 1e-9
    return V, N2, F

def load_mesh(path):
    if path.endswith('.t2mesh') or path.endswith('.bin'):
        return load_t2mesh(path)
    return load_glb(path)

# ── renderer (matches render_glb.py lighting) ─────────────────────────
def Rmat(el, az):
    a, e = np.radians(az), np.radians(el)
    Ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0, np.sin(e), np.cos(e)]])
    return Rx @ Ry

def render_view(V, F, N, el, az, W=600, H=600, SS=2):
    """Render mesh with uniform gray clay PBR shading + supersampling."""
    Ws, Hs = W * SS, H * SS
    c = (V.max(0) + V.min(0)) / 2
    Vc = V - c
    Vc *= 0.90 / np.abs(Vc).max()

    r = Rmat(el, az)
    P = Vc @ r.T
    Nr = N @ r.T
    Nr /= np.linalg.norm(Nr, axis=1, keepdims=True) + 1e-9

    sx = ((P[:, 0] * .5 + .5) * (Ws - 1)).astype(np.int32)
    sy = ((.5 - P[:, 1] * .5) * (Hs - 1)).astype(np.int32)
    z = P[:, 2]

    # Uniform clay material
    base_v = np.full((len(V), 3), 0.65)
    metal_v = np.zeros(len(V))
    rough_v = np.full(len(V), 0.45)

    view = np.array([0., 0., 1.])
    nv_ = np.abs(Nr @ view)
    F0 = 0.04 * (1 - metal_v)[:, None] + base_v * metal_v[:, None]
    Fr = F0 + (1 - F0) * ((1 - nv_)[:, None] ** 5)
    shin = 6.0 + 214.0 * np.clip(1 - rough_v, 0, 1) ** 1.5
    sn = (shin + 2.0) / (2 * np.pi)

    LC = [(np.array([.32, .55, .77]), np.array([1., .96, .88]), .85),
          (np.array([-.65, .20, .45]), np.array([.70, .80, 1.]), .45),
          (np.array([.10, -.75, .35]), np.array([.85, .85, .90]), .25)]

    diffuse = np.zeros_like(base_v)
    specular = np.zeros_like(base_v)
    for Ld, Lc, w in LC:
        l = Ld / np.linalg.norm(Ld)
        ndl = np.abs(Nr @ l)
        diffuse += Lc[None] * w * ndl[:, None]
        h = l + view; h /= np.linalg.norm(h)
        nh = np.abs(Nr @ h)
        specular += Lc[None] * w * (nh[:, None] ** shin[:, None]) * sn[:, None] * ndl[:, None]

    albedo = base_v * (1 - metal_v)[:, None]
    hemi = np.abs(N[:, 1]) * 0.5 + 0.5
    amb = (1 - hemi)[:, None] * np.array([.20, .19, .22]) + hemi[:, None] * np.array([.55, .58, .66])
    col = np.clip(albedo * (amb * 0.55 + diffuse * 0.75) + specular * Fr + Fr * 0.25, 0, 1)

    # Rasterize with SS supersampling
    img = np.ones((Hs, Ws, 3))
    idx = np.argsort(z)
    sxi, syi, coli = sx[idx], sy[idx], col[idx]
    for dx in range(SS + 1):
        for dy in range(SS + 1):
            xx = sxi + dx
            yy = syi + dy
            m = (xx >= 0) & (xx < Ws) & (yy >= 0) & (yy < Hs)
            img[yy[m], xx[m]] = coli[m]

    # Downsample + gamma
    return (img ** (1 / 2.2) * 255).reshape(H, SS, W, SS, 3).mean((1, 3)).astype(np.uint8)

# ── main ───────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', nargs='+', required=True)
    ap.add_argument('--labels', default=None)
    ap.add_argument('--output', default='docs/backend_comparison.png')
    ap.add_argument('--views', default='front,side,top')
    ap.add_argument('--cell', type=int, default=520, help='pixels per view')
    args = ap.parse_args()

    view_angles = {'front': (12, 25), 'side': (12, 150), 'top': (65, 20), 'back': (12, 200)}
    views = [view_angles[v.strip()] for v in args.views.split(',')]
    view_names = [v.strip() for v in args.views.split(',')]

    labels = args.labels.split(',') if args.labels else [f"Backend {i+1}" for i in range(len(args.inputs))]

    # Load all meshes
    meshes = []
    for p in args.inputs:
        V, N, F = load_mesh(p)
        meshes.append((V, F, N, p))
        print(f"  loaded {p}: {len(V):,} verts, {len(F):,} faces", flush=True)

    cell = args.cell
    n_cols = len(views)
    n_rows = len(meshes)
    label_h = 56          # space for label text below each row
    sep_h = 6             # colored separator between rows
    view_label_h = 40     # top header for view names
    total_w = n_cols * cell
    total_h = view_label_h + n_rows * (cell + label_h) + (n_rows - 1) * sep_h

    canvas = np.ones((total_h, total_w, 3), dtype=np.uint8) * 248
    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)

    try:
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
        font_info = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
        font_view = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except Exception:
        font_title = font_info = font_view = ImageFont.load_default()

    # Row accent colours
    accents = [(70, 130, 220), (50, 180, 80), (200, 100, 50)]  # blue, green, orange

    # ── top header bar ──
    draw.rectangle([(0, 0), (total_w, view_label_h - 1)], fill=(40, 40, 50))
    for j, name in enumerate(view_names):
        x = j * cell + cell // 2
        draw.text((x - 30, 10), name.capitalize(), fill=(240, 240, 240), font=font_view)

    # Render each mesh
    for i, (V, F, N, path) in enumerate(meshes):
        y_base = view_label_h + i * (cell + label_h + sep_h)
        accent = accents[i % len(accents)]

        # Draw separator line above row
        if i > 0:
            sy = y_base - sep_h
            draw.rectangle([(0, sy), (total_w, sy + sep_h - 1)], fill=accent)

        # Render views
        for j, (el, az) in enumerate(views):
            render = render_view(V, F, N, el, az, W=cell, H=cell, SS=2)
            img.paste(Image.fromarray(render), (j * cell, y_base))

        # Label area below renders
        ly = y_base + cell + 4
        # Accent dot
        draw.ellipse([(12, ly + 2), (24, ly + 14)], fill=accent)
        draw.text((30, ly), labels[i], fill=(30, 30, 30), font=font_title)
        nv, nf = len(V), len(F)
        draw.text((30, ly + 28), f"{nv:,} verts  |  {nf:,} faces", fill=(110, 110, 110), font=font_info)

    img.save(args.output)
    print(f"Wrote {args.output} ({total_w}x{total_h})", flush=True)

if __name__ == '__main__':
    main()
