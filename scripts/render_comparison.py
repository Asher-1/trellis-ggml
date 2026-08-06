#!/usr/bin/env python3
"""Render multiple GLB meshes side-by-side for comparison.

All meshes are rendered with the same lighting/shading (no textures,
uniform gray base color with PBR-like shading) so geometry differences
are clearly visible.

Usage:
    python scripts/render_comparison.py out.png input1.glb input2.glb [input3.glb ...]
    python scripts/render_comparison.py out.png --labels "PyTorch,ggml CUDA,ggml Vulkan" input1.glb input2.glb input3.glb
"""
import sys, numpy as np, trimesh
from PIL import Image, ImageDraw, ImageFont

def load_mesh(path):
    scene = trimesh.load(path, process=False)
    mesh = scene if isinstance(scene, trimesh.Trimesh) else list(scene.geometry.values())[0]
    V = np.asarray(mesh.vertices, np.float64)
    F = np.asarray(mesh.faces, np.int64)
    # Compute vertex normals
    N = np.zeros_like(V)
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    for k in range(3): np.add.at(N, F[:, k], fn)
    N /= np.linalg.norm(N, axis=1, keepdims=True) + 1e-9
    return V, F, N

def Rmat(el, az):
    a, e = np.radians(az), np.radians(el)
    Ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0, np.sin(e), np.cos(e)]])
    return Rx @ Ry

def render_mesh(V, F, N, el, az, W=480, H=480):
    """Render a mesh with uniform gray PBR-like shading."""
    # Center and scale
    c = (V.max(0) + V.min(0)) / 2
    Vc = V - c
    Vc *= 0.88 / np.abs(Vc).max()
    
    # Rotate
    r = Rmat(el, az)
    P = Vc @ r.T
    Nr = N @ r.T
    Nr /= np.linalg.norm(Nr, axis=1, keepdims=True) + 1e-9
    
    # Project
    sx = ((P[:, 0]*.5+.5)*(W-1)).astype(np.int32)
    sy = ((.5-P[:, 1]*.5)*(H-1)).astype(np.int32)
    z = P[:, 2]
    
    # Uniform gray base color (no textures)
    base_v = np.full((len(V), 3), 0.65)
    metal_v = np.zeros(len(V))
    rough_v = np.full(len(V), 0.45)
    
    # Lighting
    view = np.array([0., 0., 1.])
    nv_ = np.abs(Nr @ view)
    F0 = 0.04 * (1 - metal_v)[:, None] + base_v * metal_v[:, None]
    Fr = F0 + (1 - F0) * ((1 - nv_)[:, None]**5)
    shin = 6.0 + 214.0 * np.clip(1 - rough_v, 0, 1)**1.5
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
        h = l + view; h = h / np.linalg.norm(h)
        nh = np.abs(Nr @ h)
        specular += Lc[None] * w * (nh[:, None]**shin[:, None]) * sn[:, None] * ndl[:, None]
    
    albedo = base_v * (1 - metal_v)[:, None]
    hemi = np.abs(N[:, 1]) * 0.5 + 0.5
    amb = (1 - hemi)[:, None] * np.array([.20, .19, .22]) + hemi[:, None] * np.array([.55, .58, .66])
    col = np.clip(albedo * (amb * 0.55 + diffuse * 0.75) + specular * Fr + Fr * 0.25, 0, 1)
    
    # Rasterize (sorted by depth)
    img = np.ones((H, W, 3))
    idx = np.argsort(z)
    sxi, syi, coli = sx[idx], sy[idx], col[idx]
    m = (sxi >= 0) & (sxi < W) & (syi >= 0) & (syi < H)
    img[syi[m], sxi[m]] = coli[m]
    
    return (img**(1/2.2) * 255).astype(np.uint8)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('output', help='Output PNG path')
    ap.add_argument('inputs', nargs='+', help='Input GLB files')
    ap.add_argument('--labels', default=None, help='Comma-separated labels')
    ap.add_argument('--views', default='front,side,top', help='Views: comma-separated from front,side,top,back')
    args = ap.parse_args()
    
    view_angles = {
        'front': (12, 25),
        'side': (12, 150),
        'top': (65, 20),
        'back': (12, 200),
    }
    views = [view_angles[v.strip()] for v in args.views.split(',')]
    
    # Load all meshes
    meshes = []
    for path in args.inputs:
        V, F, N = load_mesh(path)
        meshes.append((V, F, N, path))
        print(f"Loaded: {path} ({len(V):,} verts, {len(F):,} faces)")
    
    labels = args.labels.split(',') if args.labels else [f"Mesh {i+1}" for i in range(len(meshes))]
    
    W, H = 480, 480
    n_meshes = len(meshes)
    n_views = len(views)
    
    # Create grid: rows = meshes, cols = views
    label_h = 40
    view_label_h = 30
    total_w = n_views * W
    total_h = n_meshes * (H + label_h) + view_label_h
    
    grid = np.ones((total_h, total_w, 3), dtype=np.uint8) * 240
    
    # Draw view labels at top
    img = Image.fromarray(grid)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except:
        font = ImageFont.load_default()
        font_small = font
    
    view_names = [v.strip() for v in args.views.split(',')]
    for j, name in enumerate(view_names):
        draw.text((j * W + W//2 - 30, 5), name.capitalize(), fill=(60, 60, 60), font=font)
    
    # Render each mesh from each view
    for i, (V, F, N, path) in enumerate(meshes):
        y_offset = view_label_h + i * (H + label_h)
        
        # Draw mesh label
        draw.text((10, y_offset + H + 8), labels[i], fill=(40, 40, 40), font=font)
        n_v, n_f = len(V), len(F)
        draw.text((10, y_offset + H + 26), f"{n_v:,} verts, {n_f:,} faces", fill=(100, 100, 100), font=font_small)
        
        for j, (el, az) in enumerate(views):
            render = render_mesh(V, F, N, el, az, W, H)
            x_offset = j * W
            img.paste(Image.fromarray(render), (x_offset, y_offset))
    
    img.save(args.output)
    print(f"Wrote {args.output} ({total_w}x{total_h})")

if __name__ == '__main__':
    main()
