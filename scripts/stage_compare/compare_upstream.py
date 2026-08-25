#!/usr/bin/env python3
"""Compare generated PBR GLBs against the upstream reference GLB.

Checks: geometry (verts/tris/bbox), material structure (baseColor texture,
metallicRoughness texture, alphaMode, doubleSided), texture resolution/size,
alpha-channel coverage, and file size.

Usage: python3 compare_upstream.py OUTDIR [--upstream UP.glb]
"""
import io, json, os, struct, sys, zlib
import numpy as np
from PIL import Image

UPSTREAM = "/home/asher/cloudViewer_data/trellis_T_upstream_q8.glb"

def load_glb(path):
    with open(path, "rb") as f:
        magic, version, length = struct.unpack("<III", f.read(12))
        chunks = []
        while f.tell() < length:
            clen, ctype = struct.unpack("<II", f.read(8))
            chunks.append((ctype, f.read(clen)))
    j = json.loads(chunks[0][1].decode())
    b = chunks[1][1] if len(chunks) > 1 else b""
    return j, b

def read_accessor(j, b, aidx):
    acc = j["accessors"][aidx]
    bv = j["bufferViews"][acc["bufferView"]]
    off = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
    comp = acc["componentType"]
    fmt = {5126: "f", 5123: "H", 5125: "I", 5121: "B", 5122: "h"}[comp]
    cols = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[acc["type"]]
    arr = np.frombuffer(b, dtype=np.dtype(fmt), count=acc["count"] * cols, offset=off)
    return arr.reshape(acc["count"], cols)

def glb_report(path):
    j, b = load_glb(path)
    size = os.path.getsize(path)
    r = {"file": os.path.basename(path), "size_mb": size / 1e6}
    # geometry
    prim = j["meshes"][0]["primitives"][0]
    pos = read_accessor(j, b, prim["attributes"]["POSITION"])
    idx = read_accessor(j, b, prim["indices"])
    r["verts"] = len(pos)
    r["tris"] = len(idx) // 3
    mn, mx = pos.min(axis=0), pos.max(axis=0)
    r["bbox_min"] = mn.round(4)
    r["bbox_max"] = mx.round(4)
    r["bbox_diag"] = (mx - mn).round(4)
    # material
    mat = j["materials"][prim.get("material", 0)]
    pbr = mat.get("pbrMetallicRoughness", {})
    r["alphaMode"] = mat.get("alphaMode", "OPAQUE")
    r["doubleSided"] = mat.get("doubleSided", False)
    r["has_baseColorTex"] = "baseColorTexture" in pbr
    r["has_mrTex"] = "metallicRoughnessTexture" in pbr
    r["baseColorFactor"] = pbr.get("baseColorFactor")
    # textures
    r["images"] = []
    for i, im in enumerate(j.get("images", [])):
        bv = j["bufferViews"][im["bufferView"]]
        png = b[bv["byteOffset"]: bv["byteOffset"] + bv["byteLength"]]
        img = Image.open(io.BytesIO(png))
        a = np.asarray(img.convert("RGBA"), dtype=np.float32) / 255.0
        cov = a[..., 3] > 0.05          # mesh-covered texels (padding is alpha=0)
        cov_a = a[..., 3][cov]
        cov_rgb = a[..., :3][cov]
        r["images"].append({
            "size": img.size, "png_kb": len(png) // 1024,
            "mean_rgb": cov_rgb.mean(axis=0).round(3) if len(cov_rgb) else None,
            "std_rgb": cov_rgb.std(axis=0).round(3) if len(cov_rgb) else None,
            "coverage_pct": round(100.0 * cov.mean(), 2),
            "alpha_mean_cov": round(float(cov_a.mean()), 3) if len(cov_a) else None,
            "translucent_pct_cov": round(100.0 * (cov_a < 0.95).mean(), 1) if len(cov_a) else 0.0,
        })
    return r

def fmt_r(r):
    lines = []
    lines.append(f"{r['file']}  ({r['size_mb']:.1f} MB)")
    lines.append(f"  verts={r['verts']:,}  tris={r['tris']:,}  "
                 f"bbox_diag={tuple(r['bbox_diag'])}")
    lines.append(f"  material: alphaMode={r['alphaMode']} doubleSided={r['doubleSided']} "
                 f"baseTex={r['has_baseColorTex']} mrTex={r['has_mrTex']}")
    for im in r["images"]:
        lines.append(f"  img {im['size'][0]}x{im['size'][1]}  png={im['png_kb']}KB  "
                     f"coverage={im['coverage_pct']}%  mean_rgb={im['mean_rgb']}  "
                     f"std_rgb={im['std_rgb']}  alpha_cov={im['alpha_mean_cov']}  "
                     f"translucent_cov={im['translucent_pct_cov']}%")
    return "\n".join(lines)

def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/coarse_test/e2e_final"
    up = UPSTREAM
    if "--upstream" in sys.argv:
        up = sys.argv[sys.argv.index("--upstream") + 1]

    print("=" * 78)
    print("PBR GLB vs Upstream comparison")
    print("=" * 78)
    ref = glb_report(up)
    print(fmt_r(ref))
    print()

    files = [
        ("CUDA q8",  f"{outdir}/cuda_q8_pbr.glb"),
        ("CUDA f16", f"{outdir}/cuda_f16_pbr.glb"),
        ("Vulkan q8", f"{outdir}/vulkan_q8_pbr.glb"),
        ("Vulkan f16", f"{outdir}/vulkan_f16_pbr.glb"),
        ("CPU q8",   f"{outdir}/cpu_q8_pbr.glb"),
        ("CPU f16",  f"{outdir}/cpu_f16_pbr.glb"),
    ]
    print("-" * 78)
    print(f"{'combo':<12} {'verts':>10} {'tris':>10} {'dverts%':>8} {'dtris%':>8} "
          f"{'sizeMB':>8} {'tex':>10} {'alphaMode':>9}")
    print("-" * 78)
    print(f"{'upstream':<12} {ref['verts']:>10,} {ref['tris']:>10,} {'-':>8} {'-':>8} "
          f"{ref['size_mb']:>8.1f} {str(ref['images'][0]['size']):>10} {ref['alphaMode']:>9}")
    for label, path in files:
        if not os.path.exists(path):
            print(f"{label:<12} MISSING ({path})")
            continue
        r = glb_report(path)
        dverts = 100.0 * (r["verts"] - ref["verts"]) / ref["verts"]
        dtris = 100.0 * (r["tris"] - ref["tris"]) / ref["tris"]
        tex = f"{r['images'][0]['size'][0]}x{r['images'][0]['size'][1]}" if r["images"] else "-"
        print(f"{label:<12} {r['verts']:>10,} {r['tris']:>10,} {dverts:>+8.2f} {dtris:>+8.2f} "
              f"{r['size_mb']:>8.1f} {tex:>10} {r['alphaMode']:>9}")
    print()

    # texture stats table for covered texels
    print("Texture stats (mesh-covered texels only):")
    print(f"{'combo':<12} {'mean_rgb':>24} {'std_rgb':>24} {'alpha':>7} {'transl%':>8}")
    print("-" * 78)
    for label, path in [("upstream", up)] + files:
        if not os.path.exists(path):
            continue
        r = glb_report(path)
        im = r["images"][0]
        print(f"{label:<12} {str(im['mean_rgb']):>24} {str(im['std_rgb']):>24} "
              f"{im['alpha_mean_cov']:>7} {im['translucent_pct_cov']:>8}")
    print()
    print("Detailed per-GLB:")
    for label, path in files:
        if os.path.exists(path):
            print(fmt_r(glb_report(path)))

if __name__ == "__main__":
    main()
