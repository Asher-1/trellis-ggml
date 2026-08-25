#!/usr/bin/env python3
"""Extract texture/mesh stats from a GLB for cross-backend comparison.

Usage: python3 glb_stats.py a.glb b.glb
Prints per-GLB: verts/tris, texture size, texture mean/std per channel, bbox.
"""
import json, struct, sys, zlib
import numpy as np

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

def bufview(b, accessor, bufviews):
    bv = bufviews[accessor["bufferView"]]
    off = bv.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    comp = accessor["componentType"]
    sz = {5120:1,5121:1,5122:2,5123:2,5125:4,5126:4}[comp]
    n = accessor["count"]
    fmt = {5126:"f",5123:"H",5125:"I",5121:"B",5122:"h"}[comp]
    items = [accessor["type"].count("SCALAR"),]*0
    cols = {"SCALAR":1,"VEC2":2,"VEC3":3,"VEC4":4,"MAT4":16}[accessor["type"]]
    arr = np.frombuffer(b, dtype=np.dtype(fmt), count=n*cols, offset=off)
    return arr.reshape(n, cols)

def main():
    a_path, b_path = sys.argv[1], sys.argv[2]
    for p in (a_path, b_path):
        j, b = load_glb(p)
        meshes = j.get("meshes", [])
        nv = nt = 0
        for m in meshes:
            for prim in m["primitives"]:
                pos = prim["attributes"].get("POSITION")
                if pos is not None:
                    nv += j["accessors"][pos]["count"]
                idx = prim.get("indices")
                if idx is not None:
                    nt += j["accessors"][idx]["count"] // 3
        tex = j.get("textures", [])
        img = j.get("images", [])
        print(f"--- {p} ---")
        print(f"  verts={nv} tris={nt} textures={len(tex)} images={len(img)}")
        for i, im in enumerate(img):
            bv = j["bufferViews"][im["bufferView"]]
            png = b[bv["byteOffset"]:bv["byteOffset"]+bv["byteLength"]]
            try:
                import PIL.Image as PILImage
                im2 = PILImage.open(__import__("io").BytesIO(png))
                a = np.asarray(im2.convert("RGBA"), dtype=np.float32)
                print(f"  img[{i}] {im2.size} png={len(png)//1024}KB "
                      f"mean={a[...,:3].mean(axis=(0,1)).round(3)} "
                      f"std={a[...,:3].std(axis=(0,1)).round(3)} "
                      f"alpha_mean={a[...,3].mean().round(3)}")
            except Exception as e:
                print(f"  img[{i}] png={len(png)//1024}KB (decode failed: {e})")
        # bbox from first mesh
        for m in meshes:
            for prim in m["primitives"]:
                pos = prim["attributes"].get("POSITION")
                if pos is not None:
                    acc = j["accessors"][pos]
                    bv = j["bufferViews"][acc["bufferView"]]
                    v = bufview(b, acc, j["bufferViews"])
                    mn, mx = v.min(axis=0), v.max(axis=0)
                    print(f"  bbox=({mn[0]:.3f},{mn[1]:.3f},{mn[2]:.3f})..({mx[0]:.3f},{mx[1]:.3f},{mx[2]:.3f}) "
                          f"diag=({(mx-mn)[0]:.3f},{(mx-mn)[1]:.3f},{(mx-mn)[2]:.3f})")
                    break
            break

if __name__ == "__main__":
    main()
