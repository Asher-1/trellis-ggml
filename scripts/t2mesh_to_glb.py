#!/usr/bin/env python3
"""t2mesh -> vertex-coloured GLB (non-interleaved layout)."""
import sys, struct, json
import numpy as np

def load(path):
    data = open(path, 'rb').read()
    magic = data[:8]
    nv, nt = struct.unpack('<II', data[8:16])
    o = 16
    verts = np.frombuffer(data, dtype='<f4', count=nv * 3, offset=o).astype(np.float32).reshape(-1, 3)
    o += nv * 12
    normals = np.frombuffer(data, dtype='<f4', count=nv * 3, offset=o).astype(np.float32).reshape(-1, 3)
    o += nv * 12
    pbr = None
    if magic == b'T2MESH03':
        pbr = np.frombuffer(data, dtype='<f4', count=nv * 6, offset=o).astype(np.float32).reshape(-1, 6)
        o += nv * 24
    tris = np.frombuffer(data, dtype='<i4', count=nt * 3, offset=o).astype(np.uint32).reshape(-1, 3)
    return verts, normals, pbr, tris

def to_glb(verts, normals, pbr, tris, out_path):
    nv = len(verts)
    # Vertex colours from PBR base_color + alpha
    colors = np.zeros((nv, 4), dtype=np.uint8)
    if pbr is not None:
        bc = pbr[:, :3].clip(0, 1)
        al = pbr[:, 3].clip(0, 1)
        colors[:, :3] = (bc * 255).astype(np.uint8)
        colors[:, 3] = (al * 255).astype(np.uint8)
    else:
        colors[:, :3] = 200
        colors[:, 3] = 255

    # Non-interleaved: all positions, all normals, all colours, then indices
    pos_bin = verts.tobytes()
    nrm_bin = normals.tobytes()
    clr_bin = colors.tobytes()  # 4 bytes per vertex (uint8)
    idx_bin = tris.astype(np.uint32).tobytes()

    # Single buffer: concatenate all parts
    parts = [pos_bin, nrm_bin, clr_bin, idx_bin]
    offsets = []
    o = 0
    for p in parts:
        offsets.append(o)
        o += len(p)
    buf = b''.join(parts)

    gltf = {
        "asset": {"version": "2.0", "generator": "t2mesh_to_glb"},
        "scene": 0, "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{
            "attributes": {"POSITION": 0, "NORMAL": 1, "COLOR_0": 2},
            "indices": 3
        }]}],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": nv, "type": "VEC3",
             "min": verts.min(0).tolist(), "max": verts.max(0).tolist()},
            {"bufferView": 1, "componentType": 5126, "count": nv, "type": "VEC3"},
            {"bufferView": 2, "componentType": 5121, "count": nv, "type": "VEC4",
             "normalized": True},
            {"bufferView": 3, "componentType": 5125, "count": len(tris) * 3, "type": "SCALAR"},
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": offsets[0], "byteLength": len(pos_bin)},
            {"buffer": 0, "byteOffset": offsets[1], "byteLength": len(nrm_bin)},
            {"buffer": 0, "byteOffset": offsets[2], "byteLength": len(clr_bin)},
            {"buffer": 0, "byteOffset": offsets[3], "byteLength": len(idx_bin)},
        ],
        "buffers": [{"byteLength": len(buf)}]
    }
    json_str = json.dumps(gltf, separators=(',', ':')).encode()
    json_pad = (4 - (len(json_str) & 3)) & 3
    json_str += b' ' * json_pad
    buf_pad = (4 - (len(buf) & 3)) & 3
    buf += b'\x00' * buf_pad

    hdr = bytearray(12)
    struct.pack_into('<I', hdr, 0, 0x46546C67)
    struct.pack_into('<I', hdr, 4, 2)
    struct.pack_into('<I', hdr, 8, 12 + 8 + len(json_str) + 8 + len(buf))
    with open(out_path, 'wb') as f:
        f.write(hdr)
        f.write(struct.pack('<II', len(json_str), 0x4E4F534A))
        f.write(json_str)
        f.write(struct.pack('<II', len(buf), 0x004E4942))
        f.write(buf)
    print(f"wrote {out_path}: {nv} verts, {len(tris)} tris, {len(buf)/1e6:.1f} MB")

if __name__ == '__main__':
    verts, normals, pbr, tris = load(sys.argv[1])
    to_glb(verts, normals, pbr, tris, sys.argv[2])
