#!/usr/bin/env python3
"""Debug: reproduce dino embd tap via numpy conv and compare to reference."""
import sys, struct
import numpy as np
from safetensors.torch import load_file

def read_gguf_tensors(path, want_names=None):
    data = open(path, 'rb').read()
    n_tensors, n_kv = struct.unpack('<QQ', data[8:24])
    off = 24
    def rs():
        nonlocal off
        l = struct.unpack('<Q', data[off:off+8])[0]; off += 8
        s = data[off:off+l].decode(); off += l
        return s
    def rv(t):
        nonlocal off
        if t in (0,1,7): off += 1
        elif t in (2,3): off += 2
        elif t in (4,5,6): off += 4
        elif t == 8: rs()
        elif t == 9:
            typ = struct.unpack('<I', data[off:off+4])[0]; off += 4
            n = struct.unpack('<Q', data[off:off+8])[0]; off += 8
            for _ in range(n): rv(typ)
    for _ in range(n_kv):
        rs(); t = struct.unpack('<I', data[off:off+4])[0]; off += 4; rv(t)
    tensors = {}
    for _ in range(n_tensors):
        name = rs()
        nd = struct.unpack('<I', data[off:off+4])[0]; off += 4
        dims = list(struct.unpack('<'+'Q'*nd, data[off:off+8*nd])); off += 8*nd
        gtype = struct.unpack('<I', data[off:off+4])[0]; off += 4
        t_off = struct.unpack('<Q', data[off:off+8])[0]; off += 8
        tensors[name] = (gtype, dims, t_off)
    data_start = (off + 31) & ~31
    out = {}
    for name, (gtype, dims, t_off) in tensors.items():
        if want_names and name not in want_names:
            continue
        nelem = int(np.prod(dims))
        o = data_start + t_off
        if gtype == 0:
            arr = np.frombuffer(data[o:o+4*nelem], dtype='<f4')
        elif gtype == 1:
            arr = np.frombuffer(data[o:o+2*nelem], dtype='<f2').astype(np.float32)
        else:
            continue
        out[name] = arr.reshape(tuple(reversed(dims)))
    return out

def np_conv2d(x, w, b, stride):
    # x: [IC, H, W], w: [OC, IC, KH, KW] (pytorch layout)
    IC, H, W = x.shape
    OC, IC2, KH, KW = w.shape
    Ho = (H - KH) // stride + 1
    Wo = (W - KW) // stride + 1
    out = np.zeros((OC, Ho, Wo), dtype=np.float64)
    for oc in range(OC):
        acc = np.zeros((Ho, Wo), dtype=np.float64)
        for ic in range(IC):
            for kh in range(KH):
                for kw in range(KW):
                    acc += x[ic, kh:kh+Ho*stride:stride, kw:kw+Wo*stride:stride] * w[oc, ic, kh, kw]
        out[oc] = acc + b[oc]
    return out

ref = read_gguf_tensors('dumps/reference_dino.gguf',
                        ['pixel_values', 'embd', 'embeddings.patch_embeddings.weight',
                         'embeddings.patch_embeddings.bias', 'embeddings.cls_token',
                         'embeddings.register_tokens'])
sd = load_file('models/dinov3-vitl16/model.safetensors')

pix = ref['pixel_values']  # flat [1,3,512,512] torch CHW order
print('pix raw shape', pix.shape)
x = pix.reshape(3, 512, 512).astype(np.float64)  # [3,512,512] CHW
w = np.asarray(sd['embeddings.patch_embeddings.weight'], dtype=np.float64)  # [OC,IC,KH,KW]
b = np.asarray(sd['embeddings.patch_embeddings.bias'], dtype=np.float64)
print('pix shape', pix.shape, 'w shape', w.shape, 'stride 16')

conv = np_conv2d(x, w, b, 16)  # [OC, 32, 32]
print('conv out shape', conv.shape, 'range', conv.min(), conv.max())
P = 32 * 32
tokens = conv.reshape(conv.shape[0], P).T  # [P, C]
cls = np.asarray(sd['embeddings.cls_token'], dtype=np.float64).reshape(1, -1)
reg = np.asarray(sd['embeddings.register_tokens'], dtype=np.float64).reshape(4, -1)
h = np.concatenate([cls, reg, tokens], axis=0)  # [1029, C]
print('h shape', h.shape)

ref_embd = ref['embd']  # [1,1029,1024]
ref_embd = ref_embd[0]
d = np.abs(h - ref_embd)
print('embd: max|d|=%.6e mean=%.6e' % (d.max(), d.mean()))

# also check just the conv part against reference embd (last 1024 rows)
conv_only = h[5:]
d2 = np.abs(conv_only - ref_embd[5:])
print('conv-only vs ref: max|d|=%.6e mean=%.6e' % (d2.max(), d2.mean()))
d3 = np.abs(h[:5] - ref_embd[:5])
print('prefix(cls+reg) vs ref: max|d|=%.6e' % d3.max())
