#!/usr/bin/env python3
"""Compare dino_f32.gguf weights against the safetensors source."""
import sys, struct
import numpy as np
from safetensors.torch import load_file

GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1

def read_gguf(path):
    with open(path, 'rb') as f:
        data = f.read()
    assert data[:4] == b'GGUF', 'not gguf'
    version = struct.unpack('<I', data[4:8])[0]
    n_tensors, n_kv = struct.unpack('<QQ', data[8:24])
    off = 24
    def read_str():
        nonlocal off
        l = struct.unpack('<Q', data[off:off+8])[0]; off += 8
        s = data[off:off+l].decode(); off += l
        return s
    def read_val(t):
        nonlocal off
        if t == 0:  # uint8
            v = data[off]; off += 1; return v
        elif t == 1:  # int8
            v = struct.unpack('<b', data[off:off+1])[0]; off += 1; return v
        elif t == 2:  # uint16
            v = struct.unpack('<H', data[off:off+2])[0]; off += 2; return v
        elif t == 3:  # int16
            v = struct.unpack('<h', data[off:off+2])[0]; off += 2; return v
        elif t == 4:  # uint32
            v = struct.unpack('<I', data[off:off+4])[0]; off += 4; return v
        elif t == 5:  # int32
            v = struct.unpack('<i', data[off:off+4])[0]; off += 4; return v
        elif t == 6:  # float32
            v = struct.unpack('<f', data[off:off+4])[0]; off += 4; return v
        elif t == 7:  # bool
            v = data[off]; off += 1; return bool(v)
        elif t == 8:  # string
            return read_str()
        elif t == 9:  # array
            typ = struct.unpack('<I', data[off:off+4])[0]; off += 4
            n = struct.unpack('<Q', data[off:off+8])[0]; off += 8
            return [read_val(typ) for _ in range(n)]
        else:
            raise ValueError(f'unknown kv type {t}')
    for _ in range(n_kv):
        k = read_str()
        t = struct.unpack('<I', data[off:off+4])[0]; off += 4
        read_val(t)
    tensors = {}
    for _ in range(n_tensors):
        name = read_str()
        n_dims = struct.unpack('<I', data[off:off+4])[0]; off += 4
        dims = list(struct.unpack('<'+'Q'*n_dims, data[off:off+8*n_dims])); off += 8*n_dims
        gtype = struct.unpack('<I', data[off:off+4])[0]; off += 4
        t_off = struct.unpack('<Q', data[off:off+8])[0]; off += 8
        tensors[name] = (gtype, dims, t_off)
    return data, tensors, off

def load_tensor(data, info, offset_base):
    gtype, dims, t_off = info
    off = offset_base + t_off
    nelem = int(np.prod(dims))
    if gtype == GGML_TYPE_F32:
        raw = data[off:off + 4*nelem]
        return np.frombuffer(raw, dtype='<f4').reshape(dims)
    elif gtype == GGML_TYPE_F16:
        raw = data[off:off + 2*nelem]
        return np.frombuffer(raw, dtype='<f2').astype(np.float32).reshape(dims)
    raise ValueError(f'type {gtype}')

def main():
    gguf_path = sys.argv[1]
    st_path = sys.argv[2]
    names = sys.argv[3].split(',') if len(sys.argv) > 3 else None

    data, tensors, off = read_gguf(gguf_path)
    data_start = (off + 31) & ~31
    sd = load_file(st_path)
    for name in sorted(tensors.keys()):
        if names and name not in names:
            continue
        got = load_tensor(data, tensors[name], data_start)
        want_shape = tuple(reversed(tensors[name][1]))
        got = got.reshape(want_shape)
        ref = np.asarray(sd[name], dtype=np.float32)
        if ref.shape != got.shape:
            print(f'{name}: SHAPE MISMATCH got={got.shape} ref={ref.shape}')
            continue
        d = np.abs(got - ref)
        print(f'{name}: max|d|={d.max():.3e} mean={d.mean():.3e} (shape {got.shape})')

if __name__ == '__main__':
    main()
