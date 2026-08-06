#!/usr/bin/env python3
"""
Quantize GGUF model weights from f16/f32 to Q8_0.

Reads an existing GGUF file, selectively quantizes large weight tensors to
Q8_0 (~8.5 bits per weight, ~47% size reduction), and writes a new GGUF file.
Sensitive small tensors (biases, norms, gammas, modulation, token embeddings)
stay at f32 to preserve inference precision.

The Q8_0 implementation exactly matches ggml's quantize_row_q8_0_ref():
  - Block size: 32 weights per block
  - Each block: 1 x fp16 scale + 32 x int8 quanta = 34 bytes
  - Quantization: d = amax / 127, q[i] = round(x[i] / d)
  - Dequantization: y[i] = q[i] * float(d)

CUDA compatibility note:
  The ggml CUDA backend now supports ggml_cpy for Q8_0→Q8_0 (added in this
  project). However, precision-sensitive decoders (shape_dec, tex_dec,
  shape_enc) must stay at F16 because sparse subdivision and UV decoding
  are not robust to Q8 weight rounding.

Usage:
    python quantize_to_q8.py models/ss_flow_f16.gguf models/ss_flow_q8.gguf
    python quantize_to_q8.py models/ss_flow_f16.gguf --list   # inspect tensors

    # Batch quantize all models into models/ (skip precision-sensitive decoders):
    python quantize_to_q8.py --batch models/ models/ --skip shape_dec --skip tex_dec --skip shape_enc
"""

import argparse
import os
import struct
import sys

import numpy as np

# ── GGUF / GGML constants ────────────────────────────────────────────────────
GGUF_MAGIC = b"GGUF"
GGUF_ALIGNMENT = 32

# GGML tensor types
GGML_TYPE_F32  = 0
GGML_TYPE_F16  = 1
GGML_TYPE_Q8_0 = 8

# GGUF metadata value types (enum gguf_type from gguf.h)
GGUF_VT_UINT8   = 0
GGUF_VT_INT8    = 1
GGUF_VT_UINT16  = 2
GGUF_VT_INT16   = 3
GGUF_VT_UINT32  = 4
GGUF_VT_INT32   = 5
GGUF_VT_FLOAT32 = 6
GGUF_VT_BOOL    = 7
GGUF_VT_STRING  = 8
GGUF_VT_ARRAY   = 9
GGUF_VT_UINT64  = 10
GGUF_VT_INT64   = 11
GGUF_VT_FLOAT64 = 12

QK8_0 = 32  # Q8_0 block size (must match ggml-common.h)

# Tensor name patterns that must stay f32 for precision.
# Mirrors the choose_type() policy in the conversion scripts: norms, gammas,
# modulation parameters, and biases are precision-sensitive.
# Token embeddings (cls_token, register_tokens, mask_token) must stay f32
# because they are concatenated with computed F32 tensors in the graph.
_KEEP_F32_PATTERNS = (
    "gamma", "beta", "norm", "bias", "modulation",
    "embed_tokens", "token_type", "position_embed",
    "cls_token", "register_token", "mask_token",
)


# ── GGUF reader ──────────────────────────────────────────────────────────────

def _read_str(f):
    n = struct.unpack("<Q", f.read(8))[0]
    return f.read(n).decode("utf-8")


def _read_value(f, vtype):
    """Read a GGUF metadata value. Returns (type_tag, raw_bytes, parsed_value)."""
    if vtype == GGUF_VT_UINT8:
        v = struct.unpack("<B", f.read(1))[0]
        return ("uint8", struct.pack("<B", v), v)
    if vtype == GGUF_VT_INT8:
        v = struct.unpack("<b", f.read(1))[0]
        return ("int8", struct.pack("<b", v), v)
    if vtype == GGUF_VT_UINT16:
        v = struct.unpack("<H", f.read(2))[0]
        return ("uint16", struct.pack("<H", v), v)
    if vtype == GGUF_VT_INT16:
        v = struct.unpack("<h", f.read(2))[0]
        return ("int16", struct.pack("<h", v), v)
    if vtype == GGUF_VT_UINT32:
        v = struct.unpack("<I", f.read(4))[0]
        return ("uint32", struct.pack("<I", v), v)
    if vtype == GGUF_VT_INT32:
        v = struct.unpack("<i", f.read(4))[0]
        return ("int32", struct.pack("<i", v), v)
    if vtype == GGUF_VT_FLOAT32:
        v = struct.unpack("<f", f.read(4))[0]
        return ("float32", struct.pack("<f", v), v)
    if vtype == GGUF_VT_BOOL:
        v = struct.unpack("<b", f.read(1))[0]  # bool stored as int8_t
        return ("bool", struct.pack("<b", v), bool(v))
    if vtype == GGUF_VT_STRING:
        s = _read_str(f)
        return ("string", None, s)  # string payload written separately
    if vtype == GGUF_VT_ARRAY:
        atype = struct.unpack("<I", f.read(4))[0]
        n = struct.unpack("<Q", f.read(8))[0]
        elems = []
        for _ in range(n):
            elems.append(_read_value(f, atype))
        return ("array", None, (atype, elems))
    if vtype == GGUF_VT_UINT64:
        v = struct.unpack("<Q", f.read(8))[0]
        return ("uint64", struct.pack("<Q", v), v)
    if vtype == GGUF_VT_INT64:
        v = struct.unpack("<q", f.read(8))[0]
        return ("int64", struct.pack("<q", v), v)
    if vtype == GGUF_VT_FLOAT64:
        v = struct.unpack("<d", f.read(8))[0]
        return ("float64", struct.pack("<d", v), v)
    raise ValueError(f"unknown metadata value type {vtype}")


def read_gguf(path):
    """Parse a GGUF file. Returns (metadata, tensor_infos, data_offset, file_data).

    metadata: list of (key, vtype_int, raw_payload_bytes_or_None, parsed_value)
    tensor_infos: list of (name, ggml_type, dims_ggml_order, abs_pos, nbytes)
    """
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != GGUF_MAGIC:
            raise ValueError(f"not a GGUF file: {path}")
        version = struct.unpack("<I", f.read(4))[0]
        if version != 3:
            raise ValueError(f"unsupported GGUF version {version}")
        n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]

        metadata = []
        for _ in range(n_kv):
            key = _read_str(f)
            vtype = struct.unpack("<I", f.read(4))[0]
            tag, raw, val = _read_value(f, vtype)
            metadata.append((key, vtype, raw, val))

        tensor_infos = []
        for _ in range(n_tensors):
            name = _read_str(f)
            ndim = struct.unpack("<I", f.read(4))[0]
            dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(ndim)]
            gtype = struct.unpack("<I", f.read(4))[0]
            offset = struct.unpack("<Q", f.read(8))[0]
            tensor_infos.append((name, gtype, dims, offset))

        data_offset = f.tell()
        data_offset = _align(data_offset)

        f.seek(0)
        file_data = f.read()

    # Resolve absolute positions
    resolved = []
    for name, gtype, dims, off in tensor_infos:
        abs_pos = data_offset + off
        ne = 1
        for d in dims:
            ne *= d
        if gtype in (GGML_TYPE_F32,):
            nbytes = ne * 4
        elif gtype in (GGML_TYPE_F16,):
            nbytes = ne * 2
        elif gtype == GGML_TYPE_Q8_0:
            n_blocks = (ne + QK8_0 - 1) // QK8_0
            nbytes = n_blocks * 34
        else:
            raise ValueError(f"unsupported ggml type {gtype} for '{name}'")
        resolved.append((name, gtype, dims, abs_pos, nbytes))

    return metadata, resolved, data_offset, file_data


def extract_tensor_data(file_data, info):
    """Read raw bytes for a single tensor from the file data."""
    name, gtype, dims, abs_pos, nbytes = info
    raw = file_data[abs_pos:abs_pos + nbytes]
    if len(raw) < nbytes:
        raise ValueError(f"truncated data for tensor '{name}'")
    return raw


def dequant_f16(raw, ne):
    """f16 raw bytes -> float32 numpy array."""
    arr = np.frombuffer(raw[:ne * 2], dtype="<f2").astype(np.float32)
    return arr


def dequant_f32(raw, ne):
    """f32 raw bytes -> float32 numpy array."""
    return np.frombuffer(raw[:ne * 4], dtype="<f4").copy()


# ── Q8_0 quantization (matches ggml quantize_row_q8_0_ref exactly) ──────────

def quantize_q8_0(f32_array):
    """Quantize a flat float32 array to Q8_0 blocks.

    Returns raw bytes: for each block of 32 weights:
      - fp16 scale d (2 bytes)
      - 32 x int8 quanta (32 bytes)
      Total: 34 bytes per block.
    """
    n = len(f32_array)
    # Pad to multiple of QK8_0
    pad = (QK8_0 - n % QK8_0) % QK8_0
    if pad:
        f32_array = np.concatenate([f32_array, np.zeros(pad, dtype=np.float32)])
    n_padded = len(f32_array)
    n_blocks = n_padded // QK8_0

    blocks = f32_array.reshape(n_blocks, QK8_0)

    # amax per block
    amax = np.max(np.abs(blocks), axis=1)

    # d = amax / 127, matching ggml: const float d = amax / ((1 << 7) - 1);
    d = amax / 127.0

    # id = 1/d, with zero-division guard: const float id = d ? 1.0f/d : 0.0f;
    id = np.where(d > 0, 1.0 / d, 0.0)

    # q[i] = roundf(x[i] * id)
    scaled = blocks * id[:, np.newaxis]
    # roundf = round-half-away-from-zero (C99), not banker's rounding
    quants = np.sign(scaled) * np.floor(np.abs(scaled) + 0.5)
    quants = np.clip(quants, -128, 127).astype(np.int8)

    # Pack: fp16 scale + int8 quanta per block
    d_fp16 = d.astype("<f2")
    out = bytearray()
    for i in range(n_blocks):
        out += d_fp16[i].tobytes()
        out += quants[i].tobytes()

    return bytes(out), n


# ── GGUF writer ──────────────────────────────────────────────────────────────

def _write_str_to(buf, s):
    sb = s.encode("utf-8")
    buf += struct.pack("<Q", len(sb)) + sb


def _write_value_to(buf, vtype, raw, val):
    """Write a metadata value payload (type tag already written by caller)."""
    if vtype == GGUF_VT_STRING:
        _write_str_to(buf, val)
    elif vtype == GGUF_VT_ARRAY:
        atype, elems = val
        buf += struct.pack("<I", atype)
        buf += struct.pack("<Q", len(elems))
        for elem_entry in elems:
            if len(elem_entry) == 3:
                _, elem_raw, elem_val = elem_entry
            else:
                elem_raw, elem_val = None, elem_entry
            _write_value_to(buf, atype, elem_raw, elem_val)
    elif raw is not None:
        buf += raw
    elif val is not None:
        # Scalar value without raw bytes (e.g. array element) - encode it
        fmt = {
            GGUF_VT_UINT8: "<B", GGUF_VT_INT8: "<b",
            GGUF_VT_UINT16: "<H", GGUF_VT_INT16: "<h",
            GGUF_VT_UINT32: "<I", GGUF_VT_INT32: "<i",
            GGUF_VT_FLOAT32: "<f", GGUF_VT_BOOL: "<b",
            GGUF_VT_UINT64: "<Q", GGUF_VT_INT64: "<q",
            GGUF_VT_FLOAT64: "<d",
        }
        if vtype in fmt:
            buf += struct.pack(fmt[vtype], val)
        else:
            raise ValueError(f"cannot encode value type {vtype}")
    else:
        raise ValueError(f"cannot write value for type {vtype}")


def _write_kv_to(buf, key, vtype, raw, val):
    _write_str_to(buf, key)
    buf += struct.pack("<I", vtype)
    _write_value_to(buf, vtype, raw, val)


def _align(n, a=GGUF_ALIGNMENT):
    return (n + a - 1) // a * a


def ggml_type_size(gtype):
    """Return (block_size, bytes_per_block) for a ggml type."""
    table = {
        GGML_TYPE_F32:  (1, 4),
        GGML_TYPE_F16:  (1, 2),
        GGML_TYPE_Q8_0: (QK8_0, 34),
    }
    if gtype not in table:
        raise ValueError(f"unsupported output type {gtype}")
    return table[gtype]


def tensor_data_size(gtype, ne):
    blksz, blkbytes = ggml_type_size(gtype)
    n_blocks = (ne + blksz - 1) // blksz
    return n_blocks * blkbytes


# ── Quantization policy ──────────────────────────────────────────────────────

def should_quantize(name, gtype, dims):
    """Decide whether a tensor should be quantized to Q8_0.

    Policy: quantize large (>=2D) f16/f32 weight matrices where the innermost
    dimension (ne[0], ggml order) is a multiple of QK8_0 (32).  This excludes
    3D conv kernels (ne[0]=3) which ggml cannot store as Q8_0, as well as
    precision-sensitive small tensors (biases, norms, gammas, modulation).
    """
    ne = 1
    for d in dims:
        ne *= d
    if gtype not in (GGML_TYPE_F16, GGML_TYPE_F32):
        return False
    if len(dims) < 2:
        return False
    # ggml requires ne[0] % blck_size == 0 for quantized types
    if dims[0] % QK8_0 != 0:
        return False
    for pat in _KEEP_F32_PATTERNS:
        if pat in name:
            return False
    return True


# ── Main ─────────────────────────────────────────────────────────────────────

def quantize_file(in_path, out_path, quiet=False):
    """Quantize a single GGUF file from f16/f32 to Q8_0."""
    metadata, tensor_infos, data_offset, file_data = read_gguf(in_path)

    # Update file_type metadata (general.file_type is uint32 = type 4)
    new_metadata = []
    for key, vtype, raw, val in metadata:
        if key == "general.file_type":
            new_metadata.append((key, GGUF_VT_UINT32, struct.pack("<I", 8), 8))
        else:
            new_metadata.append((key, vtype, raw, val))
    metadata = new_metadata

    # Process tensors
    out_tensors = []  # (name, new_gtype, dims, raw_bytes)
    stats = {"quantized": 0, "kept": 0, "saved": 0}

    for info in tensor_infos:
        name, gtype, dims, abs_pos, nbytes = info
        ne = 1
        for d in dims:
            ne *= d

        if should_quantize(name, gtype, dims):
            if gtype == GGML_TYPE_F16:
                f32 = dequant_f16(file_data[abs_pos:abs_pos + nbytes], ne)
            else:
                f32 = dequant_f32(file_data[abs_pos:abs_pos + nbytes], ne)
            q8_bytes, orig_ne = quantize_q8_0(f32)
            out_tensors.append((name, GGML_TYPE_Q8_0, dims, q8_bytes))
            stats["quantized"] += 1
            stats["saved"] += nbytes - len(q8_bytes)
            if not quiet:
                print(f"  Q8  {name}  {dims}  {nbytes:>12,} -> {len(q8_bytes):>12,}")
        else:
            raw = file_data[abs_pos:abs_pos + nbytes]
            out_tensors.append((name, gtype, dims, raw))
            stats["kept"] += 1
            if not quiet:
                tname = {GGML_TYPE_F32: "f32", GGML_TYPE_F16: "f16"}.get(gtype, f"?{gtype}")
                print(f"  {tname:<3} {name}  {dims}  {nbytes:>12,}")

    # Write output GGUF
    header = bytearray()
    header += GGUF_MAGIC
    header += struct.pack("<I", 3)
    header += struct.pack("<Q", len(out_tensors))
    header += struct.pack("<Q", len(metadata))
    for key, vtype, raw, val in metadata:
        _write_kv_to(header, key, vtype, raw, val)

    infos = bytearray()
    offset = 0
    offsets = []
    for name, gtype, dims, raw in out_tensors:
        offsets.append(offset)
        offset = _align(offset + len(raw))
    for (name, gtype, dims, raw), off in zip(out_tensors, offsets):
        _write_str_to(infos, name)
        infos += struct.pack("<I", len(dims))
        for d in dims:
            infos += struct.pack("<Q", int(d))
        infos += struct.pack("<I", gtype)
        infos += struct.pack("<Q", off)

    pre_data = len(header) + len(infos)
    pad0 = _align(pre_data) - pre_data

    with open(out_path, "wb") as fout:
        fout.write(header)
        fout.write(infos)
        fout.write(b"\x00" * pad0)
        for (name, gtype, dims, raw), off in zip(out_tensors, offsets):
            fout.write(raw)
            pad = _align(len(raw)) - len(raw)
            if pad:
                fout.write(b"\x00" * pad)

    in_size = os.path.getsize(in_path)
    out_size = os.path.getsize(out_path)
    if not quiet:
        print(f"\nwrote {out_path}")
        print(f"  tensors: {stats['quantized']} quantized (Q8_0), "
              f"{stats['kept']} kept (f32/f16)")
        print(f"  file size: {in_size:>12,} -> {out_size:>12,}  "
              f"({out_size / in_size:.1%})")
    return stats


def main():
    ap = argparse.ArgumentParser(
        description="Quantize GGUF model weights to Q8_0 (~47% size reduction)")
    ap.add_argument("input", help="input GGUF file (f16 or f32)")
    ap.add_argument("output", nargs="?", help="output GGUF file (q8)")
    ap.add_argument("--list", action="store_true",
                    help="list tensors and exit (no quantization)")
    ap.add_argument("--batch", action="store_true",
                    help="batch mode: input=dir_in, output=dir_out")
    ap.add_argument("--skip", nargs="*", default=[],
                    help="skip models matching these substrings (e.g. --skip shape_dec)")
    ap.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args()

    if args.list:
        metadata, tensor_infos, data_offset, file_data = read_gguf(args.input)
        type_names = {0: "f32", 1: "f16", 8: "q8_0", 30: "bf16"}
        counts = {}
        total_bytes = 0
        for name, gtype, dims, abs_pos, nbytes in tensor_infos:
            tname = type_names.get(gtype, f"?{gtype}")
            q = "->q8" if should_quantize(name, gtype, dims) else "    "
            print(f"  {q}  {tname:<5} {nbytes:>12,}  {name}  {dims}")
            counts[gtype] = counts.get(gtype, 0) + 1
            total_bytes += nbytes
        print(f"\n  total: {len(tensor_infos)} tensors, "
              f"{total_bytes:,} bytes raw data")
        for t, c in sorted(counts.items()):
            print(f"    {type_names.get(t, f'?{t}')}: {c}")
        return

    if args.batch:
        # Batch mode: quantize all .gguf files in input dir
        in_dir = args.input
        out_dir = args.output or (in_dir.rstrip("/") + "_q8")
        os.makedirs(out_dir, exist_ok=True)
        files = sorted(f for f in os.listdir(in_dir) if f.endswith(".gguf"))
        if not files:
            print(f"no .gguf files found in {in_dir}")
            sys.exit(1)
        total_in = total_out = 0
        for fname in files:
            if any(s in fname for s in args.skip):
                print(f"\nSkipping {fname} (matched --skip filter)")
                continue
            in_path = os.path.join(in_dir, fname)
            out_name = fname.replace("_f16", "_q8").replace("_f32", "_q8")
            if out_name == fname:
                out_name = fname.replace(".gguf", "_q8.gguf")
            out_path = os.path.join(out_dir, out_name)
            print(f"\n{'='*60}")
            print(f"Quantizing {fname}...")
            stats = quantize_file(in_path, out_path, quiet=args.quiet)
            total_in += os.path.getsize(in_path)
            total_out += os.path.getsize(out_path)
        print(f"\n{'='*60}")
        print(f"Batch complete: {len(files)} files")
        print(f"  total: {total_in:>12,} → {total_out:>12,}  "
              f"({total_out / total_in:.1%})")
        return

    if not args.output:
        # Default output name
        args.output = args.input.replace(".gguf", "_q8.gguf")
        if args.output == args.input:
            args.output = args.input + ".q8.gguf"

    quantize_file(args.input, args.output, quiet=args.quiet)


if __name__ == "__main__":
    main()
