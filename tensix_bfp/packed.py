# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Xylolabs
"""Load, check and write packed checkpoints: BFP4 and BFP8 matrices stored as TT-NN tile bytes.

A packed checkpoint stores each quantized matrix as the bytes TT-NN keeps for a bfloat4_b or
bfloat8_b tensor in tile layout (4.5 or 8.5 bits a weight) instead of bf16 values on the grid.
`tensix_pack.json` lists the packed tensors; every other tensor is stored as it was.

Format "tensix-bfp-tiles", version 1:

- A packed tensor is U8 [n_tiles, tile_bytes], tile_bytes 576 (BFP4) or 1088 (BFP8), or
  [experts, n_tiles, tile_bytes] for a 3-D tensor. Its record in `tensix_pack.json` gives `format`,
  `stored_shape` (its shape in the bf16 checkpoint), `orientation` (`transposed`: each matrix's last
  two axes swapped, TT-NN's [in, out] for a PyTorch [out, in] weight; `as_stored`: no swap),
  `packed_shape` (one matrix after orientation, both axes multiples of 32), `tiles`, `bytes` and
  `bf16_sha256`. A tensor rounded when it was packed also has `"quantized_at_pack": "ttnn"` and
  `source_sha256`, the SHA-256 of the tensor it was rounded from.
- Tiles are 32x32, in row-major tile order, each cut into four 16x16 faces: top-left, top-right,
  bottom-left, bottom-right. A tile's bytes are 64 shared exponents (byte 16 f + r for row r of
  face f, biased by 127), then the faces' values row by row: BFP8 a byte each (bit 7 the sign,
  bits 0-6 the magnitude m), BFP4 a nibble each, the first of a pair in the low nibble (bit 3 the
  sign, bits 0-2 m).
- A value is sign * m / 2^(k-1) * 2^(e-127), k = 7 for BFP8 and 3 for BFP4, e its row's exponent.
  A row's exponent is the largest exponent of its 16 values. A zero is stored with sign 0, so a
  -0.0 of the bf16 checkpoint loads as +0.0.
- `bf16_sha256` is the SHA-256 of the decoded tensor's bf16 bytes in `stored_shape`, row-major,
  little-endian.

This file needs only torch and safetensors, and also works on its own, copied next to a checkpoint.

Usage:
  python packed.py verify PACKED_DIR             decode and check every packed tensor
  python packed.py unpack PACKED_DIR OUT_DIR     write the bf16 checkpoint, loadable by Transformers
  python packed.py pack BF16_DIR OUT_DIR [--at-pack REGEX=FMT ...]
                                                 pack a bf16-grid export (it needs quant4.json)

In Python, load_bf16(PACKED_DIR) returns {name: tensor} with the packed tensors decoded to bf16
and the others as stored; iter_bf16() yields them one at a time.
"""

import argparse
import ctypes
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

FORMAT = "tensix-bfp-tiles"
FORMAT_VERSION = 1
TILE = 32
FACE = 16
MANTISSA_BITS = {"bfp4": 3, "bfp8": 7}
TILE_BYTES = {"bfp4": 64 + 512, "bfp8": 64 + 1024}
ORIENTATIONS = ("transposed", "as_stored")
PACK_JSON = "tensix_pack.json"
INDEX = "model.safetensors.index.json"
SKIP = ("mtp.", "visual.", "vision", "audio", "depth_decoder")

__all__ = [
    "FORMAT",
    "FORMAT_VERSION",
    "TILE_BYTES",
    "decode",
    "encode",
    "iter_bf16",
    "load_bf16",
    "pack",
    "pack_tensor",
    "sha256",
    "unpack",
    "unpack_tensor",
    "verify",
]


# --- tiles ---------------------------------------------------------------------------------------


def _fmt_bits(fmt):
    if fmt not in MANTISSA_BITS:
        raise ValueError(f"format {fmt!r}: bfp4 or bfp8")
    return MANTISSA_BITS[fmt]


def _to_faces(m):
    """[R, C] -> [tiles, 4, 16, 16]: tiles in row-major order, faces top-left, top-right,
    bottom-left, bottom-right."""
    r, c = m.shape
    return (
        m.reshape(r // TILE, 2, FACE, c // TILE, 2, FACE)
        .permute(0, 3, 1, 4, 2, 5)
        .reshape(-1, 4, FACE, FACE)
    )


def _from_faces(f, r, c):
    """The inverse of _to_faces."""
    return f.reshape(r // TILE, c // TILE, 2, 2, FACE, FACE).permute(0, 2, 4, 1, 3, 5).reshape(r, c)


def _pow2(p):
    """2^p as float32 for integers p in [-126, 127], built from its bits."""
    return ((p + 127).to(torch.int32) << 23).view(torch.float32)


def _codes(faces, fmt):
    """(exponents [T, 4, 16], codes [T, 4, 16, 16]), uint8, of float32 faces, rounded as stock TT-NN
    rounds when it packs: a face row shares its largest exponent field, each value's mantissa
    (leading one included) is shifted to it and rounded to nearest even to k bits, then clamped at
    2^k - 1; a value whose exponent field is 0 becomes 0, and a zero gets sign 0."""
    k = _fmt_bits(fmt)
    b = faces.contiguous().view(torch.int32)
    exp = (b >> 23) & 0xFF
    shared = exp.amax(-1, keepdim=True)
    m = ((b & 0x7FFFFF) | 0x800000) >> (shared - exp).clamp(max=31)
    shift = 24 - k
    rest = m & ((1 << shift) - 1)
    m = m >> shift
    tie = 1 << (shift - 1)
    m = m + ((rest > tie) | ((rest == tie) & ((m & 1) == 1))).to(torch.int32)
    m = m.clamp(max=(1 << k) - 1)
    m = torch.where(exp == 0, torch.zeros_like(m), m)
    sign = ((b < 0) & (m != 0)).to(torch.int32)
    return shared.squeeze(-1).to(torch.uint8), ((sign << k) | m).to(torch.uint8)


def _values(exps, codes, fmt):
    """float32 values [T, 4, 16, 16] of exponents and codes; a zero magnitude decodes to +0.0."""
    k = _fmt_bits(fmt)
    mag = (codes & ((1 << k) - 1)).float()
    p = exps.to(torch.int32).unsqueeze(-1) - (127 + k - 1)
    hi = p.clamp(min=-126)
    v = mag * _pow2(hi) * _pow2(p - hi)  # both factors powers of two: exact
    return torch.where(((codes >> k) == 1) & (mag != 0), -v, v)


def _split(tiles, fmt):
    """(exponents [T, 4, 16], codes [T, 4, 16, 16]) of tile bytes [T, tile_bytes]."""
    t = tiles.shape[0]
    exps = tiles[:, :64].reshape(t, 4, FACE)
    body = tiles[:, 64:]
    if fmt == "bfp4":
        body = torch.stack((body & 0xF, body >> 4), dim=-1)
    return exps, body.reshape(t, 4, FACE, FACE)


def encode(matrix, fmt, exact=True):
    """Tile bytes [n_tiles, tile_bytes] (uint8) of a [R, C] matrix, R and C multiples of 32, in
    fmt ("bfp4" or "bfp8"). With exact (the default) every value must lie on its grid already
    (-0.0 is stored as +0.0), or a ValueError is raised. With exact=False the values are rounded as
    stock TT-NN rounds them when it packs a tensor (ttnn.as_tensor)."""
    k = _fmt_bits(fmt)
    if matrix.dim() != 2 or matrix.shape[0] % TILE or matrix.shape[1] % TILE:
        raise ValueError(f"shape {list(matrix.shape)}: a matrix with both axes multiples of {TILE}")
    x = _to_faces(matrix.detach().cpu().float())
    exps, codes = _codes(x, fmt)
    if exact:
        off = int((_values(exps, codes, fmt) != x).sum())  # -0.0 == +0.0
        if off:
            raise ValueError(f"{off} values are not on the {fmt} grid")
    t = exps.shape[0]
    codes = codes.reshape(t, 4 * FACE * FACE)
    if k == 3:
        codes = codes[:, 0::2] | (codes[:, 1::2] << 4)
    return torch.cat([exps.reshape(t, 64), codes], dim=1).contiguous()


def decode(tiles, fmt, shape):
    """The bf16 [R, C] matrix (shape) of tile bytes [n_tiles, tile_bytes] in fmt."""
    r, c = shape
    _fmt_bits(fmt)
    want = [(r // TILE) * (c // TILE), TILE_BYTES[fmt]]
    if r % TILE or c % TILE or tiles.dtype != torch.uint8 or list(tiles.shape) != want:
        raise ValueError(
            f"{tiles.dtype} {list(tiles.shape)}: tiles of {fmt} {[r, c]} are uint8 {want}"
        )
    exps, codes = _split(tiles, fmt)
    return _from_faces(_values(exps, codes, fmt), r, c).to(torch.bfloat16)


# --- tensors -------------------------------------------------------------------------------------


def sha256(t):
    """SHA-256 of a tensor's bytes, row-major, as stored on a little-endian machine."""
    if sys.byteorder != "little":
        raise RuntimeError("sha256: needs a little-endian machine")
    t = t.detach().cpu().contiguous()
    h = hashlib.sha256()
    n = t.numel() * t.element_size()
    if n:
        h.update(memoryview((ctypes.c_ubyte * n).from_address(t.data_ptr())))
    return h.hexdigest()


def _oriented(w, orientation):
    """The matrices [n, R, C] a 2-D or 3-D tensor packs as."""
    if w.dim() not in (2, 3):
        raise ValueError(f"shape {list(w.shape)}: a 2-D or 3-D tensor")
    if orientation not in ORIENTATIONS:
        raise ValueError(f"orientation {orientation!r}: one of {ORIENTATIONS}")
    mats = w if w.dim() == 3 else w.unsqueeze(0)
    return mats.transpose(1, 2) if orientation == "transposed" else mats


def pack_tensor(w, fmt, orientation="transposed", at_pack=False):
    """(uint8 tile bytes, tensix_pack.json record) of a 2-D or 3-D tensor (3-D: one matrix per
    entry of the first axis). The values must lie on fmt's grid, unless at_pack: then they are
    rounded as stock TT-NN rounds them, and the record says so."""
    mats = _oriented(w, orientation)
    r, c = mats.shape[1:]
    out = torch.stack([encode(m, fmt, exact=not at_pack) for m in mats])
    out = out if w.dim() == 3 else out[0]
    rec = {
        "format": fmt,
        "stored_shape": list(w.shape),
        "orientation": orientation,
        "packed_shape": [r, c],
        "tiles": [r // TILE, c // TILE],
        "bytes": out.numel(),
    }
    rec["bf16_sha256"] = sha256(_decode_tensor(out, rec))
    if at_pack:
        rec["quantized_at_pack"] = "ttnn"
        rec["source_sha256"] = sha256(w)
    return out, rec


def _record_problems(packed, rec):
    """What is wrong with a record, or with the tensor against it."""
    fmt, shape = rec.get("format"), rec.get("stored_shape")
    if fmt not in MANTISSA_BITS:
        return [f"format {fmt!r}: bfp4 or bfp8"]
    if rec.get("orientation") not in ORIENTATIONS:
        return [f"orientation {rec.get('orientation')!r}: one of {ORIENTATIONS}"]
    if not isinstance(shape, list) or len(shape) not in (2, 3):
        return [f"stored_shape {shape!r}: 2 or 3 axes"]
    r, c = shape[-2:] if rec["orientation"] == "as_stored" else shape[:-3:-1]
    if rec.get("packed_shape") != [r, c] or r % TILE or c % TILE:
        return [f"packed_shape {rec.get('packed_shape')!r}: {[r, c]}, both multiples of {TILE}"]
    if rec.get("tiles") != [r // TILE, c // TILE]:
        return [f"tiles {rec.get('tiles')!r}: {[r // TILE, c // TILE]}"]
    want = [r * c // (TILE * TILE), TILE_BYTES[fmt]]
    if len(shape) == 3:
        want = [shape[0], *want]
    if packed.dtype != torch.uint8 or list(packed.shape) != want:
        return [f"{packed.dtype} {list(packed.shape)}: uint8 {want}"]
    if rec.get("bytes") != packed.numel():
        return [f"bytes {rec.get('bytes')!r}: {packed.numel()}"]
    if not re.fullmatch(r"[0-9a-f]{64}", str(rec.get("bf16_sha256"))):
        return [f"bf16_sha256 {rec.get('bf16_sha256')!r}: 64 lowercase hex digits"]
    return []


def unpack_tensor(packed, rec):
    """The bf16 tensor, in rec's stored_shape, of a packed tensor and its tensix_pack.json record."""
    problems = _record_problems(packed, rec)
    if problems:
        raise ValueError("; ".join(problems))
    return _decode_tensor(packed, rec)


def _decode_tensor(packed, rec):
    """unpack_tensor without the record check."""
    mats = packed if packed.dim() == 3 else packed.unsqueeze(0)
    out = torch.stack([decode(t, rec["format"], rec["packed_shape"]) for t in mats])
    if rec["orientation"] == "transposed":
        out = out.transpose(1, 2)
    return (out if packed.dim() == 3 else out[0]).contiguous()


def check_tensor(packed, rec):
    """Problems of one packed tensor: its record and shape, zeros stored with sign 1, decoded values
    that miss bf16_sha256, and bytes that differ from the encoding of the decoded values (the
    exponent and the zeros stock TT-NN writes). An empty list when it is sound."""
    problems = _record_problems(packed, rec)
    if problems:
        return problems
    fmt = rec["format"]
    k = MANTISSA_BITS[fmt]
    _, codes = _split(packed.reshape(-1, TILE_BYTES[fmt]), fmt)
    signed = int((codes == (1 << k)).sum())
    if signed:
        problems.append(f"{signed} zeros stored with sign 1")
    t = unpack_tensor(packed, rec)
    if sha256(t) != rec.get("bf16_sha256"):
        problems.append("decoded values do not match bf16_sha256")
    mats = packed.reshape(-1, *packed.shape[-2:])
    for i, m in enumerate(_oriented(t, rec["orientation"])):
        if not torch.equal(encode(m, fmt), mats[i]):
            problems.append("bytes differ from the encoding of the decoded values")
            break
    return problems


# --- checkpoints ---------------------------------------------------------------------------------


def shards(model_dir):
    """{tensor name: shard file name} of a checkpoint directory: its index, or every .safetensors."""
    d = Path(model_dir)
    if (d / INDEX).exists():
        return json.loads((d / INDEX).read_text())["weight_map"]
    names = {}
    for f in sorted(d.glob("*.safetensors")):
        with safe_open(str(f), framework="pt") as g:
            for k in g.keys():  # noqa: SIM118 (a safe_open handle, not a dict)
                names[k] = f.name
    return names


def read_manifest(model_dir):
    """tensix_pack.json of a packed checkpoint, checked to be a format this file reads."""
    m = json.loads((Path(model_dir) / PACK_JSON).read_text())
    if m.get("format") != FORMAT:
        raise ValueError(f"{PACK_JSON}: format {m.get('format')!r}, not {FORMAT!r}")
    if m.get("format_version") != FORMAT_VERSION:
        raise ValueError(
            f"{PACK_JSON}: format_version {m.get('format_version')!r}; this loader reads "
            f"version {FORMAT_VERSION}"
        )
    if m.get("tile") != [TILE, TILE] or m.get("face") != [FACE, FACE]:
        raise ValueError(f"{PACK_JSON}: tile {m.get('tile')!r}, face {m.get('face')!r}")
    if not isinstance(m.get("tensors"), dict):
        raise TypeError(f"{PACK_JSON}: tensors is not an object")
    return m


def _dtypes(path):
    """{tensor name: safetensors dtype string} from a file's header."""
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(n))
    return {k: v["dtype"] for k, v in header.items() if k != "__metadata__"}


def _shard(d, shard, records, check):
    """(name, tensor) of one shard, packed tensors decoded."""
    with safe_open(str(d / shard), framework="pt") as g:
        for name in g.keys():  # noqa: SIM118 (a safe_open handle, not a dict)
            t = g.get_tensor(name)
            rec = records.get(name)
            if rec is not None:
                t = unpack_tensor(t, rec)
                if check and sha256(t) != rec["bf16_sha256"]:
                    raise ValueError(f"{name}: decoded values do not match bf16_sha256")
            yield name, t


def iter_bf16(model_dir, check=True):
    """(name, tensor) for every tensor of a packed checkpoint, one shard at a time: the packed
    tensors decoded to bf16 in their stored shape, the others as stored. With check (the default)
    a packed tensor whose decoded values miss its bf16_sha256 raises a ValueError."""
    d = Path(model_dir)
    records = read_manifest(d)["tensors"]
    weight_map = shards(d)
    missing = sorted(set(records) - set(weight_map))
    if missing:
        raise ValueError(f"{PACK_JSON} lists tensors no shard holds: {missing[:5]}")
    for shard in sorted(set(weight_map.values())):
        yield from _shard(d, shard, records, check)


def load_bf16(model_dir, check=True):
    """{name: tensor} of a packed checkpoint (see iter_bf16), for load_state_dict."""
    return dict(iter_bf16(model_dir, check))


def verify(model_dir):
    """(problems, number of packed tensors checked) of a packed checkpoint: check_tensor for each
    tensor tensix_pack.json lists, listed tensors no shard holds, and U8 tensors it does not list."""
    d = Path(model_dir)
    try:
        records = read_manifest(d)["tensors"]
    except (OSError, TypeError, ValueError) as e:
        return [str(e)], 0
    weight_map = shards(d)
    problems = [
        f"{n}: listed in {PACK_JSON}, in no shard" for n in sorted(set(records) - set(weight_map))
    ]
    checked = 0
    for shard in sorted(set(weight_map.values())):
        dtypes = _dtypes(d / shard)
        with safe_open(str(d / shard), framework="pt") as g:
            for name in g.keys():  # noqa: SIM118 (a safe_open handle, not a dict)
                rec = records.get(name)
                if rec is None:
                    if dtypes[name] == "U8":
                        problems.append(f"{name}: a U8 tensor {PACK_JSON} does not list")
                    continue
                checked += 1
                problems += [f"{name}: {p}" for p in check_tensor(g.get_tensor(name), rec)]
    return problems, checked


def _copy_files(src, dst, skip):
    for f in sorted(src.iterdir()):
        if (
            f.is_file()
            and not f.name.startswith(".")
            and f.suffix != ".safetensors"
            and f.name not in skip
        ):
            shutil.copy2(f, dst / f.name)


def _out_dir(out_dir):
    o = Path(out_dir)
    if o.exists() and any(o.iterdir()):
        raise SystemExit(f"{o}: the output directory is not empty")
    o.mkdir(parents=True, exist_ok=True)
    return o


def unpack(model_dir, out_dir, check=True):
    """Write the bf16 checkpoint of a packed one to out_dir: the same shards and tensor names with
    the packed tensors decoded, and the other files copied (tensix_pack.json left out)."""
    d = Path(model_dir)
    records = read_manifest(d)["tensors"]
    weight_map = shards(d)
    o = _out_dir(out_dir)
    for shard in sorted(set(weight_map.values())):
        with safe_open(str(d / shard), framework="pt") as g:
            meta = dict(g.metadata() or {})
        meta.setdefault("format", "pt")
        save_file(dict(_shard(d, shard, records, check)), str(o / shard), metadata=meta)
    _copy_files(d, o, {PACK_JSON})
    return o


def quantized_tensors(manifest, weight_map):
    """(tensor name, format) of every tensor quant4.json lists as bfp4 or bfp8: the layers'
    projections, numbered experts and the LM head."""
    for layer, projections in manifest.get("layers", {}).items():
        for name, record in projections.items():
            fmt = record.get("format") if isinstance(record, dict) else None
            if fmt not in MANTISSA_BITS:
                continue
            suffixes = (f"layers.{layer}.{name}.weight", f"layers.{layer}.{name}")
            keys = [k for k in weight_map if k.endswith(suffixes) and not any(s in k for s in SKIP)]
            if not keys and ".experts." in name:
                module, projection = name.rsplit(".experts.", 1)
                parts = ("gate_proj", "up_proj") if projection == "gate_up_proj" else ()
                pattern = re.compile(
                    rf"(?:^|\.)layers\.{re.escape(str(layer))}\.{re.escape(module)}"
                    rf"\.experts\.\d+\.({'|'.join(map(re.escape, (projection, *parts)))})(?:\.weight)?$"
                )
                keys = [
                    k for k in weight_map if pattern.search(k) and not any(s in k for s in SKIP)
                ]
            if not keys:
                raise ValueError(f"projection {layer}/{name}: no tensor in the weight map")
            for k in keys:
                yield k, fmt
    fmt = manifest.get("lm_head")
    if fmt in MANTISSA_BITS:
        name = manifest.get("lm_head_tensor", "lm_head.weight")
        if name not in weight_map:
            raise ValueError(f"quantized LM head {name!r} is missing from the weight map")
        yield name, fmt


def pack(src_dir, out_dir, at_pack=()):
    """Write the packed checkpoint of a bf16-grid export (with its quant4.json) to out_dir. Each
    tensor quant4.json lists as bfp4 or bfp8 is packed transposed (TT-NN's [in, out]); each other
    tensor whose name matches a regular expression of at_pack, a list of (regex, format), is rounded
    as stock TT-NN rounds it and packed the same way. A tensor whose packed matrices are not
    multiples of 32 in both axes stays as stored. Shards become model-0000k-of-0000n.safetensors
    (model.safetensors when there is one). Returns (records, names left as stored)."""
    s = Path(src_dir)
    manifest = json.loads((s / "quant4.json").read_text())
    weight_map = shards(s)
    fmts = dict(quantized_tensors(manifest, weight_map))
    rules = []
    for rx, fmt in at_pack:
        _fmt_bits(fmt)
        rules.append((re.compile(rx), fmt))
    o = _out_dir(out_dir)
    files = sorted(set(weight_map.values()))
    n = len(files)
    renamed = {
        f: "model.safetensors" if n == 1 else f"model-{i:05d}-of-{n:05d}.safetensors"
        for i, f in enumerate(files, 1)
    }
    records, new_map, kept = {}, {}, []
    for f in files:
        out = {}
        with safe_open(str(s / f), framework="pt") as g:
            for name in g.keys():  # noqa: SIM118 (a safe_open handle, not a dict)
                t = g.get_tensor(name)
                fmt, at = fmts.get(name), False
                if fmt is None:
                    fmt = next((rf for rx, rf in rules if rx.search(name)), None)
                    at = fmt is not None
                if fmt is not None:
                    aligned = (
                        t.dim() in (2, 3) and t.shape[-1] % TILE == 0 and t.shape[-2] % TILE == 0
                    )
                    if aligned and t.is_floating_point():
                        t, records[name] = pack_tensor(t, fmt, "transposed", at_pack=at)
                    else:
                        kept.append(name)
                out[name] = t
                new_map[name] = renamed[f]
        save_file(out, str(o / renamed[f]), metadata={"format": "pt"})
    if n > 1:
        meta = {}
        if (s / INDEX).exists():
            meta = json.loads((s / INDEX).read_text()).get("metadata", {})
        (o / INDEX).write_text(json.dumps({"metadata": meta, "weight_map": new_map}, indent=1))
    manifest_out = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "tile": [TILE, TILE],
        "face": [FACE, FACE],
        "tensors": records,
    }
    (o / PACK_JSON).write_text(json.dumps(manifest_out, indent=1))
    _copy_files(s, o, {INDEX, PACK_JSON})
    return records, kept


# --- command line --------------------------------------------------------------------------------


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    v = sub.add_parser("verify", help="decode and check every packed tensor")
    v.add_argument("model_dir")
    u = sub.add_parser("unpack", help="write the bf16 checkpoint of a packed one")
    u.add_argument("model_dir")
    u.add_argument("out_dir")
    p = sub.add_parser("pack", help="pack a bf16-grid export")
    p.add_argument("src_dir")
    p.add_argument("out_dir")
    p.add_argument(
        "--at-pack",
        action="append",
        default=[],
        metavar="REGEX=FMT",
        help="also round and pack the tensors whose names match REGEX (repeatable)",
    )
    a = ap.parse_args(argv)
    if a.command == "verify":
        problems, checked = verify(a.model_dir)
        for problem in problems:
            print(f"BAD: {problem}")
        print(f"{checked} packed tensors checked, {len(problems)} problems")
        return 0 if checked and not problems else 1
    if a.command == "unpack":
        print(unpack(a.model_dir, a.out_dir))
        return 0
    rules = [tuple(x.rsplit("=", 1)) for x in a.at_pack]
    if any(len(r) != 2 for r in rules):
        ap.error("--at-pack takes REGEX=FMT")
    records, kept = pack(a.src_dir, a.out_dir, rules)
    for name in kept:
        print(f"kept as stored (not tile-aligned): {name}")
    print(f"{len(records)} tensors packed, {len(kept)} kept as stored")
    return 0


if __name__ == "__main__":
    sys.exit(main())
