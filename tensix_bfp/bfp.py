# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Xylolabs
"""Block floating point (BFP) rounding as TT-NN stores bfloat8_b and bfloat4_b tensors.

A block is 16 consecutive values along the last dimension. The block shares one 8-bit exponent,
the largest exponent among its values, and each value keeps a sign and a mantissa whose leading
one is explicit: 7 bits for bfloat8_b, 3 bits for bfloat4_b. A value is m * 2^(E - 127 - (bits - 1))
with m in 0 .. 2^bits - 1 and E the shared exponent.

TT-NN multiplies x @ W with the weight laid out [in, out], the transpose of a PyTorch Linear
weight, so a block is 16 output channels of one input channel. The `*_weight` functions take a
PyTorch [out, in] weight and round it in that layout.

Format names used across the package: "bf16", "bfp8" (bfloat8_b), "bfp4" (bfloat4_b).
"""

import torch

# exponent offsets tried per block by the exponent search of BFP4 GPTQ (E - 1 first, then E)
EXP_OFFSETS = (-1, 0)


def bfp(x, mant_bits, block=16):
    """x rounded to the block floating point grid exactly as tt-metal's host packer does it
    (round to nearest even, the mantissa clamped at its maximum, values whose exponent field is
    0 flushed to 0), returned as float32. Blocks are `block` consecutive values along the last
    dimension. mant_bits 7: bfloat8_b, 3: bfloat4_b."""
    shape = x.shape
    assert shape[-1] % block == 0, f"bfp: last dimension {shape[-1]} is not a multiple of {block}"
    b = x.float().contiguous().reshape(-1, block).view(torch.int32)
    exp = (b >> 23) & 0xFF
    mant = (b & 0x7FFFFF) | 0x800000
    shared = exp.amax(1, keepdim=True)
    m = mant >> (shared - exp).clamp(max=31)
    shift = 24 - mant_bits
    rnd = m & ((1 << shift) - 1)
    m = m >> shift
    tie = 1 << (shift - 1)
    m = m + ((rnd > tie) | ((rnd == tie) & ((m & 1) == 1))).int()
    m = m.clamp(max=(1 << mant_bits) - 1)
    m = torch.where(exp == 0, torch.zeros_like(m), m)
    v = torch.ldexp(m.float(), (shared - 127 - (mant_bits - 1)).float())
    return torch.where(b < 0, -v, v).reshape(shape)


def bfp_forced(x, e, mant_bits):
    """x rounded to the BFP grid of the given shared exponents e (integers broadcast to x):
    value = m * 2^(e - 127 - (mant_bits - 1)), m in 0 .. 2^mant_bits - 1, round half to even."""
    step = torch.ldexp(torch.ones_like(x, dtype=torch.float32), (e - 127 - (mant_bits - 1)).float())
    m = (x.float().abs() / step).round().clamp(max=(1 << mant_bits) - 1)
    return torch.sign(x) * m * step


def bfp_weight(w, mant_bits):
    """bfp() of a PyTorch [out, in] weight in TT-NN's [in, out] layout (blocks of 16 outputs)."""
    return bfp(w.float().T.contiguous(), mant_bits).T.contiguous()


def bfp_search(x, mant_bits, cands=(0, -1), block=16):
    """x rounded per block to the BFP grid with the shared exponent among E + c (c in cands, E
    the block's largest exponent) that gives the smallest squared error; on a tie the earlier
    candidate wins. c = 0 is bfp() itself; c = -1 clips the block's largest values at the top
    code and puts the rest on a grid twice as fine. Every code stays below 2^mant_bits, so the
    packer, which takes the block's largest exponent, stores the result unchanged."""
    shape = x.shape
    b = x.float().contiguous().reshape(-1, block)
    exp = (b.view(torch.int32) >> 23) & 0xFF
    shared = exp.amax(1, keepdim=True)
    best = best_err = None
    for c in cands:
        if c == 0:
            q = bfp(b, mant_bits, block)
        else:
            q = bfp_forced(b, (shared + c).expand_as(b), mant_bits)
            q = torch.where(exp == 0, torch.zeros_like(q), q)
        err = (q - b).pow(2).sum(1, keepdim=True)
        if best is None:
            best, best_err = q, err
        else:
            better = err < best_err
            best = torch.where(better, q, best)
            best_err = torch.where(better, err, best_err)
    return best.reshape(shape)


def bfp_search_weight(w, mant_bits, cands=(0, -1)):
    """bfp_search() of a PyTorch [out, in] weight in TT-NN's [in, out] layout."""
    return bfp_search(w.float().T.contiguous(), mant_bits, cands).T.contiguous()


def bf16_round(s):
    """s rounded to bfloat16 and back to float32."""
    return s.to(torch.bfloat16).float()


def bf16_ceil(s):
    """The smallest bfloat16 value >= s, as float32 (s > 0, float32)."""
    b = s.to(torch.bfloat16).float()
    up = (b.view(torch.int32) + 0x10000).view(torch.float32)
    return torch.where(b >= s, b, up)


def parse_fmt(fmt):
    """(kind, mantissa bits, block) of a format name: ("bfp", 7 or 3, 16) or ("bf16", 16, 0)."""
    if fmt == "bf16":
        return "bf16", 16, 0
    if fmt in ("bfp8", "bfp4"):
        return "bfp", {"bfp8": 7, "bfp4": 3}[fmt], 16
    raise ValueError(f"unknown format {fmt!r} (bf16, bfp8 or bfp4)")


def quantize(w, fmt):
    """Round-to-nearest of a PyTorch [out, in] weight to fmt, returned as float32."""
    kind, bits, _ = parse_fmt(fmt)
    if kind == "bf16":
        return w.float().bfloat16().float()
    return bfp_weight(w, bits)


def on_grid(w, fmt):
    """Whether a PyTorch [out, in] weight already lies on fmt's grid (re-rounding is a no-op)."""
    kind, bits, _ = parse_fmt(fmt)
    if kind == "bf16":
        return torch.equal(w.float().bfloat16().float(), w.float())
    return torch.equal(bfp_weight(w.float(), bits), w.float())


def rel(ref, out):
    """Relative Frobenius error of out against ref."""
    return ((ref.float() - out.float()).norm() / ref.float().norm()).item()
