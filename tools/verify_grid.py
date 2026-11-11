#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Xylolabs
"""Check that every quantized matrix of this checkpoint lies exactly on its block floating-point grid.

The checkpoint stores bf16 values. `quant4.json` names each linear projection's format per layer:
bfp4 = TT-NN's bfloat4_b (BFP4), bfp8 = bfloat8_b (BFP8). TT-NN multiplies x @ W with W laid out
[in, out] (the transpose of the PyTorch weight), and in tile layout each block of 16 consecutive
values of a row (16 output channels of one input channel) shares one 8-bit exponent: the largest
exponent of the 16. Each value keeps a sign and a 3-bit (BFP4) or 7-bit (BFP8) mantissa. A value is
on the grid when re-quantizing it with its block's exponent gives it back unchanged; then TT-NN's
packer stores it as it is.

Needs only torch and safetensors (no Tenstorrent software).

Usage: python verify_grid.py [MODEL_DIR] [--layers 0,1,...]
Exit status 0 when every listed matrix is on its grid, 1 otherwise.
"""

import argparse
import json
import os
import sys

import torch
from safetensors import safe_open

MANTISSA_BITS = {"bfp4": 3, "bfp8": 7}
SKIP = ("mtp.", "visual.", "vision", "audio", "depth_decoder")


def on_grid(w_in_out: torch.Tensor, bits: int) -> torch.Tensor:
    """Re-quantize a [in, out] matrix to its block grid; return the re-quantized matrix (float32)."""
    x = w_in_out.float()
    rows, cols = x.shape
    pad = (-cols) % 16
    if pad:
        x = torch.nn.functional.pad(x, (0, pad))
    blocks = x.reshape(rows, -1, 16)
    # the biased exponent field of each value (float32 and bf16 share the 8-bit exponent)
    expo = (blocks.abs().view(torch.int32) >> 23) & 0xFF
    shared = expo.amax(dim=-1, keepdim=True)  # the block's exponent: its largest
    # step of the mantissa grid: a value with the shared exponent is 1.f x 2^(E-127);
    # with `bits` mantissa bits (the leading one included) the step is 2^(E-127-(bits-1))
    step = torch.ldexp(
        torch.ones_like(shared, dtype=torch.float32), (shared - 127 - (bits - 1)).to(torch.int32)
    )
    m = torch.round(blocks.abs() / step)  # torch.round rounds half to even
    m = m.clamp(max=2**bits - 1)
    q = torch.sign(blocks) * m * step
    q = q.reshape(rows, -1)[:, :cols]
    return q


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("model_dir", nargs="?", default=".")
    ap.add_argument("--layers", help="comma-separated layer numbers to check (default: all)")
    a = ap.parse_args()
    d = a.model_dir
    manifest = json.load(open(os.path.join(d, "quant4.json")))
    weight_map = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
    want = set(a.layers.split(",")) if a.layers else None
    checked, off = 0, 0
    handles = {}
    for layer, projs in manifest["layers"].items():
        if want and layer not in want:
            continue
        for name, info in projs.items():
            fmt = info.get("format") if isinstance(info, dict) else None
            if fmt not in MANTISSA_BITS:
                continue
            keys = [
                k
                for k in weight_map
                if k.endswith(f"layers.{layer}.{name}.weight") and not any(s in k for s in SKIP)
            ]
            if len(keys) != 1:
                print(f"layer {layer} {name}: {len(keys)} matching tensors, skipped")
                continue
            f = weight_map[keys[0]]
            if f not in handles:
                handles[f] = safe_open(os.path.join(d, f), "pt")
            w = handles[f].get_tensor(keys[0])
            mats = w if w.dim() == 3 else w.unsqueeze(0)  # experts: [E, out, in]
            bad = 0
            for m in mats:
                w_in_out = m.t().contiguous()
                bad += int(not torch.equal(on_grid(w_in_out, MANTISSA_BITS[fmt]), w_in_out.float()))
            checked += 1
            if bad:
                off += 1
                print(f"OFF GRID: layer {layer} {name} ({fmt}), {bad} of {len(mats)} matrices")
    print(f"{checked} matrices checked, {off} off the grid")
    return 0 if off == 0 and checked > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
