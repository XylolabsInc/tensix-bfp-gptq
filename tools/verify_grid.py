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
import re
import sys

import torch
from safetensors import safe_open

MANTISSA_BITS = {"bfp4": 3, "bfp8": 7}
SKIP = ("mtp.", "visual.", "vision", "audio", "depth_decoder")


def quantized_weights(manifest, weight_map, layers=None):
    """Every declared quantized tensor, including numbered experts and the LM head."""
    for layer, projections in manifest.get("layers", {}).items():
        if layers is not None and layer not in layers:
            continue
        for name, record in projections.items():
            fmt = record.get("format") if isinstance(record, dict) else None
            if fmt not in ("bfp4", "bfp8"):
                continue
            suffixes = (f"layers.{layer}.{name}.weight", f"layers.{layer}.{name}")
            keys = [k for k in weight_map if k.endswith(suffixes) and not any(s in k for s in SKIP)]
            if not keys and ".experts." in name:
                module, projection = name.rsplit(".experts.", 1)

                def numbered(part):
                    pattern = re.compile(
                        rf"(?:^|\.)layers\.{re.escape(str(layer))}\.{re.escape(module)}"
                        rf"\.experts\.(\d+)\.{re.escape(part)}(?:\.weight)?$"
                    )
                    matches = [
                        (int(m.group(1)), k)
                        for k in weight_map
                        if (m := pattern.search(k)) and not any(s in k for s in SKIP)
                    ]
                    if len({i for i, _ in matches}) != len(matches):
                        raise ValueError(f"projection {layer}/{name} has ambiguous experts")
                    return sorted(matches)

                matches = numbered(projection)
                if matches:
                    yield from ((k, fmt) for _, k in matches)
                    continue
                if projection == "gate_up_proj":
                    gates, ups = numbered("gate_proj"), numbered("up_proj")
                    if gates and [i for i, _ in gates] == [i for i, _ in ups]:
                        for (_, gate), (_, up) in zip(gates, ups):
                            yield gate, fmt
                            yield up, fmt
                        continue
            if len(keys) != 1:
                raise ValueError(f"projection {layer}/{name} has no unambiguous tensors")
            yield keys[0], fmt
    fmt = manifest.get("lm_head")
    if fmt in ("bfp4", "bfp8"):
        name = manifest.get("lm_head_tensor", "lm_head.weight")
        if not isinstance(name, str) or name not in weight_map:
            raise ValueError(f"quantized LM head {name!r} is missing from the weight map")
        yield name, fmt


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
    ap.add_argument(
        "--layers", help="layer numbers to check; the quantized LM head is always checked"
    )
    a = ap.parse_args()
    d = a.model_dir
    manifest = json.load(open(os.path.join(d, "quant4.json")))
    weight_map = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
    want = set(a.layers.split(",")) if a.layers else None
    weights = list(quantized_weights(manifest, weight_map, want))
    checked, off = 0, 0
    handles = {}
    for name, fmt in weights:
        f = weight_map[name]
        if f not in handles:
            handles[f] = safe_open(os.path.join(d, f), "pt")
        w = handles[f].get_tensor(name)
        mats = w if w.dim() == 3 else w.unsqueeze(0)  # experts: [E, out, in]
        bad = 0
        for m in mats:
            w_in_out = m.t().contiguous()
            bad += int(not torch.equal(on_grid(w_in_out, MANTISSA_BITS[fmt]), w_in_out.float()))
        checked += 1
        if bad:
            off += 1
            print(f"OFF GRID: {name} ({fmt}), {bad} of {len(mats)} matrices")
    print(f"{checked} matrices checked, {off} off the grid")
    return 0 if off == 0 and checked > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
