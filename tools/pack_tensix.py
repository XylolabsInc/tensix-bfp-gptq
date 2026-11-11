#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Xylolabs
"""Pack this checkpoint's quantized matrices into TT-NN's block floating-point formats and check the round trip.

For each linear projection that `quant4.json` lists as bfp4 or bfp8, the PyTorch weight [out, in]
is transposed to [in, out] (the layout TT-NN's matmul reads), packed with
`ttnn.as_tensor(..., dtype=ttnn.bfloat4_b or ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)`, unpacked
with `ttnn.to_torch`, and compared with the stored bf16 values: they must be equal, bit for bit,
because every value already sits on the format's grid. With --device the packed tensor also goes
to a Tensix device's DRAM and back. With --cache-dir each packed tensor is written as a TT-NN
weight cache file that `ttnn.as_tensor(..., cache_file_name=...)` loads later without re-packing.

Needs ttnn (github.com/tenstorrent/tt-metal), torch and safetensors.

Expert tensors stored 3-D ([experts, out, in]) are packed one expert matrix at a time. With
--as bfp8 every matrix is packed into bfloat8_b whatever its own format: a bfloat4_b value is also
on the bfloat8_b grid of the same block, so this must reproduce the stored values too.

Usage: python pack_tensix.py [MODEL_DIR] [--layers 0,1] [--device] [--cache-dir DIR] [--as bfp4|bfp8]
Exit status 0 when every packed matrix unpacks to its stored values, 1 otherwise.
"""

import argparse
import json
import os
import sys

import torch
import ttnn
from safetensors import safe_open

DTYPES = {"bfp4": ttnn.bfloat4_b, "bfp8": ttnn.bfloat8_b}
SKIP = ("mtp.", "visual.", "vision", "audio", "depth_decoder")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("model_dir", nargs="?", default=".")
    ap.add_argument("--layers", help="comma-separated layer numbers (default: all)")
    ap.add_argument(
        "--device", action="store_true", help="also write each tensor to device 0 and read it back"
    )
    ap.add_argument("--cache-dir", help="write TT-NN weight cache files here")
    ap.add_argument(
        "--as",
        dest="as_fmt",
        choices=sorted(DTYPES),
        help="pack every matrix in this format instead",
    )
    a = ap.parse_args()
    d = a.model_dir
    manifest = json.load(open(os.path.join(d, "quant4.json")))
    weight_map = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
    want = set(a.layers.split(",")) if a.layers else None
    device = ttnn.open_device(device_id=0) if a.device else None
    packed = mismatched = 0
    handles = {}
    try:
        for layer, projs in manifest["layers"].items():
            if want and layer not in want:
                continue
            for name, info in projs.items():
                fmt = info.get("format") if isinstance(info, dict) else None
                if fmt not in DTYPES:
                    continue
                keys = [
                    k
                    for k in weight_map
                    if k.endswith(f"layers.{layer}.{name}.weight") and not any(s in k for s in SKIP)
                ]
                if len(keys) != 1:
                    continue
                f = weight_map[keys[0]]
                if f not in handles:
                    handles[f] = safe_open(os.path.join(d, f), "pt")
                full = handles[f].get_tensor(keys[0])
                mats = full if full.dim() == 3 else full.unsqueeze(0)  # experts: [E, out, in]
                use = a.as_fmt or fmt
                for e, m in enumerate(mats):
                    w = m.t().contiguous()  # [in, out]: the layout TT-NN's matmul reads
                    tag = f"{keys[0]}.{e}" if full.dim() == 3 else keys[0]
                    cache = os.path.join(a.cache_dir, f"{tag}.{use}") if a.cache_dir else None
                    t = ttnn.as_tensor(
                        w,
                        dtype=DTYPES[use],
                        layout=ttnn.TILE_LAYOUT,
                        device=device,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG if device else None,
                        cache_file_name=cache,
                    )
                    back = ttnn.to_torch(t)
                    packed += 1
                    if not torch.equal(back.float(), w.float()):
                        mismatched += 1
                        diff = (back.float() != w.float()).sum().item()
                        print(
                            f"MISMATCH {tag} ({fmt} packed as {use}): {diff} of {w.numel()} values differ"
                        )
    finally:
        if device is not None:
            ttnn.close_device(device)
    print(f"{packed} matrices packed with TT-NN, {mismatched} differ from the stored values")
    return 0 if packed and not mismatched else 1


if __name__ == "__main__":
    sys.exit(main())
