#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Xylolabs
"""Pack this checkpoint's quantized matrices into TT-NN's block floating-point formats and check the round trip.

For each tensor that `quant4.json` declares as bfp4 or bfp8 (layer projections, the LM head,
`global_linears` and `extra`), the PyTorch weight [out, in]
is transposed to [in, out] (the layout TT-NN's matmul reads), packed with
`ttnn.as_tensor(..., dtype=ttnn.bfloat4_b or ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)`, unpacked
with `ttnn.to_torch`, and compared with the stored bf16 values: they must be equal, bit for bit,
because every value already sits on the format's grid. With --device the packed tensor also goes
to a Tensix device's DRAM and back. With --cache-dir each packed tensor is written as a TT-NN
weight cache file that `ttnn.as_tensor(..., cache_file_name=...)` loads later without re-packing.

A top-level `qkv_row_order` in `quant4.json` names a row order (see row_order.py) that is applied to
each fused `self_attn.q_k_v_proj` before it is packed.

Needs ttnn (github.com/tenstorrent/tt-metal), torch and safetensors.

Expert tensors stored 3-D ([experts, out, in]) are packed one expert matrix at a time. With
--as bfp8 every matrix is packed into bfloat8_b whatever its own format: a bfloat4_b value is also
on the bfloat8_b grid of the same block, so this must reproduce the stored values too.

Usage: python pack_tensix.py [MODEL_DIR] [--layers 0,1] [--device] [--cache-dir DIR] [--as bfp4|bfp8]
Exit status 0 when every packed matrix unpacks to its stored values, 1 otherwise.
"""

import argparse
import hashlib
import json
import os
import sys

import torch
import ttnn
from safetensors import safe_open

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "tensix_bfp")]
import packed as packed_checkpoint  # noqa: E402
import row_order  # noqa: E402

DTYPES = {"bfp4": ttnn.bfloat4_b, "bfp8": ttnn.bfloat8_b}


def quantized_weights(manifest, weight_map, layers=None):
    """Every declared quantized tensor once, whichever sections of the manifest name it."""
    return packed_checkpoint.declared_tensors(manifest, weight_map, layers)


def cache_file(cache_dir, tag, w, fmt):
    """The TT-NN cache file of matrix w as packed: named by its tensor, format, dtype, shape and
    bytes, so that a changed weight or row order gets a new file instead of an old one."""
    key = f"{tag}\0{fmt}\0{w.dtype}\0{list(w.shape)}\0{packed_checkpoint.sha256(w)}"
    return os.path.join(cache_dir, f"{hashlib.sha256(key.encode()).hexdigest()}.{fmt}")


def index_weight_map(model_dir):
    """The index's weight map; exits with the error when a shard name leaves the directory."""
    index = os.path.join(model_dir, packed_checkpoint.INDEX)
    if not os.path.exists(index):
        raise SystemExit(f"ERROR: {index} is missing")
    try:
        return packed_checkpoint.shards(model_dir)
    except ValueError as e:
        raise SystemExit(f"ERROR: {e}") from None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("model_dir", nargs="?", default=".")
    ap.add_argument(
        "--layers",
        help="layer numbers to check; the quantized LM head, global linears and extra matrices are "
        "always checked",
    )
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
    order = row_order.check_declared(manifest)
    config = row_order.load_config(d, order)
    weight_map = index_weight_map(d)
    want = set(a.layers.split(",")) if a.layers else None
    weights = list(quantized_weights(manifest, weight_map, want))
    if a.cache_dir:
        os.makedirs(a.cache_dir, exist_ok=True)
    device = ttnn.open_device(device_id=0) if a.device else None
    packed = mismatched = 0
    handles = {}
    try:
        for name, fmt in weights:
            f = weight_map[name]
            if f not in handles:
                handles[f] = safe_open(os.path.join(d, f), "pt")
            full = handles[f].get_tensor(name)
            mats = full if full.dim() == 3 else full.unsqueeze(0)  # experts: [E, out, in]
            use = a.as_fmt or fmt
            for e, m in enumerate(mats):
                m = row_order.reorder(m, name, config, order)
                w = m.t().contiguous()  # [in, out]: the layout TT-NN's matmul reads
                tag = f"{name}.{e}" if full.dim() == 3 else name
                cache = cache_file(a.cache_dir, tag, w, use) if a.cache_dir else None
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
