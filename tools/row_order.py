# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Xylolabs
"""Row orders of a fused q/k/v projection, shared by verify_grid.py and pack_tensix.py.

Some exports store the fused projection `self_attn.q_k_v_proj` in the base model's row order, while
its values lie on the block floating-point grid only in another row order. `quant4.json` names that
order in its top-level field `qkv_row_order`; the check tools apply it to the projection's rows
before they check or pack the matrix. The stored files are never rewritten.

`spark-llama` puts each rotary pair of a q or k head side by side, as a runner with interleaved
rotary pairs expects, and uses the same permutation for q and k, so attention scores do not change.
The v rows stay in place.
"""

import json
import os
import re

import torch

ORDERS = ("spark-llama",)
FUSED = re.compile(r"(?:^|\.)layers\.(\d+)\.self_attn\.q_k_v_proj(?:\.weight)?$")
NEEDED = (
    "head_dim",
    "num_attention_heads",
    "num_key_value_heads",
    "layer_types",
    "rope_parameters",
)


def head_order(head_dim: int, rope_dim: int) -> list[int]:
    """For one q or k head: new slot -> old dimension."""
    half = head_dim // 2
    pairs = rope_dim // 2
    m = half - pairs
    placed = (
        list(range(pairs))
        + list(range(rope_dim, rope_dim + m))
        + list(range(pairs, rope_dim))
        + list(range(rope_dim + m, head_dim))
    )
    order = []
    for i in range(half):
        order += [placed[i], placed[half + i]]
    return order


def fused_qkv_order(config: dict, layer: int, order: str) -> list[int]:
    """For layer `layer`'s fused projection: new row -> old row, over q heads, k heads, then v heads."""
    if order not in ORDERS:
        raise SystemExit(f"unknown qkv_row_order {order!r}; known: {', '.join(ORDERS)}")
    for field in NEEDED:
        if field not in config:
            raise SystemExit(f"config.json has no {field!r}, which qkv_row_order {order!r} needs")
    head_dim = config["head_dim"]
    layer_type = config["layer_types"][layer]
    rope = config["rope_parameters"].get(layer_type, {})
    if "partial_rotary_factor" not in rope:
        raise SystemExit(
            f"config.json has no 'rope_parameters' entry with a 'partial_rotary_factor' "
            f"for layer type {layer_type!r}, which qkv_row_order {order!r} needs"
        )
    rope_dim = int(head_dim * rope["partial_rotary_factor"])
    within = head_order(head_dim, rope_dim)
    heads = config["num_attention_heads"] + config["num_key_value_heads"]
    rows = []
    for h in range(heads):
        rows += [h * head_dim + j for j in within]
    v_start = heads * head_dim
    rows += list(range(v_start, v_start + config["num_key_value_heads"] * head_dim))
    return rows


def check_declared(manifest: dict) -> str | None:
    """The manifest's declared order, or None; exits when it is not a known one."""
    order = manifest.get("qkv_row_order")
    if order is not None and order not in ORDERS:
        raise SystemExit(f"unknown qkv_row_order {order!r}; known: {', '.join(ORDERS)}")
    return order


def load_config(model_dir: str, order: str | None) -> dict | None:
    """The export's config.json when an order is declared, else None."""
    if order is None:
        return None
    path = os.path.join(model_dir, "config.json")
    if not os.path.exists(path):
        raise SystemExit(f"{path} is missing, which qkv_row_order {order!r} needs")
    with open(path) as f:
        return json.load(f)


def reorder(matrix: torch.Tensor, name: str, config: dict | None, order: str | None):
    """The [out, in] weight `name` with its rows reordered when it is a fused q/k/v projection."""
    found = FUSED.search(name)
    if order is None or found is None:
        return matrix
    rows = fused_qkv_order(config, int(found.group(1)), order)
    assert matrix.shape[0] == len(rows), (
        f"{name} has {matrix.shape[0]} rows, expected "
        f"(num_attention_heads + 2 * num_key_value_heads) * head_dim = {len(rows)}"
    )
    return matrix[torch.tensor(rows)]
