# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Xylolabs
"""The standalone checkers must cover every declared quantized projection."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import row_order  # noqa: E402
import verify_grid  # noqa: E402


def checkpoint(directory, tensors, layers, head="bf16", head_name=None, extra=None):
    save_file(tensors, str(directory / "weights.safetensors"))
    manifest = {"layers": layers, "lm_head": head}
    if head_name is not None:
        manifest["lm_head_tensor"] = head_name
    manifest.update(extra or {})
    (directory / "quant4.json").write_text(json.dumps(manifest))
    (directory / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: "weights.safetensors" for name in tensors}})
    )


def verify(directory, *args):
    return subprocess.run(
        [sys.executable, "-I", str(ROOT / "tools/verify_grid.py"), str(directory), *args],
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("fmt,value", [("bfp4", 1.25), ("bfp8", 1.015625)])
def test_head_only_checkpoint(tmp_path, fmt, value):
    checkpoint(
        tmp_path,
        {"model.embed_tokens.weight": torch.full((32, 32), value, dtype=torch.bfloat16)},
        {},
        fmt,
        "model.embed_tokens.weight",
    )
    result = verify(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 matrices checked, 0 off the grid" in result.stdout


@pytest.mark.parametrize("fmt,bad", [("bfp4", 1.125), ("bfp8", 1.0078125)])
def test_off_grid_head_fails_even_with_layer_selection(tmp_path, fmt, bad):
    head = torch.ones(32, 32, dtype=torch.bfloat16)
    head[0, 0] = bad
    checkpoint(
        tmp_path,
        {
            "model.layers.0.mlp.up_proj.weight": torch.ones(32, 32, dtype=torch.bfloat16),
            "lm_head.weight": head,
        },
        {"0": {"mlp.up_proj": {"format": "bfp4"}}},
        fmt,
    )
    result = verify(tmp_path, "--layers", "0")
    assert result.returncode == 1
    assert "OFF GRID: lm_head.weight" in result.stdout
    assert "2 matrices checked, 1 off the grid" in result.stdout


def test_numbered_experts_are_all_checked(tmp_path):
    tensors = {
        f"model.layers.0.mlp.experts.{i}.up_proj.weight": torch.ones(32, 32, dtype=torch.bfloat16)
        for i in range(2)
    }
    tensors["model.layers.0.mlp.experts.1.up_proj.weight"][0, 0] = 1.125
    checkpoint(tmp_path, tensors, {"0": {"mlp.experts.up_proj": {"format": "bfp4"}}})
    result = verify(tmp_path)
    assert result.returncode == 1
    assert "OFF GRID: model.layers.0.mlp.experts.1.up_proj.weight" in result.stdout
    assert "2 matrices checked, 1 off the grid" in result.stdout


@pytest.mark.parametrize("missing_up", [False, True])
def test_grouped_gate_up_covers_both_projections(tmp_path, missing_up):
    tensors = {
        f"model.layers.0.mlp.experts.{i}.{projection}.weight": torch.ones(
            32, 32, dtype=torch.bfloat16
        )
        for i in range(2)
        for projection in ("gate_proj", "up_proj")
    }
    if missing_up:
        del tensors["model.layers.0.mlp.experts.1.up_proj.weight"]
    checkpoint(tmp_path, tensors, {"0": {"mlp.experts.gate_up_proj": {"format": "bfp4"}}})
    result = verify(tmp_path)
    assert result.returncode == int(missing_up), result.stdout + result.stderr
    if not missing_up:
        assert "4 matrices checked, 0 off the grid" in result.stdout


@pytest.mark.parametrize("ambiguous", [False, True])
def test_missing_or_ambiguous_projection_cannot_pass(tmp_path, ambiguous):
    tensors = {"model.layers.0.mlp.valid.weight": torch.ones(32, 32, dtype=torch.bfloat16)}
    if ambiguous:
        for prefix in ["a", "b"]:
            tensors[f"{prefix}.layers.0.mlp.other.weight"] = torch.ones(
                32, 32, dtype=torch.bfloat16
            )
    checkpoint(
        tmp_path,
        tensors,
        {"0": {"mlp.valid": {"format": "bfp4"}, "mlp.other": {"format": "bfp4"}}},
    )
    result = verify(tmp_path)
    assert result.returncode == 1
    assert "no unambiguous tensors" in result.stderr


def test_missing_declared_head_cannot_pass(tmp_path):
    checkpoint(
        tmp_path,
        {"model.layers.0.mlp.up_proj.weight": torch.ones(32, 32, dtype=torch.bfloat16)},
        {"0": {"mlp.up_proj": {"format": "bfp4"}}},
        "bfp4",
        "custom.head.weight",
    )
    result = verify(tmp_path)
    assert result.returncode == 1
    assert "quantized LM head 'custom.head.weight' is missing" in result.stderr


def test_unquantized_head_needs_no_tensor(tmp_path):
    checkpoint(
        tmp_path,
        {"model.layers.0.mlp.up_proj.weight": torch.ones(32, 32, dtype=torch.bfloat16)},
        {"0": {"mlp.up_proj": {"format": "bfp4"}}},
    )
    result = verify(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("fmt,value", [("bfp4", 1.25), ("bfp8", 1.015625)])
@pytest.mark.parametrize("fused", [False, True])
def test_bare_parameter_is_checked(tmp_path, fmt, value, fused):
    projection = "mlp.experts.gate_up_proj" if fused else "mlp.up_proj"
    key = f"model.layers.0.{projection}"
    shape = (2, 32, 32) if fused else (32, 32)
    checkpoint(
        tmp_path,
        {key: torch.full(shape, value, dtype=torch.bfloat16)},
        {"0": {projection: {"format": fmt}}},
    )
    result = verify(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 matrices checked, 0 off the grid" in result.stdout


def test_second_fused_expert_is_checked(tmp_path):
    key = "model.layers.0.mlp.experts.gate_up_proj"
    values = torch.ones(2, 32, 32, dtype=torch.bfloat16)
    values[1, 0, 0] = 1.125
    checkpoint(tmp_path, {key: values}, {"0": {"mlp.experts.gate_up_proj": {"format": "bfp4"}}})
    result = verify(tmp_path)
    assert result.returncode == 1
    assert "1 of 2 matrices" in result.stdout


def test_bare_and_weight_aliases_are_ambiguous(tmp_path):
    key = "model.layers.0.mlp.up_proj"
    tensors = {name: torch.ones(32, 32, dtype=torch.bfloat16) for name in [key, key + ".weight"]}
    checkpoint(tmp_path, tensors, {"0": {"mlp.up_proj": {"format": "bfp4"}}})
    result = verify(tmp_path)
    assert result.returncode == 1
    assert "no unambiguous tensors" in result.stderr


@pytest.mark.parametrize("ambiguous", [False, True])
def test_numbered_bare_parameters(tmp_path, ambiguous):
    prefix = "model.layers.0.mlp.experts."
    tensors = {
        prefix + str(i) + ".up_proj": torch.ones(32, 32, dtype=torch.bfloat16) for i in range(2)
    }
    if ambiguous:
        tensors[prefix + "0.up_proj.weight"] = torch.ones(32, 32, dtype=torch.bfloat16)
    checkpoint(tmp_path, tensors, {"0": {"mlp.experts.up_proj": {"format": "bfp4"}}})
    result = verify(tmp_path)
    assert result.returncode == int(ambiguous), result.stdout + result.stderr
    if not ambiguous:
        assert "2 matrices checked, 0 off the grid" in result.stdout


def test_head_order_sliding():
    assert row_order.head_order(8, 8) == [0, 4, 1, 5, 2, 6, 3, 7]


def test_head_order_partial():
    assert row_order.head_order(16, 4) == [0, 2, 1, 3, 4, 10, 5, 11, 6, 12, 7, 13, 8, 14, 9, 15]


@pytest.mark.parametrize("head_dim,rope_dim", [(256, 64), (256, 256), (128, 32)])
def test_head_order_is_permutation(head_dim, rope_dim):
    assert sorted(row_order.head_order(head_dim, rope_dim)) == list(range(head_dim))


SPARK_CONFIG = {
    "head_dim": 8,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "layer_types": ["sliding_attention", "full_attention"],
    "rope_parameters": {
        "sliding_attention": {"partial_rotary_factor": 1.0},
        "full_attention": {"partial_rotary_factor": 0.5},
    },
}


def test_fused_qkv_order_shapes():
    for layer, rope_dim in enumerate((8, 4)):
        order = row_order.fused_qkv_order(SPARK_CONFIG, layer, "spark-llama")
        assert len(order) == 32
        assert order[24:] == list(range(24, 32))
        within = row_order.head_order(8, rope_dim)
        for head in range(3):
            assert order[head * 8 : head * 8 + 8] == [head * 8 + j for j in within]


def test_declared_order_passes_and_raw_fails(tmp_path):
    # head_dim 32 so the permutation moves rows across the 16-row blocks that share an exponent
    config = dict(SPARK_CONFIG, head_dim=32)
    torch.manual_seed(0)
    order = row_order.fused_qkv_order(config, 0, "spark-llama")
    assert len(order) == 128
    permuted = torch.randn(128, 32).to(torch.bfloat16)
    snapped = verify_grid.on_grid(permuted.t().contiguous(), 7).t().contiguous()
    snapped = snapped.to(torch.bfloat16)
    stored = torch.empty_like(snapped)
    stored[torch.tensor(order)] = snapped
    layers = {"0": {"self_attn.q_k_v_proj": {"format": "bfp8"}}}
    tensors = {"model.layers.0.self_attn.q_k_v_proj.weight": stored}
    (tmp_path / "config.json").write_text(json.dumps(config))
    checkpoint(tmp_path, tensors, layers, extra={"qkv_row_order": "spark-llama"})
    result = verify(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 matrices checked, 0 off the grid" in result.stdout
    checkpoint(tmp_path, tensors, layers)
    result = verify(tmp_path)
    assert result.returncode == 1
    assert "OFF GRID: model.layers.0.self_attn.q_k_v_proj.weight" in result.stdout


def test_unknown_order_refused(tmp_path):
    checkpoint(
        tmp_path,
        {"model.layers.0.self_attn.q_k_v_proj.weight": torch.ones(32, 32, dtype=torch.bfloat16)},
        {"0": {"self_attn.q_k_v_proj": {"format": "bfp8"}}},
        extra={"qkv_row_order": "other"},
    )
    result = verify(tmp_path)
    assert result.returncode != 0
    assert "'other'" in result.stderr


GLOBAL = "model.per_layer_model_projection.weight"


def global_checkpoint(directory, record=None, name=GLOBAL, off_grid=False, present=True):
    glob = torch.full((32, 64), 1.25, dtype=torch.bfloat16)
    if off_grid:
        glob[0, 0] = 1.125
    tensors = {"model.layers.0.mlp.up_proj.weight": torch.ones(32, 32, dtype=torch.bfloat16)}
    if present:
        tensors[GLOBAL] = glob
    checkpoint(directory, tensors, {"0": {"mlp.up_proj": {"format": "bfp4"}}})
    manifest = json.loads((directory / "quant4.json").read_text())
    manifest["global_linears"] = {name: record or {"format": "bfp4", "rounding": "rtn"}}
    (directory / "quant4.json").write_text(json.dumps(manifest))


def test_global_linear_is_checked(tmp_path):
    global_checkpoint(tmp_path)
    result = verify(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "2 matrices checked, 0 off the grid" in result.stdout


def test_global_linear_off_grid_fails(tmp_path):
    global_checkpoint(tmp_path, off_grid=True)
    result = verify(tmp_path)
    assert result.returncode == 1
    assert f"OFF GRID: {GLOBAL}" in result.stdout


def test_global_linear_bf16_entry_is_skipped(tmp_path):
    global_checkpoint(tmp_path, record={"format": "bf16"}, off_grid=True)
    result = verify(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 matrices checked, 0 off the grid" in result.stdout


def test_global_linear_missing_fails(tmp_path):
    global_checkpoint(tmp_path, name="model.absent_projection", present=False)
    result = verify(tmp_path)
    assert result.returncode == 1
    assert "model.absent_projection" in result.stderr


def test_global_linear_name_without_suffix_and_ambiguity(tmp_path):
    global_checkpoint(tmp_path, name=GLOBAL[: -len(".weight")])
    result = verify(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "2 matrices checked, 0 off the grid" in result.stdout
    tensors = {
        "model.layers.0.mlp.up_proj.weight": torch.ones(32, 32, dtype=torch.bfloat16),
        GLOBAL: torch.ones(32, 64, dtype=torch.bfloat16),
        GLOBAL[: -len(".weight")]: torch.ones(32, 64, dtype=torch.bfloat16),
    }
    checkpoint(tmp_path, tensors, {"0": {"mlp.up_proj": {"format": "bfp4"}}})
    manifest = json.loads((tmp_path / "quant4.json").read_text())
    manifest["global_linears"] = {GLOBAL[: -len(".weight")]: {"format": "bfp4"}}
    (tmp_path / "quant4.json").write_text(json.dumps(manifest))
    result = verify(tmp_path)
    assert result.returncode == 1
    assert GLOBAL[: -len(".weight")] in result.stderr


def test_layers_option_keeps_global_linears(tmp_path):
    global_checkpoint(tmp_path)
    result = verify(tmp_path, "--layers", "5")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 matrices checked, 0 off the grid" in result.stdout


def test_global_linear_named_twice_is_counted_once(tmp_path):
    global_checkpoint(tmp_path)
    manifest = json.loads((tmp_path / "quant4.json").read_text())
    manifest["lm_head"] = "bfp4"
    manifest["lm_head_tensor"] = GLOBAL
    (tmp_path / "quant4.json").write_text(json.dumps(manifest))
    result = verify(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "2 matrices checked, 0 off the grid" in result.stdout
