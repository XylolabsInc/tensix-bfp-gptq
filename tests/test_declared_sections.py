# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Xylolabs
"""Every declared section (layers, LM head, global linears, extras) is resolved and packed the same
way by packed.py and the two check tools; conflicting declarations and exports the packer cannot
handle are refused."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from tensix_bfp import packed

ROOT = Path(__file__).resolve().parents[1]
INDEX = "model.safetensors.index.json"
LAYER = "model.layers.0.self_attn.q_proj.weight"
HEAD = "lm_head.weight"
GLOBAL = "model.per_layer_projection.weight"
EXTRA = "model.extra_matrix.weight"


def ones(*shape, value=1.0):
    return torch.full(shape, value, dtype=torch.bfloat16)


def checkpoint(d, manifest=None, tensors=None):
    """A one-file export: a layer projection, the LM head, a global linear and an extra matrix."""
    d.mkdir(parents=True, exist_ok=True)
    tensors = tensors or {
        LAYER: ones(32, 32),
        HEAD: ones(32, 32),
        GLOBAL: ones(32, 64),
        EXTRA: ones(64, 32),
        "model.norm.weight": ones(32),
    }
    base = {
        "layers": {"0": {"self_attn.q_proj": {"format": "bfp4"}}},
        "lm_head": "bfp8",
        "global_linears": {GLOBAL: {"format": "bfp4"}},
        "extra": {EXTRA: {"format": "bfp8"}},
    }
    base.update(manifest or {})
    save_file(tensors, str(d / "weights.safetensors"), metadata={"format": "pt"})
    (d / INDEX).write_text(
        json.dumps({"weight_map": dict.fromkeys(tensors, "weights.safetensors")})
    )
    (d / "quant4.json").write_text(json.dumps(base))
    return d


def run_tool(tool, d):
    return subprocess.run(
        [sys.executable, "-I", str(ROOT / "tools" / tool), str(d)], capture_output=True, text=True
    )


def test_pack_covers_every_declared_section(tmp_path):
    src = checkpoint(tmp_path / "src")
    records, kept = packed.pack(src, tmp_path / "out")
    assert sorted(records) == sorted([LAYER, HEAD, GLOBAL, EXTRA]) and kept == []
    assert records[LAYER]["format"] == "bfp4" and records[EXTRA]["format"] == "bfp8"
    out = load_file(str(tmp_path / "out" / "model.safetensors"))
    for name in records:
        assert out[name].dtype == torch.uint8, name
    assert out["model.norm.weight"].dtype == torch.bfloat16
    assert packed.verify(tmp_path / "out") == ([], 4)
    assert packed.main(["pack", str(src), str(tmp_path / "again")]) == 0


def test_bf16_entries_stay_bf16(tmp_path):
    src = checkpoint(
        tmp_path / "src",
        {"lm_head": "bf16", "extra": {EXTRA: {"format": "bf16"}}, "global_linears": {}},
    )
    records, _ = packed.pack(src, tmp_path / "out")
    assert sorted(records) == [LAYER]


@pytest.mark.parametrize("section", ["extra", "global_linears"])
def test_missing_declared_name_fails(tmp_path, section):
    tensors = {LAYER: ones(32, 32), HEAD: ones(32, 32)}
    src = checkpoint(
        tmp_path / "src",
        {"global_linears": {}, "extra": {}, section: {"model.absent.weight": {"format": "bfp4"}}},
        tensors,
    )
    with pytest.raises(ValueError, match="model.absent.weight"):
        packed.pack(src, tmp_path / "out")
    assert not (tmp_path / "out").exists()
    result = run_tool("verify_grid.py", src)
    assert result.returncode == 1 and "model.absent.weight" in result.stderr


def test_name_with_and_without_weight_fails(tmp_path):
    tensors = {
        LAYER: ones(32, 32),
        HEAD: ones(32, 32),
        "model.both": ones(32, 32),
        "model.both.weight": ones(32, 32),
    }
    src = checkpoint(
        tmp_path / "src",
        {"global_linears": {}, "extra": {"model.both": {"format": "bfp4"}}},
        tensors,
    )
    with pytest.raises(ValueError, match="with and without"):
        packed.pack(src, tmp_path / "out")
    assert not (tmp_path / "out").exists()
    assert run_tool("verify_grid.py", src).returncode == 1


def test_name_without_weight_suffix_resolves(tmp_path):
    src = checkpoint(tmp_path / "src", {"extra": {EXTRA[: -len(".weight")]: {"format": "bfp8"}}})
    records, _ = packed.pack(src, tmp_path / "out")
    assert EXTRA in records


def test_tensor_declared_in_two_sections_is_packed_once(tmp_path):
    src = checkpoint(
        tmp_path / "src",
        {"global_linears": {GLOBAL: {"format": "bfp4"}}, "extra": {GLOBAL: {"format": "bfp4"}}},
        {LAYER: ones(32, 32), HEAD: ones(32, 32), GLOBAL: ones(32, 64)},
    )
    manifest = json.loads((src / "quant4.json").read_text())
    pairs = packed.declared_tensors(manifest, packed.shards(src))
    assert [n for n, _ in pairs].count(GLOBAL) == 1
    records, _ = packed.pack(src, tmp_path / "out")
    assert sorted(records) == sorted([LAYER, HEAD, GLOBAL])
    result = run_tool("verify_grid.py", src)
    assert "3 matrices checked, 0 off the grid" in result.stdout


def conflicting(d):
    """The LM head declared bfp8 and under extra as bfp4, its values on the bfp8 grid only."""
    return checkpoint(
        d,
        {"global_linears": {}, "extra": {HEAD: {"format": "bfp4"}}},
        {LAYER: ones(32, 32), HEAD: ones(32, 32, value=1.015625)},
    )


def test_conflicting_formats_fail_in_the_packer_and_verify_grid(tmp_path):
    src = conflicting(tmp_path / "src")
    with pytest.raises(ValueError) as e:
        packed.pack(src, tmp_path / "out")
    assert HEAD in str(e.value) and "bfp8" in str(e.value) and "bfp4" in str(e.value)
    assert not (tmp_path / "out").exists()
    result = run_tool("verify_grid.py", src)
    assert result.returncode == 1 and "declared as bfp8 and as bfp4" in result.stderr
    assert "0 matrices checked" not in result.stdout


def test_conflict_is_found_whatever_the_order_and_layer_selection(tmp_path):
    src = checkpoint(
        tmp_path / "src",
        {
            "lm_head": None,
            "global_linears": {HEAD: {"format": "bfp4"}},
            "extra": {HEAD: {"format": "bfp8"}},
            "layers": {"0": {"self_attn.q_proj": {"format": "bfp4"}}},
        },
        {LAYER: ones(32, 32), HEAD: ones(32, 32)},
    )
    manifest = json.loads((src / "quant4.json").read_text())
    with pytest.raises(ValueError, match="declared as bfp4 and as bfp8"):
        packed.declared_tensors(manifest, packed.shards(src), layers={"7"})
    manifest["layers"]["3"] = {"self_attn.q_proj": {"format": "bfp8"}}
    weight_map = dict.fromkeys([LAYER, HEAD], "x")
    weight_map["model.layers.3.self_attn.q_proj.weight"] = "x"
    manifest["global_linears"] = {}
    manifest["extra"] = {"model.layers.3.self_attn.q_proj": {"format": "bfp4"}}
    with pytest.raises(ValueError, match="declared as bfp8 and as bfp4"):
        packed.declared_tensors(manifest, weight_map, layers={"0"})


def test_same_format_twice_is_fine(tmp_path):
    src = checkpoint(tmp_path / "src", {"extra": {HEAD: {"format": "bfp8"}}})
    records, _ = packed.pack(src, tmp_path / "out")
    assert HEAD in records


@pytest.mark.parametrize("where", ["lm_head", "global_linears", "extra", "layers"])
def test_unknown_format_is_an_error(tmp_path, where):
    manifest = {
        "lm_head": {"lm_head": "bfp2"},
        "global_linears": {"global_linears": {GLOBAL: {"format": "bfp2"}}},
        "extra": {"extra": {EXTRA: {"format": "fp8"}}},
        "layers": {"layers": {"0": {"self_attn.q_proj": {"format": "bfp5"}}}},
    }[where]
    src = checkpoint(tmp_path / "src", manifest)
    with pytest.raises(ValueError, match="neither bfp4, bfp8 nor bf16"):
        packed.pack(src, tmp_path / "out")
    assert not (tmp_path / "out").exists()
    assert run_tool("verify_grid.py", src).returncode == 1


def test_qkv_row_order_export_is_refused_by_the_packer(tmp_path, capsys):
    src = checkpoint(tmp_path / "src", {"qkv_row_order": "spark-llama"})
    with pytest.raises(ValueError, match="qkv_row_order.*not support"):
        packed.pack(src, tmp_path / "out")
    assert not (tmp_path / "out").exists()
    assert packed.main(["pack", str(src), str(tmp_path / "out")]) == 1
    assert "qkv_row_order" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()
    # the check tools keep accepting a declared order (the tensors here are not fused projections)
    (src / "config.json").write_text("{}")
    assert run_tool("verify_grid.py", src).returncode == 0


def test_too_large_fused_projection_raises_without_asserts():
    code = (
        "import sys, torch\n"
        f"sys.path.insert(0, {str(ROOT / 'tools')!r})\n"
        "import row_order\n"
        "config = {'head_dim': 8, 'num_attention_heads': 2, 'num_key_value_heads': 1,"
        " 'layer_types': ['full_attention'],"
        " 'rope_parameters': {'full_attention': {'partial_rotary_factor': 1.0}}}\n"
        "name = 'model.layers.0.self_attn.q_k_v_proj.weight'\n"
        "try:\n"
        "    row_order.reorder(torch.zeros(40, 8), name, config, 'spark-llama')\n"
        "except (ValueError, SystemExit) as e:\n"
        "    print('RAISED', e)\n"
        "else:\n"
        "    print('SILENT')\n"
    )
    for flags in ([], ["-O"]):
        r = subprocess.run([sys.executable, *flags, "-c", code], capture_output=True, text=True)
        assert "RAISED" in r.stdout, (flags, r.stdout, r.stderr)
        assert "has 40 rows, expected" in r.stdout and "= 32" in r.stdout


@pytest.mark.parametrize("head,extra", [("bf16", "bfp4"), ("bfp4", "bf16")])
def test_bf16_conflicts_with_block_format(head, extra):
    manifest = {"lm_head": head, "extra": {HEAD: {"format": extra}}}
    with pytest.raises(ValueError, match="declared as"):
        packed.declared_tensors(manifest, {HEAD: "weights.safetensors"})


def test_pack_rejects_quantized_tensor_missing_from_indexed_shard(tmp_path):
    src = checkpoint(tmp_path / "src")
    index = json.loads((src / INDEX).read_text())
    index["weight_map"]["missing.weight"] = "weights.safetensors"
    (src / INDEX).write_text(json.dumps(index))
    manifest = json.loads((src / "quant4.json").read_text())
    manifest["extra"]["missing.weight"] = {"format": "bfp4"}
    (src / "quant4.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="missing.weight"):
        packed.pack(src, tmp_path / "out")
