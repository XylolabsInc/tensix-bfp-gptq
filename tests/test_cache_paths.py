# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Xylolabs
"""Exercise cache filenames without a TT-NN device."""

import importlib.util
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

TOOL = Path(__file__).resolve().parents[1] / "tools" / "pack_tensix.py"


def load_tool(monkeypatch, as_tensor):
    """tools/pack_tensix.py with a stand-in ttnn whose as_tensor is the given function."""
    fake = SimpleNamespace(
        bfloat4_b="bfp4",
        bfloat8_b="bfp8",
        TILE_LAYOUT="tile",
        as_tensor=as_tensor,
        to_torch=lambda t: t,
    )
    monkeypatch.setitem(sys.modules, "ttnn", fake)
    spec = importlib.util.spec_from_file_location("pack_tensix_test", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cache_paths_are_confined_and_directory_is_created(tmp_path, monkeypatch):
    paths = []

    def as_tensor(w, **kwargs):
        paths.append(Path(kwargs["cache_file_name"]))
        return w

    module = load_tool(monkeypatch, as_tensor)
    names = ["../escape", "folder/name", "experts/name"]
    save_file(
        {
            names[0]: torch.ones(32, 32),
            names[1]: torch.ones(32, 32),
            names[2]: torch.ones(2, 32, 32),
        },
        str(tmp_path / "weights.safetensors"),
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(names, "weights.safetensors")})
    )
    (tmp_path / "quant4.json").write_text(
        json.dumps({"extra": {name: {"format": "bfp4"} for name in names}})
    )
    cache = tmp_path / "nested" / "cache"
    monkeypatch.setattr(sys, "argv", [str(TOOL), str(tmp_path), "--cache-dir", str(cache)])
    assert module.main() == 0
    first = paths.copy()
    assert cache.is_dir() and len(first) == len(set(first)) == 4
    assert all(p.parent == cache and re.fullmatch(r"[0-9a-f]{64}\.bfp4", p.name) for p in first)
    paths.clear()
    assert module.main() == 0
    assert paths == first


def test_cache_entry_of_a_changed_weight_is_not_reused(tmp_path, monkeypatch):
    def as_tensor(w, **kwargs):
        # as ttnn.as_tensor does: an existing cache file is loaded instead of packing w
        path = Path(kwargs["cache_file_name"])
        if not path.exists():
            torch.save(w, path)
        return torch.load(path)

    module = load_tool(monkeypatch, as_tensor)
    weights = {"a": torch.ones(32, 32), "b": torch.ones(32, 32)}
    save_file(weights, str(tmp_path / "weights.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(weights, "weights.safetensors")})
    )
    (tmp_path / "quant4.json").write_text(
        json.dumps({"extra": {name: {"format": "bfp4"} for name in weights}})
    )
    cache = tmp_path / "cache"
    monkeypatch.setattr(sys, "argv", [str(TOOL), str(tmp_path), "--cache-dir", str(cache)])
    assert module.main() == 0
    first = set(cache.iterdir())
    weights["a"][0, :16] = 0.5  # still on the bfp4 grid
    save_file(weights, str(tmp_path / "weights.safetensors"))
    assert module.main() == 0  # a's new values are packed and checked, not the cached ones
    second = set(cache.iterdir())
    assert len(first) == 2 and first < second and len(second) == 3
