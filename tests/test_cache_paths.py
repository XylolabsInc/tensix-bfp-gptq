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


def test_cache_paths_are_confined_and_directory_is_created(tmp_path, monkeypatch):
    paths = []

    def as_tensor(w, **kwargs):
        paths.append(Path(kwargs["cache_file_name"]))
        return w

    fake = SimpleNamespace(
        bfloat4_b="bfp4",
        bfloat8_b="bfp8",
        TILE_LAYOUT="tile",
        as_tensor=as_tensor,
        to_torch=lambda t: t,
    )
    monkeypatch.setitem(sys.modules, "ttnn", fake)
    tool = Path(__file__).resolve().parents[1] / "tools" / "pack_tensix.py"
    spec = importlib.util.spec_from_file_location("pack_tensix_test", tool)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
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
    monkeypatch.setattr(sys, "argv", [str(tool), str(tmp_path), "--cache-dir", str(cache)])
    assert module.main() == 0
    first = paths.copy()
    assert cache.is_dir() and len(first) == len(set(first)) == 4
    assert all(p.parent == cache and re.fullmatch(r"[0-9a-f]{64}\.bfp4", p.name) for p in first)
    paths.clear()
    assert module.main() == 0
    assert paths == first
