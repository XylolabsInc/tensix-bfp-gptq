# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Xylolabs
"""Shard names of an index file are untrusted: a name that leaves its directory is refused by every
tool before anything is read or written outside it, and a cache-style snapshot whose shard files
are symbolic links still reads."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from tensix_bfp import packed

ROOT = Path(__file__).resolve().parents[1]
INDEX = "model.safetensors.index.json"
HAS_TTNN = importlib.util.find_spec("ttnn") is not None
BAD = ["../outside.safetensors", "ABSOLUTE", "sub/x.safetensors", "a\\b.safetensors", "..", "x.bin"]


def ones(*shape):
    return torch.ones(*shape, dtype=torch.bfloat16)


def export(d):
    """A two-shard bf16-grid export (layer 0 q_proj bfp4, LM head bfp8) with quant4.json."""
    d.mkdir(parents=True)
    shards = {
        "a.safetensors": {
            "model.layers.0.self_attn.q_proj.weight": ones(32, 32),
            "model.norm.weight": ones(32),
        },
        "b.safetensors": {"lm_head.weight": ones(32, 32)},
    }
    weight_map = {}
    for f, ts in shards.items():
        save_file(ts, str(d / f), metadata={"format": "pt"})
        weight_map.update(dict.fromkeys(ts, f))
    (d / INDEX).write_text(json.dumps({"weight_map": weight_map}))
    manifest = {
        "layers": {"0": {"self_attn.q_proj": {"format": "bfp4"}}},
        "lm_head": "bfp8",
        "lm_head_tensor": "lm_head.weight",
    }
    (d / "quant4.json").write_text(json.dumps(manifest))
    return d


def packed_dir(root):
    export(root / "src")
    packed.pack(root / "src", root / "packed")
    return root / "packed"


def retarget(d, name, outside):
    """Point the last shard's tensors at `name`, and put a real shard file at `outside`."""
    index = json.loads((d / INDEX).read_text())
    last = sorted(set(index["weight_map"].values()))[-1]
    outside.write_bytes((d / last).read_bytes())
    for k, v in index["weight_map"].items():
        if v == last:
            index["weight_map"][k] = name
    (d / INDEX).write_text(json.dumps(index))


def snapshot(root):
    return {
        str(p.relative_to(root)): p.read_bytes() if p.is_file() else None
        for p in sorted(root.rglob("*"))
    }


def bad_name(name, outside):
    return str(outside) if name == "ABSOLUTE" else name


def run_tool(tool, d):
    return subprocess.run(
        [sys.executable, "-I", str(ROOT / "tools" / tool), str(d)], capture_output=True, text=True
    )


@pytest.mark.parametrize("name", BAD)
def test_verify_refuses(tmp_path, name):
    w = tmp_path / "w"
    w.mkdir()
    d = packed_dir(w)
    name = bad_name(name, w / "outside.safetensors")
    retarget(d, name, w / "outside.safetensors")
    before = snapshot(tmp_path)
    problems, checked = packed.verify(d)
    assert checked == 0 and len(problems) == 1 and repr(name) in problems[0]
    assert INDEX in problems[0]
    assert snapshot(tmp_path) == before
    with pytest.raises(ValueError, match="shard name"):
        packed.load_bf16(d)
    assert packed.main(["verify", str(d)]) == 1


@pytest.mark.parametrize("name", BAD)
def test_unpack_refuses_and_writes_nothing(tmp_path, name, capsys):
    w = tmp_path / "w"
    w.mkdir()
    d = packed_dir(w)
    name = bad_name(name, w / "outside.safetensors")
    retarget(d, name, w / "outside.safetensors")
    before = snapshot(tmp_path)
    with pytest.raises(ValueError, match="shard name"):
        packed.unpack(d, tmp_path / "out")
    assert snapshot(tmp_path) == before
    assert packed.main(["unpack", str(d), str(tmp_path / "out")]) == 1
    assert repr(name) in capsys.readouterr().err
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("name", BAD)
def test_pack_refuses_and_writes_nothing(tmp_path, name, capsys):
    w = tmp_path / "w"
    w.mkdir()
    d = export(w / "src")
    name = bad_name(name, w / "outside.safetensors")
    retarget(d, name, w / "outside.safetensors")
    before = snapshot(tmp_path)
    with pytest.raises(ValueError, match="shard name"):
        packed.pack(d, tmp_path / "out")
    assert snapshot(tmp_path) == before
    assert packed.main(["pack", str(d), str(tmp_path / "out")]) == 1
    assert repr(name) in capsys.readouterr().err
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("name", BAD)
def test_verify_grid_refuses(tmp_path, name):
    w = tmp_path / "w"
    w.mkdir()
    d = export(w / "src")
    name = bad_name(name, w / "outside.safetensors")
    retarget(d, name, w / "outside.safetensors")
    before = snapshot(tmp_path)
    result = run_tool("verify_grid.py", d)
    assert result.returncode == 1 and repr(name) in result.stderr
    assert "Traceback" not in result.stderr and INDEX in result.stderr
    assert snapshot(tmp_path) == before


@pytest.mark.skipif(not HAS_TTNN, reason="ttnn is not installed")
@pytest.mark.parametrize("name", BAD)
def test_pack_tensix_refuses(tmp_path, name):
    w = tmp_path / "w"
    w.mkdir()
    d = export(w / "src")
    name = bad_name(name, w / "outside.safetensors")
    retarget(d, name, w / "outside.safetensors")
    before = snapshot(tmp_path)
    result = run_tool("pack_tensix.py", d)
    assert result.returncode == 1 and repr(name) in result.stderr
    assert "Traceback" not in result.stderr and INDEX in result.stderr
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("name", ["C:x.safetensors", ".", "a\0b.safetensors", ""])
def test_more_invalid_names(tmp_path, name):
    d = export(tmp_path / "src")
    index = json.loads((d / INDEX).read_text())
    index["weight_map"]["lm_head.weight"] = name
    (d / INDEX).write_text(json.dumps(index))
    with pytest.raises(ValueError, match="shard name"):
        packed.shards(d)


@pytest.mark.parametrize("index", [{}, {"weight_map": []}, {"weight_map": {"a": 1}}, []])
def test_malformed_index_is_a_value_error(tmp_path, index):
    d = export(tmp_path / "src")
    (d / INDEX).write_text(json.dumps(index))
    with pytest.raises(ValueError, match=INDEX):
        packed.shards(d)
    with pytest.raises(ValueError, match=INDEX):
        packed.pack(d, tmp_path / "out")
    result = run_tool("verify_grid.py", d)
    assert result.returncode == 1 and "Traceback" not in result.stderr
    assert not (tmp_path / "out").exists()


def cache_style(d):
    """Move every shard of d to ../blobs/<hash> and link it back, as the Hugging Face cache does."""
    blobs = d.parent / "blobs"
    blobs.mkdir()
    for i, f in enumerate(sorted(d.glob("*.safetensors"))):
        blob = blobs / f"{i:040x}"
        f.rename(blob)
        f.symlink_to(Path("..") / "blobs" / blob.name)
    return d


def test_cache_style_snapshot_reads(tmp_path):
    base = tmp_path / "models--m"
    (base / "snapshots").mkdir(parents=True)
    d = packed_dir(base / "snapshots")
    d = d.rename(base / "snapshots" / "rev")
    cache_style(d)
    assert all(f.is_symlink() for f in d.glob("*.safetensors"))
    problems, checked = packed.verify(d)
    assert (problems, checked) == ([], 2)
    packed.unpack(d, tmp_path / "out")
    assert len(list((tmp_path / "out").glob("*.safetensors"))) == 2
    assert packed.main(["verify", str(d)]) == 0


def test_cache_style_export_checks(tmp_path):
    (tmp_path / "snapshots").mkdir()
    d = export(tmp_path / "snapshots" / "rev")
    cache_style(d)
    result = run_tool("verify_grid.py", d)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "2 matrices checked, 0 off the grid" in result.stdout
    packed.pack(d, tmp_path / "out")
    assert (tmp_path / "out" / "tensix_pack.json").exists()


def test_output_directory_symlink_is_refused(tmp_path, capsys):
    d = packed_dir(tmp_path)
    real = tmp_path / "real"
    real.mkdir()
    (tmp_path / "link").symlink_to(real)
    with pytest.raises(ValueError, match="symbolic link"):
        packed.unpack(d, tmp_path / "link")
    with pytest.raises(ValueError, match="symbolic link"):
        packed.pack(tmp_path / "src", tmp_path / "link")
    assert list(real.iterdir()) == []
    assert packed.main(["unpack", str(d), str(tmp_path / "link")]) == 1
    assert "symbolic link" in capsys.readouterr().err


@pytest.mark.parametrize("command", ["unpack", "pack"])
def test_symlink_at_an_output_shard_is_refused(tmp_path, command):
    d = packed_dir(tmp_path)
    src = d if command == "unpack" else tmp_path / "src"
    out = tmp_path / "out"
    out.mkdir()
    victim = tmp_path / "victim.bin"
    victim.write_bytes(b"keep me")
    shard = sorted(set(packed.shards(src).values()))[0]
    shard = shard if command == "unpack" else "model-00001-of-00002.safetensors"
    (out / shard).symlink_to(victim)
    with pytest.raises(ValueError, match="symbolic link"):
        getattr(packed, command)(src, out)
    assert victim.read_bytes() == b"keep me"
    assert (out / shard).is_symlink() and os.readlink(out / shard) == str(victim)
    assert sorted(p.name for p in out.iterdir()) == [shard]


@pytest.mark.parametrize("name", ["../x.safetensors", "sub/x.safetensors"])
def test_output_names_must_be_plain(tmp_path, name):
    with pytest.raises(ValueError, match="shard name"):
        packed._out_dir(tmp_path / "out", [name])
    assert not (tmp_path / "out").exists()
