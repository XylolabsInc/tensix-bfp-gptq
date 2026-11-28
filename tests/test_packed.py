# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Xylolabs
"""The packed checkpoint format: tile byte layout, bit-exact round trips, the checks, and the
pack, verify and unpack commands (torch and safetensors only)."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from tensix_bfp import packed
from tensix_bfp.bfp import bfp, bfp_weight

ROOT = Path(__file__).resolve().parents[1]
FORMATS = (("bfp4", 3), ("bfp8", 7))


def bits(t):
    """A bf16 tensor's bit patterns, so that -0.0 and +0.0 differ."""
    return t.to(torch.bfloat16).view(torch.int16)


def plus_zero(t):
    """t as bf16 with every -0.0 made +0.0."""
    return t.to(torch.bfloat16) + 0.0


def on_grid(shape, mant_bits, seed=0):
    """A bf16 PyTorch [out, in] weight on its grid (blocks along out), with -0.0 entries and
    scales from 1e-4 to 10 across its inputs."""
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(shape, generator=g) * torch.logspace(-4, 1, shape[-1])
    w[..., 0, :] = -1e-12  # tiny negatives in blocks with larger values: stored as -0.0
    q = bfp_weight(w.reshape(-1, shape[-1]), mant_bits).reshape(shape).to(torch.bfloat16)
    assert (bits(q) == -32768).any()  # there are -0.0 values to normalize
    return q


@pytest.mark.parametrize("fmt,k", FORMATS)
def test_round_trip_is_bit_exact(fmt, k):
    w = on_grid((64, 96), k)
    tiles, rec = packed.pack_tensor(w, fmt)
    assert tiles.dtype == torch.uint8
    assert list(tiles.shape) == [3 * 2, packed.TILE_BYTES[fmt]]
    assert rec["stored_shape"] == [64, 96] and rec["packed_shape"] == [96, 64]
    assert rec["tiles"] == [3, 2] and rec["bytes"] == tiles.numel()
    out = packed.unpack_tensor(tiles, rec)
    assert torch.equal(bits(out), bits(plus_zero(w)))
    assert rec["bf16_sha256"] == packed.sha256(plus_zero(w))
    assert packed.check_tensor(tiles, rec) == []


@pytest.mark.parametrize("fmt,k", FORMATS)
def test_bytes_follow_the_tile_layout(fmt, k):
    m = torch.zeros(32, 64)
    m[0, 1] = 1.5  # tile 0, face 0, row 0, column 1
    m[17, 18] = -0.25  # tile 0, face 3 (bottom-right), row 1, column 2
    m[5, 40] = 3.0  # tile 1, face 0, row 5, column 8
    t = packed.encode(m, fmt)
    assert list(t.shape) == [2, packed.TILE_BYTES[fmt]]
    t0, t1 = t[0].tolist(), t[1].tolist()
    # exponents: byte 16 f + r, biased by 127; all-zero rows get 0
    assert t0[0] == 127 and t0[16 * 3 + 1] == 125 and t1[5] == 128
    assert sum(1 for x in t0[:64] if x) == 2 and sum(1 for x in t1[:64] if x) == 1
    one = 1 << (k - 1)  # the magnitude of 1.0 * 2^(e - 127)
    if fmt == "bfp4":
        # two values a byte, the first in the low nibble; bit 3 the sign
        assert t0[64 + 0] == (one * 3 // 2) << 4  # value 1 of the row: the high nibble
        assert t0[64 + (3 * 256 + 16 + 2) // 2] == 0x8 | one  # value 786: a low nibble
        assert t1[64 + (5 * 16 + 8) // 2] == one * 3 // 2
    else:
        assert t0[64 + 1] == one * 3 // 2
        assert t0[64 + 3 * 256 + 16 + 2] == 0x80 | one
        assert t1[64 + 5 * 16 + 8] == one * 3 // 2
    assert sum(1 for x in t0[64:] if x) == 2 and sum(1 for x in t1[64:] if x) == 1
    assert torch.equal(packed.decode(t, fmt, (32, 64)).float(), m)


@pytest.mark.parametrize("fmt,k", FORMATS)
def test_zeros_are_stored_with_sign_zero(fmt, k):
    m = torch.zeros(32, 32, dtype=torch.bfloat16)
    m[0, 0] = -0.0
    m[3, :16] = torch.tensor([-0.0] * 15 + [2.0])
    t = packed.encode(m, fmt)
    _, codes = packed._split(t, fmt)
    assert not ((codes >> k) == 1).any()
    assert torch.equal(bits(packed.decode(t, fmt, (32, 32))), bits(plus_zero(m)))
    # a zero stored with sign 1 decodes to +0.0, and the check reports it
    rec = packed.pack_tensor(m, fmt, "as_stored")[1]
    bad = t.clone()
    if fmt == "bfp4":
        bad[0, 64] |= 0x8
    else:
        bad[0, 64] |= 0x80
    assert torch.equal(bits(packed.decode(bad, fmt, (32, 32))), bits(plus_zero(m)))
    assert "1 zeros stored with sign 1" in packed.check_tensor(bad, rec)


@pytest.mark.parametrize("fmt,k", FORMATS)
def test_off_grid_values_are_refused_or_rounded_as_ttnn(fmt, k):
    g = torch.Generator().manual_seed(1)
    x = (torch.randn(64, 64, generator=g) * torch.logspace(-3, 1, 64)).to(torch.bfloat16)
    with pytest.raises(ValueError, match="not on the"):
        packed.pack_tensor(x, fmt)
    tiles, rec = packed.pack_tensor(x, fmt, at_pack=True)
    assert rec["quantized_at_pack"] == "ttnn" and rec["source_sha256"] == packed.sha256(x)
    # the rounding is bfp()'s, which tests/test_ttnn_pack.py checks against ttnn
    assert torch.equal(bits(packed.unpack_tensor(tiles, rec)), bits(plus_zero(bfp_weight(x, k))))
    assert packed.check_tensor(tiles, rec) == []


@pytest.mark.parametrize("fmt,k", FORMATS)
def test_wide_exponent_range(fmt, k):
    g = torch.Generator().manual_seed(2)
    scale = torch.ldexp(torch.ones(128, 1), torch.randint(-100, 100, (128, 1)).float())
    q = bfp(torch.randn(128, 64, generator=g) * scale, k).to(torch.bfloat16)
    t = packed.encode(q, fmt)
    assert torch.equal(bits(packed.decode(t, fmt, (128, 64))), bits(plus_zero(q)))


@pytest.mark.parametrize("orientation", packed.ORIENTATIONS)
def test_experts_and_orientations(orientation):
    # multiples of 1/8 up to 7/8 lie on the bfp4 grid along either axis
    g = torch.Generator().manual_seed(7)
    w = (torch.randint(-7, 8, (3, 64, 32), generator=g) / 8).to(torch.bfloat16)
    w[0, 0, 0] = -0.0
    tiles, rec = packed.pack_tensor(w, "bfp4", orientation)
    assert list(tiles.shape) == [3, 2, 576]
    want = [32, 64] if orientation == "transposed" else [64, 32]
    assert rec["packed_shape"] == want and rec["stored_shape"] == [3, 64, 32]
    assert torch.equal(bits(packed.unpack_tensor(tiles, rec)), bits(plus_zero(w)))
    assert packed.check_tensor(tiles, rec) == []


def test_bad_records_are_refused():
    tiles, rec = packed.pack_tensor(on_grid((64, 32), 3), "bfp4")
    for key, value in [
        ("format", "bfp2"),
        ("orientation", "rotated"),
        ("packed_shape", [64, 32]),
        ("tiles", [2, 1]),
        ("bytes", 1),
        ("bf16_sha256", "0"),
    ]:
        with pytest.raises(ValueError):
            packed.unpack_tensor(tiles, {**rec, key: value})
    with pytest.raises(ValueError):
        packed.unpack_tensor(tiles[:1], rec)
    with pytest.raises(ValueError, match="multiples of 32"):
        packed.encode(torch.zeros(32, 48), "bfp4")


def export_dir(directory):
    """A small bf16-grid export in two shards, with quant4.json and its index."""
    head = on_grid((96, 64), 7, seed=3)
    tensors = {
        "a.safetensors": {
            "model.layers.0.self_attn.q_proj.weight": on_grid((64, 64), 3, seed=4),
            "model.layers.0.mlp.experts.gate_up_proj": on_grid((2, 64, 32), 3, seed=5),
            "model.layers.0.mlp.down_proj.weight": on_grid((48, 64), 7, seed=6),
            "model.norm.weight": torch.ones(64),
        },
        "b.safetensors": {
            "lm_head.weight": head,
            "mtp.layers.0.proj.weight": torch.randn(64, 64).to(torch.bfloat16),
            "model.embed_tokens.weight": torch.randn(96, 64).to(torch.bfloat16),
        },
    }
    weight_map = {}
    for f, ts in tensors.items():
        save_file(ts, str(directory / f), metadata={"format": "pt"})
        weight_map.update(dict.fromkeys(ts, f))
    (directory / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"quant4": "test"}, "weight_map": weight_map})
    )
    manifest = {
        "layers": {
            "0": {
                "self_attn.q_proj": {"format": "bfp4"},
                "mlp.experts.gate_up_proj": {"format": "bfp4"},
                "mlp.down_proj": {"format": "bfp8"},
            }
        },
        "lm_head": "bfp8",
        "lm_head_tensor": "lm_head.weight",
    }
    (directory / "quant4.json").write_text(json.dumps(manifest))
    (directory / "config.json").write_text("{}")
    return {k: v for ts in tensors.values() for k, v in ts.items()}


def test_checkpoint_pack_verify_load_unpack(tmp_path):
    src, dst, back = tmp_path / "bf16", tmp_path / "packed", tmp_path / "unpacked"
    src.mkdir()
    original = export_dir(src)
    records, kept = packed.pack(src, dst, [(r"^mtp\..*\.weight$", "bfp4")])
    assert kept == ["model.layers.0.mlp.down_proj.weight"]  # 48 rows: not tile-aligned
    assert set(records) == {
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.mlp.experts.gate_up_proj",
        "lm_head.weight",
        "mtp.layers.0.proj.weight",
    }
    assert records["mtp.layers.0.proj.weight"]["quantized_at_pack"] == "ttnn"
    assert "quantized_at_pack" not in records["lm_head.weight"]
    names = sorted(p.name for p in dst.iterdir())
    assert names == [
        "config.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "model.safetensors.index.json",
        "quant4.json",
        "tensix_pack.json",
    ]
    manifest = json.loads((dst / "tensix_pack.json").read_text())
    assert manifest["format"] == "tensix-bfp-tiles" and manifest["format_version"] == 1
    assert manifest["tile"] == [32, 32] and manifest["face"] == [16, 16]
    assert packed.verify(dst) == ([], 4)

    loaded = packed.load_bf16(dst)
    assert set(loaded) == set(original)
    for name, t in original.items():
        if name == "mtp.layers.0.proj.weight":
            want = plus_zero(bfp_weight(t, 3))
        elif name in records:
            want = plus_zero(t)
        else:
            want = t
        assert loaded[name].dtype == want.dtype, name
        assert torch.equal(loaded[name].view(torch.uint8), want.view(torch.uint8)), name

    packed.unpack(dst, back)
    assert not (back / "tensix_pack.json").exists()
    unpacked = {}
    for f in sorted(back.glob("*.safetensors")):
        unpacked.update(load_file(str(f)))
        with safe_open(str(f), framework="pt") as g:
            assert g.metadata()["format"] == "pt"
    assert set(unpacked) == set(loaded)
    assert all(
        torch.equal(unpacked[k].view(torch.uint8), v.view(torch.uint8)) for k, v in loaded.items()
    )
    result = subprocess.run(
        [sys.executable, "-I", str(ROOT / "tools/verify_grid.py"), str(back)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_verify_finds_tampering(tmp_path):
    src, dst = tmp_path / "bf16", tmp_path / "packed"
    src.mkdir()
    export_dir(src)
    packed.pack(src, dst)
    shard = dst / "model-00001-of-00002.safetensors"
    tensors = load_file(str(shard))
    name = "model.layers.0.self_attn.q_proj.weight"
    tensors[name][0, 64] ^= 0x1  # one magnitude bit of one value
    save_file(tensors, str(shard), metadata={"format": "pt"})
    problems, checked = packed.verify(dst)
    assert checked == 3
    assert f"{name}: decoded values do not match bf16_sha256" in problems
    with pytest.raises(ValueError, match="bf16_sha256"):
        packed.load_bf16(dst)
    assert name in packed.load_bf16(dst, check=False)
    manifest = json.loads((dst / "tensix_pack.json").read_text())
    manifest["format_version"] = 2
    (dst / "tensix_pack.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="format_version"):
        packed.load_bf16(dst)


def test_runs_standalone(tmp_path):
    """packed.py also works copied alone next to a checkpoint (as the model repos ship it)."""
    src, dst, back = tmp_path / "bf16", tmp_path / "packed", tmp_path / "unpacked"
    src.mkdir()
    export_dir(src)
    lone = tmp_path / "lone"
    lone.mkdir()
    shutil.copy(ROOT / "tensix_bfp/packed.py", lone / "packed.py")

    def run(*args):
        return subprocess.run(
            [sys.executable, "-I", str(lone / "packed.py"), *map(str, args)],
            capture_output=True,
            text=True,
            check=False,
        )

    result = run("pack", src, dst)
    assert result.returncode == 0, result.stderr
    assert "3 tensors packed, 1 kept as stored" in result.stdout
    result = run("verify", dst)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "3 packed tensors checked, 0 problems" in result.stdout
    assert run("unpack", dst, back).returncode == 0
    assert (back / "model-00002-of-00002.safetensors").exists()
    assert run("verify", src).returncode == 1  # no tensix_pack.json


def test_indexed_packed_tensor_missing_from_shard(tmp_path):
    d = tmp_path / "packed"
    d.mkdir()
    data, rec = packed.pack_tensor(torch.ones(32, 32, dtype=torch.bfloat16), "bfp4")
    save_file({"present": data}, str(d / "weights.safetensors"))
    (d / packed.INDEX).write_text(
        json.dumps(
            {"weight_map": {"present": "weights.safetensors", "absent": "weights.safetensors"}}
        )
    )
    (d / packed.PACK_JSON).write_text(
        json.dumps(
            {
                "format": packed.FORMAT,
                "format_version": packed.FORMAT_VERSION,
                "tile": [32, 32],
                "face": [16, 16],
                "tensors": {"present": rec, "absent": rec},
            }
        )
    )
    problems, checked = packed.verify(d)
    assert checked == 1 and any("absent" in p for p in problems)
    with pytest.raises(ValueError, match="absent"):
        list(packed.iter_bf16(d))
    with pytest.raises(ValueError, match="absent"):
        packed.unpack(d, tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("duplicate", [False, True])
def test_indexed_shard_ownership(tmp_path, duplicate):
    data, rec = packed.pack_tensor(torch.ones(32, 32, dtype=torch.bfloat16), "bfp4")
    other, _ = packed.pack_tensor(torch.full((32, 32), 0.5, dtype=torch.bfloat16), "bfp4")
    save_file({"x": data} if duplicate else {"a": torch.ones(1)}, str(tmp_path / "a.safetensors"))
    save_file({"x": other, "b": torch.ones(1)}, str(tmp_path / "b.safetensors"))
    (tmp_path / packed.INDEX).write_text(
        json.dumps({"weight_map": {"x": "a.safetensors", "b": "b.safetensors"}})
    )
    (tmp_path / packed.PACK_JSON).write_text(
        json.dumps(
            {
                "format": packed.FORMAT,
                "format_version": 1,
                "tile": [32, 32],
                "face": [16, 16],
                "tensors": {"x": rec},
            }
        )
    )
    message = (
        "duplicate tensor in a.safetensors and b.safetensors"
        if duplicate
        else "missing from its indexed shard"
    )
    assert any(message in p for p in packed.verify(tmp_path)[0])
    with pytest.raises(ValueError, match=message):
        packed.load_bf16(tmp_path)
    with pytest.raises(ValueError, match=message):
        packed.unpack(tmp_path, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_non_object_manifest_reports_error(tmp_path):
    (tmp_path / packed.PACK_JSON).write_text("[]")
    problems, checked = packed.verify(tmp_path)
    assert checked == 0 and "object" in problems[0]


@pytest.mark.parametrize("wrong_shard", [False, True])
def test_missing_indexed_bf16_is_rejected(tmp_path, wrong_shard):
    src, dst = tmp_path / "src", tmp_path / "packed"
    src.mkdir()
    export_dir(src)
    name = "model.embed_tokens.weight"
    b = src / "b.safetensors"
    tensors = load_file(str(b))
    value = tensors.pop(name)
    save_file(tensors, str(b))
    if wrong_shard:
        a = src / "a.safetensors"
        tensors = load_file(str(a))
        tensors[name] = value
        save_file(tensors, str(a))
    with pytest.raises(ValueError, match=name):
        packed.pack(src, dst, [("embed_tokens", "bfp4")])
    assert not dst.exists()

    # Make a valid packed checkpoint and then corrupt its unquantized index entry.
    tensors = load_file(str(b))
    tensors[name] = value
    save_file(tensors, str(b))
    if wrong_shard:
        a = src / "a.safetensors"
        tensors = load_file(str(a))
        del tensors[name]
        save_file(tensors, str(a))
    packed.pack(src, dst)
    b = dst / "model-00002-of-00002.safetensors"
    tensors = load_file(str(b))
    del tensors[name]
    save_file(tensors, str(b))
    if wrong_shard:
        a = dst / "model-00001-of-00002.safetensors"
        tensors = load_file(str(a))
        tensors[name] = value
        save_file(tensors, str(a))
    assert any(name in p for p in packed.verify(dst)[0])
    with pytest.raises(ValueError, match=name):
        list(packed.iter_bf16(dst))
    with pytest.raises(ValueError, match=name):
        packed.load_bf16(dst)
    with pytest.raises(ValueError, match=name):
        packed.unpack(dst, tmp_path / "unpacked")
