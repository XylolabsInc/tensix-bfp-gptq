# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Xylolabs
"""This package's rounding against TT-NN's own packer (skipped without ttnn): for bfloat4_b and
bfloat8_b, ttnn.as_tensor rounds a bf16 matrix exactly as bfp() does, values already on the grid
(plain or exponent-searched) pack unchanged, bfloat4_b values pack unchanged as bfloat8_b, a round
trip through a device's DRAM is exact (skipped when no device opens), and tensix_bfp.packed writes
the tile bytes ttnn writes.

Also runs as a script (python tests/test_ttnn_pack.py), for an environment without pytest."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tensix_bfp.bfp import EXP_OFFSETS, bfp, bfp_search, bfp_weight  # noqa: E402
from tensix_bfp.packed import decode, encode  # noqa: E402

try:
    import ttnn
except ImportError:  # pragma: no cover - depends on the environment
    ttnn = None


def no_chip():
    """Why ttnn cannot pack here (it needs a card or the simulator even on the host), or None."""
    try:
        ttnn.as_tensor(
            torch.zeros(32, 32, dtype=torch.bfloat16), ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT
        )
    except Exception as e:  # TT_FATAL without a card or the simulator
        return str(e).splitlines()[0] if str(e) else "no chip"
    return None


if __name__ != "__main__":
    import pytest

    if ttnn is None:
        pytest.skip("ttnn is not installed", allow_module_level=True)
    _why = no_chip()
    if _why:
        pytest.skip(
            "ttnn sees no Tenstorrent card or simulator: %s" % _why, allow_module_level=True
        )

FORMATS = (("bfp4", 3), ("bfp8", 7))


def dtype(fmt):
    return {"bfp4": ttnn.bfloat4_b, "bfp8": ttnn.bfloat8_b}[fmt]


def sample(seed=0):
    """A bf16 [in, out] matrix with scales from 1e-3 to 10 across its columns."""
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(256, 512, generator=g) * torch.logspace(-3, 1, 512)).to(torch.bfloat16)


def packed(x, dt, device=None):
    """x packed by TT-NN in tile layout (to a device's DRAM if given) and read back as float32."""
    kw = {"device": device, "memory_config": ttnn.DRAM_MEMORY_CONFIG} if device else {}
    t = ttnn.as_tensor(x.to(torch.bfloat16), dt, layout=ttnn.TILE_LAYOUT, **kw)
    return ttnn.to_torch(t).float()


def test_rounding_equals_ttnn():
    for fmt, bits in FORMATS:
        w = sample()
        back = packed(w, dtype(fmt))
        assert torch.equal(back, bfp(w, bits)), fmt
        # the same in PyTorch's [out, in] weight layout
        assert torch.equal(back.T, bfp_weight(w.T, bits)), fmt
        assert not torch.equal(back, w.float()), fmt


def test_grid_values_pack_unchanged():
    for fmt, bits in FORMATS:
        w = sample(1)
        for q in (bfp(w, bits), bfp_search(w, bits), bfp_search(w, bits, EXP_OFFSETS)):
            assert torch.equal(packed(q, dtype(fmt)), q), fmt
    q4 = bfp_search(sample(2), 3, EXP_OFFSETS)
    assert torch.equal(packed(q4, ttnn.bfloat8_b), q4)


def test_packed_tiles_are_ttnn_bytes():
    """tensix_bfp.packed writes the bytes stock ttnn keeps for a tile-layout tensor: they appear
    whole in TT-NN's weight cache file, and decode to the values ttnn.to_torch returns."""
    for fmt, _ in FORMATS:
        w = sample(4)
        w[0, :20] = 0.0  # an all-zero face row
        w[1, 3] = -0.0
        ours = encode(w, fmt, exact=False)
        with tempfile.TemporaryDirectory() as tmp:
            t = ttnn.as_tensor(
                w, dtype(fmt), layout=ttnn.TILE_LAYOUT, cache_file_name=str(Path(tmp) / "w")
            )
            files = sorted(Path(tmp).iterdir())
            assert len(files) == 1, f"{fmt}: cache files {files}"
            blob = files[0].read_bytes()
        assert blob.find(bytes(ours.flatten().tolist())) >= 0, fmt
        back = ttnn.to_torch(t).float()
        assert torch.equal(decode(ours, fmt, tuple(w.shape)).float(), back), fmt


class Skipped(Exception):
    pass


def test_dram_round_trip():
    try:
        dev = ttnn.open_device(device_id=0)
    except Exception as e:  # no device or simulator
        if __name__ == "__main__":
            raise Skipped(f"no device: {e}")
        import pytest

        pytest.skip(f"no device: {e}")
    try:
        for fmt, bits in FORMATS:
            q = bfp(sample(3), bits)
            assert torch.equal(packed(q, dtype(fmt), dev), q), fmt
    finally:
        ttnn.close_device(dev)


def test_checker_covers_numbered_experts_and_quantized_head():
    tool = Path(__file__).resolve().parents[1] / "tools/pack_tensix.py"
    for fmt, valid, bad in [("bfp4", 1.25, 1.125), ("bfp8", 1.015625, 1.0078125)]:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            head_name = "model.embed_tokens.weight"
            tensors = {
                f"model.layers.0.mlp.experts.{i}.{projection}.weight": torch.ones(
                    32, 32, dtype=torch.bfloat16
                )
                for i in range(2)
                for projection in ("gate_proj", "up_proj")
            }
            tensors[head_name] = torch.full((32, 32), valid, dtype=torch.bfloat16)
            manifest = {
                "layers": {"0": {"mlp.experts.gate_up_proj": {"format": "bfp4"}}},
                "lm_head": fmt,
                "lm_head_tensor": head_name,
            }
            (directory / "quant4.json").write_text(json.dumps(manifest))
            (directory / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {key: "weights.safetensors" for key in tensors}})
            )
            for broken in (False, True):
                if broken:
                    tensors[head_name][0, 0] = bad
                save_file(tensors, str(directory / "weights.safetensors"))
                result = subprocess.run(
                    [sys.executable, "-I", str(tool), str(directory), "--layers", "0"],
                    capture_output=True,
                    text=True,
                )
                assert result.returncode == int(broken), result.stdout + result.stderr
                assert "5 matrices packed with TT-NN" in result.stdout
                if broken:
                    assert f"MISMATCH {head_name}" in result.stdout


def test_checker_covers_bare_fused_experts():
    tool = Path(__file__).resolve().parents[1] / "tools/pack_tensix.py"
    for fmt, valid, bad in [("bfp4", 1.25, 1.125), ("bfp8", 1.015625, 1.0078125)]:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            key = "model.layers.0.mlp.experts.gate_up_proj"
            tensors = {key: torch.full((2, 32, 32), valid, dtype=torch.bfloat16)}
            (directory / "quant4.json").write_text(
                json.dumps({"layers": {"0": {"mlp.experts.gate_up_proj": {"format": fmt}}}})
            )
            (directory / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {key: "weights.safetensors"}})
            )
            for broken in (False, True):
                if broken:
                    tensors[key][1, 0, 0] = bad
                save_file(tensors, str(directory / "weights.safetensors"))
                result = subprocess.run(
                    [sys.executable, "-I", str(tool), str(directory)],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                assert result.returncode == int(broken), result.stdout + result.stderr
                assert "2 matrices packed with TT-NN" in result.stdout
                if broken:
                    assert f"MISMATCH {key}.1" in result.stdout


def test_checker_covers_global_linears():
    tool = Path(__file__).resolve().parents[1] / "tools/pack_tensix.py"
    name = "model.per_layer_model_projection.weight"
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        tensors = {
            "model.layers.0.mlp.up_proj.weight": torch.ones(32, 32, dtype=torch.bfloat16),
            name: torch.full((32, 64), 1.25, dtype=torch.bfloat16),
        }
        manifest = {
            "layers": {"0": {"mlp.up_proj": {"format": "bfp4"}}},
            "lm_head": "bf16",
            "global_linears": {name: {"format": "bfp4"}},
        }
        (directory / "quant4.json").write_text(json.dumps(manifest))
        (directory / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {key: "weights.safetensors" for key in tensors}})
        )
        for broken in (False, True):
            if broken:
                tensors[name][0, 0] = 1.125
            save_file(tensors, str(directory / "weights.safetensors"))
            result = subprocess.run(
                [sys.executable, "-I", str(tool), str(directory)],
                capture_output=True,
                text=True,
                check=False,
            )
            assert result.returncode == int(broken), result.stdout + result.stderr
            assert "2 matrices packed with TT-NN" in result.stdout
            if broken:
                assert f"MISMATCH {name}" in result.stdout


def test_checker_reads_a_cache_style_snapshot():
    """The shard files of a Hugging Face cache snapshot are links to ../../blobs/<hash>."""
    tool = Path(__file__).resolve().parents[1] / "tools/pack_tensix.py"
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp) / "models--m"
        snapshot = base / "snapshots" / "rev"
        snapshot.mkdir(parents=True)
        (base / "blobs").mkdir()
        tensors = {
            "model.layers.0.mlp.up_proj.weight": torch.full((32, 32), 1.25, dtype=torch.bfloat16),
            "model.layers.0.mlp.down_proj.weight": torch.full((32, 32), 1.0, dtype=torch.bfloat16),
        }
        manifest = {
            "layers": {
                "0": {"mlp.up_proj": {"format": "bfp4"}, "mlp.down_proj": {"format": "bfp8"}}
            },
            "lm_head": "bf16",
        }
        weight_map = {key: "weights.safetensors" for key in tensors}
        save_file(tensors, str(base / "blobs" / "0123abcd"))
        (snapshot / "weights.safetensors").symlink_to(Path("..") / ".." / "blobs" / "0123abcd")
        (snapshot / "quant4.json").write_text(json.dumps(manifest))
        (snapshot / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": weight_map})
        )
        result = subprocess.run(
            [sys.executable, "-I", str(tool), str(snapshot)], capture_output=True, text=True
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "2 matrices packed with TT-NN" in result.stdout


def test_checker_covers_extra_matrices_and_refuses_conflicts():
    tool = Path(__file__).resolve().parents[1] / "tools/pack_tensix.py"
    name = "model.extra_matrix.weight"
    head = "lm_head.weight"
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        tensors = {
            "model.layers.0.mlp.up_proj.weight": torch.ones(32, 32, dtype=torch.bfloat16),
            head: torch.full((32, 32), 1.015625, dtype=torch.bfloat16),
            name: torch.full((32, 64), 1.25, dtype=torch.bfloat16),
        }
        manifest = {
            "layers": {"0": {"mlp.up_proj": {"format": "bfp4"}}},
            "lm_head": "bfp8",
            "extra": {name: {"format": "bfp4"}, head: {"format": "bfp8"}},
        }
        (directory / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {key: "weights.safetensors" for key in tensors}})
        )
        save_file(tensors, str(directory / "weights.safetensors"))

        def run():
            (directory / "quant4.json").write_text(json.dumps(manifest))
            return subprocess.run(
                [sys.executable, "-I", str(tool), str(directory)],
                capture_output=True,
                text=True,
                check=False,
            )

        result = run()  # the head declared twice with one format counts once
        assert result.returncode == 0, result.stdout + result.stderr
        assert "3 matrices packed with TT-NN" in result.stdout
        tensors[name][0, 0] = 1.125  # off the bfp4 grid
        save_file(tensors, str(directory / "weights.safetensors"))
        result = run()
        assert result.returncode == 1 and f"MISMATCH {name}" in result.stdout
        manifest["extra"][head] = {"format": "bfp4"}  # on the bfp8 grid only
        result = run()
        assert result.returncode == 1 and "matrices packed" not in result.stdout
        assert f"{head!r} is declared as bfp8 and as bfp4" in result.stderr


if __name__ == "__main__":
    if ttnn is None:
        print("ttnn is not installed: skipped")
        sys.exit(0)
    why = no_chip()
    if why:
        print(f"ttnn sees no Tenstorrent card or simulator: skipped ({why})")
        sys.exit(0)
    failed = 0
    for name, fn in [(k, v) for k, v in globals().items() if k.startswith("test_")]:
        try:
            fn()
            print(f"{name}: passed", flush=True)
        except Skipped as e:
            print(f"{name}: skipped ({e})", flush=True)
        except AssertionError as e:
            failed += 1
            print(f"{name}: FAILED {e}", flush=True)
    sys.exit(1 if failed else 0)
