# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Xylolabs
"""This package's rounding against TT-NN's own packer (skipped without ttnn): for bfloat4_b and
bfloat8_b, ttnn.as_tensor rounds a bf16 matrix exactly as bfp() does, values already on the grid
(plain or exponent-searched) pack unchanged, bfloat4_b values pack unchanged as bfloat8_b, and a
round trip through a device's DRAM is exact (skipped when no device opens).

Also runs as a script (python tests/test_ttnn_pack.py), for an environment without pytest."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tensix_bfp.bfp import EXP_OFFSETS, bfp, bfp_search, bfp_weight  # noqa: E402

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
