# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Xylolabs
"""The BFP grids: idempotence, the exponent search, bfloat4_b inside bfloat8_b, and directed edge
vectors frozen from TT-NN (torch only; the live TT-NN comparison is skipped without ttnn)."""

import pytest
import torch

from tensix_bfp.bfp import (
    EXP_OFFSETS,
    bfp,
    bfp_search,
    bfp_search_weight,
    bfp_weight,
    on_grid,
    quantize,
)

BITS = {"bfp4": 3, "bfp8": 7}


def sample(rows=96, cols=512, seed=0):
    """bf16 values with scales from 1e-3 to 10 along the last dimension, some exact zeros."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, cols, generator=g) * torch.logspace(-3, 1, cols)
    x[:, ::37] = 0
    return x.to(torch.bfloat16).float()


def block_err(q, x):
    return (q - x).pow(2).reshape(-1, 16).sum(1)


def test_known_block():
    x = torch.tensor([1.9, 1.0, 0.3, 0.1, -0.6, 0.0] + [0.0] * 10)
    # shared exponent of 1.9: step 2^-2 for bfloat4_b; 1.9 -> 7.6 steps, clamped at 7
    want = torch.tensor([1.75, 1.0, 0.25, 0.0, -0.5, 0.0] + [0.0] * 10)
    assert torch.equal(bfp(x, 3), want)


@pytest.mark.parametrize("fmt", ["bfp4", "bfp8"])
def test_idempotent(fmt):
    x = sample()
    q = bfp(x, BITS[fmt])
    assert not torch.equal(q, x)
    assert torch.equal(bfp(q, BITS[fmt]), q)


@pytest.mark.parametrize("fmt", ["bfp4", "bfp8"])
def test_codes_fit_the_mantissa(fmt):
    bits = BITS[fmt]
    x = sample(seed=1)
    q = bfp(x, bits).reshape(-1, 16)
    e = ((x.reshape(-1, 16).view(torch.int32) >> 23) & 0xFF).amax(1, keepdim=True)
    m = q.abs() / torch.ldexp(torch.ones_like(q), (e - 127 - (bits - 1)).float())
    assert torch.equal(m, m.round())
    assert m.max() <= 2**bits - 1


@pytest.mark.parametrize("fmt", ["bfp4", "bfp8"])
@pytest.mark.parametrize("cands", [(0, -1), EXP_OFFSETS])
def test_search_never_worse(fmt, cands):
    bits = BITS[fmt]
    x = sample(seed=2)
    plain, searched = bfp(x, bits), bfp_search(x, bits, cands)
    assert (block_err(searched, x) <= block_err(plain, x)).all()
    assert (block_err(searched, x) < block_err(plain, x)).any()
    assert torch.equal(bfp(searched, bits), searched)


@pytest.mark.parametrize("fmt", ["bfp4", "bfp8"])
def test_weight_layout(fmt):
    w = sample(64, 256, seed=3)
    bits = BITS[fmt]
    assert torch.equal(bfp_weight(w, bits), bfp(w.T.contiguous(), bits).T)
    assert torch.equal(bfp_search_weight(w, bits), bfp_search(w.T.contiguous(), bits).T)
    q = quantize(w, fmt)
    assert on_grid(q, fmt) and not on_grid(w, fmt)


def test_bfp4_values_exact_in_bfp8():
    x = sample(seed=4)
    for q in (bfp(x, 3), bfp_search(x, 3, EXP_OFFSETS)):
        assert torch.equal(bfp(q, 7), q)


def test_bf16():
    x = torch.randn(32, 64)
    q = quantize(x, "bf16")
    assert torch.equal(q, x.to(torch.bfloat16).float()) and on_grid(q, "bf16")


# --- directed edge vectors against stock TT-NN ----------------------------------------------------
#
# Each vector is one 16-value block: (name, input head, expected head); both heads are padded with
# 0.0 to 16 values (a vector whose input is all subnormals is written out in full). The expected
# values come from stock TT-NN on ttsim: ttnn.as_tensor(x, dtype=ttnn.bfloat4_b or
# ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT) and ttnn.to_torch, with float32 inputs (bfloat16 gives
# the same values where the input is exact in bfloat16), taken once and frozen below. They are
# round to nearest even, the mantissa clamped at its top code, and a value whose exponent field is
# 0 (a float32 subnormal such as 2^-140) flushed to 0. Without ttnn the tests run on the literals;
# with ttnn they also check the literals against the installed TT-NN.

TTNN_VERSION = "0.81.1"  # the TT-NN that produced EDGE_VECTORS
SUB = 2.0**-140

EDGE_VECTORS = {
    "bfp4": [
        (
            "ties",
            [1.75, 1.125, 1.375, -1.125, -1.375, 1.625, -1.625, 1.875, 1.0625, 1.1875, 1.25],
            [1.75, 1.0, 1.5, -1.0, -1.5, 1.5, -1.5, 1.75, 1.0, 1.25, 1.25],
        ),
        (
            "ties_negative_max",
            [-1.75, -1.125, -1.375, 1.125, 1.375, -1.0625, -1.1875],
            [-1.75, -1.0, -1.5, 1.0, 1.5, -1.0, -1.25],
        ),
        ("clamp_at_top_code", [1.9375, 1.8125, -1.9375, 1.6875], [1.75, 1.75, -1.75, 1.75]),
        (
            "shared_exponent_below_one",
            [0.875, 0.5625, 0.6875, -0.5625, -0.6875, 0.4375],
            [0.875, 0.5, 0.75, -0.5, -0.75, 0.5],
        ),
        (
            "small_values_lost",
            [8.0, 0.5, 1.0, 3.0, 5.0, -1.0, -3.0, 0.0625],
            [8.0, 0.0, 0.0, 4.0, 4.0, 0.0, -4.0, 0.0],
        ),
        ("subnormal_next_to_normals", [1.75, SUB, -SUB, 1.125, 0.5], [1.75, 0.0, 0.0, 1.0, 0.5]),
        ("only_subnormals", [SUB, -SUB] + [SUB] * 14, [0.0] * 16),
        ("zeros", [0.0], [0.0]),
        ("negative_zero", [-0.0, 0.0, -0.0], [0.0, 0.0, 0.0]),
        ("one_nonzero", [0.0, 0.0, 0.5], [0.0, 0.0, 0.5]),
        ("one_nonzero_negative", [0.0, -3.0], [0.0, -3.0]),
        ("one_nonzero_subnormal", [0.0, SUB], [0.0, 0.0]),
        (
            "signs",
            [1.0, -1.0, 0.75, -0.75, 0.5, -0.5, 0.25, -0.25, 1.5, -1.5, 1.25, -1.25],
            [1.0, -1.0, 0.75, -0.75, 0.5, -0.5, 0.25, -0.25, 1.5, -1.5, 1.25, -1.25],
        ),
    ],
    "bfp8": [
        (
            "ties",
            [
                1.984375,
                1.0078125,
                1.0234375,
                -1.0078125,
                -1.0234375,
                1.0390625,
                1.0546875,
                1.5078125,
            ],
            [1.984375, 1.0, 1.03125, -1.0, -1.03125, 1.03125, 1.0625, 1.5],
        ),
        (
            "ties_negative_max",
            [-1.984375, -1.0078125, -1.0234375, 1.0078125, 1.0234375],
            [-1.984375, -1.0, -1.03125, 1.0, 1.03125],
        ),
        (
            "clamp_at_top_code",
            [1.9921875, -1.9921875, 1.98828125, 1.9765625],
            [1.984375, -1.984375, 1.984375, 1.96875],
        ),
        (
            "shared_exponent_below_one",
            [0.9921875, 0.50390625, 0.5078125, -0.50390625, 0.25390625],
            [0.9921875, 0.5, 0.5078125, -0.5, 0.25],
        ),
        (
            "small_values_lost",
            [128.0, 0.5, 1.0, 3.0, 5.0, -1.0, -3.0, 0.0625, 1.5, 2.25],
            [128.0, 0.0, 0.0, 4.0, 4.0, 0.0, -4.0, 0.0, 2.0, 2.0],
        ),
        (
            "subnormal_next_to_normals",
            [1.984375, SUB, -SUB, 1.0078125, 0.5],
            [1.984375, 0.0, 0.0, 1.0, 0.5],
        ),
        ("only_subnormals", [SUB, -SUB] + [SUB] * 14, [0.0] * 16),
        ("zeros", [0.0], [0.0]),
        ("negative_zero", [-0.0, 0.0, -0.0], [0.0, 0.0, 0.0]),
        ("one_nonzero", [0.0, 0.0, 0.5], [0.0, 0.0, 0.5]),
        ("one_nonzero_negative", [0.0, -3.0], [0.0, -3.0]),
        ("one_nonzero_subnormal", [0.0, SUB], [0.0, 0.0]),
        (
            "signs",
            [1.0, -1.0, 0.75, -0.75, 0.5, -0.5, 0.25, -0.25, 1.5, -1.5, 1.25, -1.25],
            [1.0, -1.0, 0.75, -0.75, 0.5, -0.5, 0.25, -0.25, 1.5, -1.5, 1.25, -1.25],
        ),
    ],
}


def block(head):
    return torch.tensor(head + [0.0] * (16 - len(head)), dtype=torch.float32)


def bits(t):
    """float32 bit patterns, so that -0.0 and +0.0 differ (TT-NN stores a zero with sign 0)."""
    return t.float().view(torch.int32)


EDGE_CASES = [(fmt, name, x, y) for fmt, vs in EDGE_VECTORS.items() for name, x, y in vs]
EDGE_IDS = [f"{fmt}-{name}" for fmt, name, _, _ in EDGE_CASES]


@pytest.mark.parametrize("fmt,name,x,y", EDGE_CASES, ids=EDGE_IDS)
def test_edge_vectors_match_the_frozen_ttnn_values(fmt, name, x, y):
    got = bfp(block(x), BITS[fmt])
    want = block(y)
    assert torch.equal(bits(got), bits(want)), (
        f"{got.tolist()} != {want.tolist()} (TT-NN {TTNN_VERSION})"
    )
    w = bfp_weight(block(x).reshape(16, 1).repeat(1, 3), BITS[fmt])
    assert torch.equal(bits(w.T[0]), bits(want))


def test_edge_vectors_cover_both_formats_and_both_signs():
    assert sorted(EDGE_VECTORS) == ["bfp4", "bfp8"]
    for fmt, vs in EDGE_VECTORS.items():
        names = {name for name, _, _ in vs}
        assert {"ties", "zeros", "one_nonzero", "signs", "only_subnormals"} <= names
        for _, x, y in vs:
            assert len(x) <= 16 and len(y) <= 16
    assert any(v < 0 for _, x, _ in EDGE_VECTORS["bfp4"] for v in x)
    assert any(v > 0 for _, x, _ in EDGE_VECTORS["bfp8"] for v in x)


def ttnn_edge_round(ttnn, fmt, xs):
    """Each 16-value block of xs ([n, 16], n <= 32) packed by stock TT-NN and read back."""
    t = torch.zeros(32, 32, dtype=torch.float32)
    t[: len(xs), :16] = xs
    dtype = {"bfp4": ttnn.bfloat4_b, "bfp8": ttnn.bfloat8_b}[fmt]
    packed = ttnn.as_tensor(t, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    return ttnn.to_torch(packed).float()[: len(xs), :16]


@pytest.fixture(scope="module")
def ttnn_module():
    ttnn = pytest.importorskip("ttnn")
    try:
        ttnn_edge_round(ttnn, "bfp8", torch.zeros(1, 16))
    except Exception as e:  # TT_FATAL without a card or the simulator
        pytest.skip(f"ttnn sees no Tenstorrent card or simulator: {str(e).splitlines()[:1]}")
    return ttnn


@pytest.mark.parametrize("fmt", ["bfp4", "bfp8"])
def test_edge_vectors_against_the_installed_ttnn(ttnn_module, fmt):
    vs = EDGE_VECTORS[fmt]
    got = ttnn_edge_round(ttnn_module, fmt, torch.stack([block(x) for _, x, _ in vs]))
    for (name, x, y), row in zip(vs, got, strict=True):
        assert torch.equal(bits(row), bits(block(y))), (
            f"{fmt} {name}: {row.tolist()} (frozen from {TTNN_VERSION})"
        )
        assert torch.equal(bits(row), bits(bfp(block(x), BITS[fmt]))), f"{fmt} {name}"
