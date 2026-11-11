# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Xylolabs
"""The BFP grids: idempotence, the exponent search, bfloat4_b inside bfloat8_b (torch only)."""

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
