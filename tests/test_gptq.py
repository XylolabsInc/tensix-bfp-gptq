# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Xylolabs
"""GPTQ onto the BFP grids: results on the grid and no worse than rounding to nearest (torch
only, CPU)."""

import pytest
import torch

from tensix_bfp.bfp import on_grid, quantize
from tensix_bfp.export import pack, symmetric, unpack
from tensix_bfp.gptq import Hessian, gptq, gptq_many, gptq_prep, gptq_search, round_proj


def problem(rows=64, cols=256, n=4096, seed=0):
    """A weight and the Hessian of correlated inputs with a few large channels."""
    g = torch.Generator().manual_seed(seed)
    mix = torch.randn(cols, cols, generator=g) / cols**0.5 + torch.eye(cols)
    x = torch.randn(n, cols, generator=g) @ mix
    x[:, :8] *= 10
    h = Hessian(cols)
    h.add(x)
    w = torch.randn(rows, cols, generator=g) * 0.05
    return w.to(torch.bfloat16).float(), h.get()


def loss(w, q, h):
    d = (q - w).double()
    return (d @ h.double() * d).sum().item()


def test_hessian_mean():
    g = torch.Generator().manual_seed(1)
    x = torch.randn(5000, 32, generator=g)
    h = Hessian(32)
    h.add(x[:3000])
    h.add(x[3000:])
    want = x.double().T @ x.double() / 5000
    assert torch.allclose(h.get().double(), want, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("fmt", ["bfp4", "bfp8"])
def test_gptq_on_grid_and_better_than_rtn(fmt):
    w, h = problem()
    q = gptq(w, h, fmt, damp=0.01)[0]
    assert on_grid(q, fmt)
    assert loss(w, q, h) < loss(w, quantize(w, fmt), h)


def test_gptq_search_on_grid_and_better_than_rtn():
    w, h = problem(seed=2)
    prep = gptq_prep(h, True, 0.1)
    q = gptq_search(w, prep)
    assert on_grid(q, "bfp4")
    assert loss(w, q, h) < loss(w, quantize(w, "bfp4"), h)
    assert torch.equal(round_proj(w, "bfp4", prep, "exp"), q)
    assert torch.equal(round_proj(w, "bfp4", prep, "gptq"), gptq(w, None, "bfp4", prep=prep)[0])


def test_gptq_many_equals_gptq():
    preps, ws = [], []
    for s in range(3):
        w, h = problem(seed=10 + s)
        ws.append(w)
        preps.append(gptq_prep(h, True, 0.1))
    many = gptq_many(torch.stack(ws), preps, "bfp4")
    for w, prep, q in zip(ws, preps, many, strict=True):
        assert torch.equal(q, gptq(w, None, "bfp4", prep=prep)[0])


def test_dead_columns():
    w, h = problem(seed=3)
    h[5, :] = 0
    h[:, 5] = 0
    q = gptq(w, h, "bfp4", damp=0.1)[0]
    assert on_grid(q, "bfp4") and (q[:, 5] == 0).all()


def test_pack_round_trip():
    g = torch.Generator().manual_seed(4)
    a = torch.randn(48, 48, generator=g)
    sym = a + a.T
    store = {"in": sym, "o": a, "n": 7}
    assert torch.equal(unpack(pack(store))["in"], sym)
    mirrored = symmetric(store)["o"]
    assert torch.equal(mirrored, a.triu() + a.triu(1).T) and store["n"] == 7
