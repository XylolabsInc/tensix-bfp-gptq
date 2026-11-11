# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Xylolabs
"""GPTQ (Frantar et al., 2022) onto the BFP grids of bfp.py.

GPTQ rounds a weight one input channel (column) at a time and spreads each column's rounding
error over the columns not yet rounded, through the upper Cholesky factor of the damped inverse
Hessian H = E[x^T x] of the projection's input. A BFP block (16 output channels of one input
channel) lies inside one column, so any column order works; columns go by decreasing diag(H)
(act order).

gptq_search() is GPTQ for bfloat4_b whose per-block exponent is searched at every column: the
block's largest exponent E and E - 1 are both tried and the smaller squared error is kept.

New tensors are created on torch's default device; the export sets it to the device it runs on.
"""

import os

import torch

from .bfp import EXP_OFFSETS, bfp, bfp_search, bfp_weight, parse_fmt, quantize

# replay each full column block of gptq_search() as a CUDA graph (same kernels, same results)
USE_GRAPH = os.environ.get("TENSIX_BFP_GRAPH", "1") != "0"


class Hessian:
    """H = sum of x^T x over the rows a projection reads, accumulated in float64 from float32
    products of 2048-row chunks; get() returns the mean in float32."""

    def __init__(self, k):
        self.h = torch.zeros(k, k, dtype=torch.float64)
        self.n = 0

    def add(self, x):
        x = x.reshape(-1, x.shape[-1]).float()
        for a in range(0, len(x), 2048):
            xc = x[a : a + 2048]
            self.h += (xc.T @ xc).double()
        self.n += len(x)

    def get(self):
        return (self.h / max(self.n, 1)).float()


def gptq_prep(h, act_order=True, damp=0.01):
    """What GPTQ needs from a Hessian, shared by the weights that read the same input: (column
    order or None, the upper Cholesky factor of the damped inverse, the dead columns)."""
    h = h.float().clone()
    dead = torch.diag(h) == 0
    h[dead, dead] = 1
    perm = None
    if act_order:
        perm = torch.argsort(torch.diag(h), descending=True)
        h = h[perm][:, perm]
    h.diagonal().add_(damp * torch.mean(torch.diag(h)))
    # the upper factor U of H^-1 (U^T U = H^-1) from one Cholesky factorization: for the
    # reversed matrix J H J = L L^T, H^-1 = (J L^-T J)(J L^-1 J), so U = J L^-1 J
    lo = torch.linalg.cholesky(h.flip(0, 1))
    del h
    eye = torch.eye(len(lo), dtype=lo.dtype)
    hinv = torch.linalg.solve_triangular(lo, eye, upper=False).flip(0, 1)
    return perm, hinv, dead


def gptq(w, h, fmt, block=128, damp=0.01, act_order=True, prep=None):
    """GPTQ of the [out, in] weight w to fmt ("bfp8" or "bfp4"; "bf16" rounds to nearest) with
    the input's Hessian h [in, in] or its gptq_prep(). Returns (the rounded weight as float32,
    the GPTQ loss, None)."""
    kind, bits, _ = parse_fmt(fmt)
    if kind == "bf16":
        return quantize(w, fmt), 0.0, None
    if prep is None:
        prep = gptq_prep(h, act_order, damp)
    perm, hinv, dead = prep
    assert (perm is not None) == act_order, "gptq: the prep's column order does not match"
    w = w.float().clone()
    cols = w.shape[1]
    w[:, dead] = 0
    if perm is not None:
        w = w[:, perm]
    q = torch.zeros_like(w)
    loss = torch.zeros((), dtype=torch.float64)
    for i1 in range(0, cols, block):
        i2 = min(i1 + block, cols)
        w1 = w[:, i1:i2].clone()
        q1 = torch.zeros_like(w1)
        err1 = torch.zeros_like(w1)
        hinv1 = hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            col = w1[:, i]
            d = hinv1[i, i]
            qc = bfp(col.reshape(1, -1), bits).reshape(-1)
            q1[:, i] = qc
            e = (col - qc) / d
            loss += (e * e).sum().double() / 2
            w1[:, i:] -= e.unsqueeze(1) @ hinv1[i, i:].unsqueeze(0)
            err1[:, i] = e
        q[:, i1:i2] = q1
        if i2 < cols:
            w[:, i2:] -= err1 @ hinv[i1:i2, i2:]
    if perm is not None:
        q = q[:, torch.argsort(perm)]
    return q, loss.item(), None


def gptq_many(ws, preps, fmt, block=128, outer="mul"):
    """gptq() of n weights of one shape at once (ws [n, out, in], preps: each weight's own
    gptq_prep()), for "bfp8" and "bfp4": the column steps batched over the weights, the rank-1
    update as the exact product e h (outer="mul") or a batched k = 1 matmul ("matmul"), the block
    update as gptq()'s own matmul per weight. Returns the rounded weights [n, out, in]."""
    kind, bits, _ = parse_fmt(fmt)
    assert kind == "bfp", f"gptq_many: {fmt} is not a BFP format"
    n, _, cols = ws.shape
    assert len(preps) == n
    w = ws.float().clone()
    for j, (perm, _, dead) in enumerate(preps):
        w[j][:, dead] = 0
        if perm is not None:
            w[j] = w[j][:, perm]
    hinv = torch.stack([p[1] for p in preps])
    q = torch.zeros_like(w)
    for i1 in range(0, cols, block):
        i2 = min(i1 + block, cols)
        w1 = w[:, :, i1:i2].clone()
        q1 = torch.zeros_like(w1)
        err1 = torch.zeros_like(w1)
        hinv1 = hinv[:, i1:i2, i1:i2]
        for i in range(i2 - i1):
            col = w1[:, :, i]
            d = hinv1[:, i, i]
            qc = bfp(col, bits)
            q1[:, :, i] = qc
            e = (col - qc) / d[:, None]
            if outer == "mul":
                w1[:, :, i:] -= e[:, :, None] * hinv1[:, i : i + 1, i:]
            else:
                w1[:, :, i:] -= torch.matmul(e[:, :, None], hinv1[:, i : i + 1, i:])
            err1[:, :, i] = e
        q[:, :, i1:i2] = q1
        if i2 < cols:
            for j in range(n):
                w[j, :, i2:] -= err1[j] @ hinv[j, i1:i2, i2:]
    for j, (perm, _, _) in enumerate(preps):
        if perm is not None:
            q[j] = q[j][:, torch.argsort(perm)]
    return q


# --- GPTQ with the exponent search (bfloat4_b) --------------------------------------------------


def _search_block(w1, q1, err1, hinv1, cands):
    """The column steps of one block, in place: w1 [rows, b] updated, q1 and err1 [rows, b]
    written, hinv1 [b, b]. The rank-1 update is the exact elementwise product e h."""
    for i in range(w1.shape[1]):
        col = w1[:, i]
        qc = bfp_search(col.reshape(1, -1), 3, cands).reshape(-1)
        q1[:, i] = qc
        e = (col - qc) / hinv1[i, i]
        w1[:, i:] -= e.unsqueeze(1) * hinv1[i, i:].unsqueeze(0)
        err1[:, i] = e


GRAPHS = {}  # (rows, block, cands, device) -> (graph, w1, q1, err1, hinv1)


def _block_graph(rows, block, cands, device):
    """A CUDA graph of _search_block() on static buffers of one shape, captured once."""
    key = (rows, block, cands, str(device))
    if key not in GRAPHS:
        w1 = torch.zeros(rows, block, device=device)
        q1, err1 = torch.zeros_like(w1), torch.zeros_like(w1)
        hinv1 = torch.eye(block, device=device)
        s = torch.cuda.Stream(device)
        s.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(s):  # a warm-up run on a side stream before capture
            _search_block(w1.clone(), q1.clone(), err1.clone(), hinv1, cands)
        torch.cuda.current_stream(device).wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            _search_block(w1, q1, err1, hinv1, cands)
        GRAPHS[key] = (g, w1, q1, err1, hinv1)
    return GRAPHS[key]


def gptq_search(w, prep, block=128, cands=EXP_OFFSETS):
    """GPTQ of the [out, in] weight w onto the bfloat4_b grid with the exponent search per block
    (column order from prep). On a CUDA device each full column block replays a CUDA graph of its
    steps. Returns the rounded weight as float32."""
    perm, hinv, dead = prep
    w = w.float().clone()
    rows, cols = w.shape
    w[:, dead] = 0
    if perm is not None:
        w = w[:, perm]
    q = torch.zeros_like(w)
    graph = w.is_cuda and USE_GRAPH
    for i1 in range(0, cols, block):
        i2 = min(i1 + block, cols)
        if graph and i2 - i1 == block:
            g, w1, q1, err1, hinv1 = _block_graph(rows, block, tuple(cands), w.device)
            w1.copy_(w[:, i1:i2])
            hinv1.copy_(hinv[i1:i2, i1:i2])
            g.replay()
        else:
            w1 = w[:, i1:i2].clone()
            q1 = torch.zeros_like(w1)
            err1 = torch.zeros_like(w1)
            hinv1 = hinv[i1:i2, i1:i2]
            _search_block(w1, q1, err1, hinv1, cands)
        q[:, i1:i2] = q1
        if i2 < cols:
            w[:, i2:] -= err1 @ hinv[i1:i2, i2:]
    if perm is not None:
        q = q[:, torch.argsort(perm)]
    return q


def check_grid(q):
    assert torch.equal(bfp_weight(q, 3), q), "gptq_search: a result is off the bfloat4_b grid"


def round_proj(w, fmt, prep, method="exp"):
    """A projection rounded by GPTQ to fmt: with method "exp" bfloat4_b takes gptq_search(),
    every other case plain gptq(). Returns the rounded weight as float32."""
    if method == "gptq" or fmt != "bfp4":
        return gptq(w, None, fmt, prep=prep)[0]
    q = gptq_search(w, prep)
    check_grid(q)
    return q
