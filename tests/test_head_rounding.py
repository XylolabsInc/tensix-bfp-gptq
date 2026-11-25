# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Xylolabs
"""The LM head is rounded as the options say, like every layer projection: `--method` picks plain
GPTQ or the exponent search for bfloat4_b, `--bfp8` picks round to nearest, the exponent search or
GPTQ for bfloat8_b, and GPTQ off rounds to nearest. A tiny random Llama model on a CPU; the
routines are spied on (and call through), the head is checked on its grid, and the options that
match the older behavior (`--method gptq --bfp8 rtn`) still give the same head bit for bit."""

import itertools
import json
import random

import pytest
import torch
from safetensors import safe_open
from test_export_tiny import WORDS, config, tiny_model

from tensix_bfp import export as E
from tensix_bfp import gptq as G
from tensix_bfp.bfp import on_grid, quantize

HEAD = (256, 64)  # the tiny model's LM head [vocab, hidden]; no layer projection has this shape


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    base = tmp_path_factory.mktemp("head")
    data = base / "data"
    data.mkdir()
    rnd = random.Random(0)
    (data / "calib_en.txt").write_text(" ".join(rnd.choice(WORDS) for _ in range(3000)))
    model = base / "llama"
    model.mkdir()
    tiny_model(model, "llama")
    return base, data, model


class Spy:
    """The calls with the head's shape of the routines a head can be rounded by (they call
    through), the weights they got and the Hessians that went into gptq_prep."""

    def __init__(self, monkeypatch):
        self.monkeypatch = monkeypatch
        self.calls, self.weights, self.hessians = [], [], []
        self.wrap(E, "round_proj", lambda fmt, prep, method="exp": (fmt, method), True)
        self.wrap(E, "bfp_search_weight", lambda bits: (bits,), True)
        self.wrap(E, "quantize", lambda fmt: (fmt,), True)
        self.wrap(G, "gptq_search", lambda prep, *a, **k: (), False)
        self.wrap(G, "gptq", lambda h, fmt, *a, **k: (fmt,), False)
        real = G.gptq_prep

        def prep(h, act_order=True, damp=0.01):
            self.hessians.append((h.clone(), act_order, damp))
            return real(h, act_order, damp)

        monkeypatch.setattr(E, "gptq_prep", prep)

    def wrap(self, module, name, describe, keep_weight):
        real = getattr(module, name)

        def spy(w, *args, **kwargs):
            if tuple(w.shape) == HEAD:
                self.calls.append((name, *describe(*args, **kwargs)))
                if keep_weight:
                    self.weights.append(w.clone())
            return real(w, *args, **kwargs)

        self.monkeypatch.setattr(module, name, spy)


def expected_calls(fmt, gptq, method, bfp8):
    """The routines a head of format fmt goes through, by the table of a layer projection."""
    if gptq and fmt == "bfp4":
        inner = ("gptq_search",) if method == "exp" else ("gptq", "bfp4")
        return [("round_proj", "bfp4", method), inner]
    if gptq and fmt == "bfp8" and bfp8 == "gptq":
        return [("round_proj", "bfp8", method), ("gptq", "bfp8")]
    if fmt == "bfp8" and bfp8 == "exp":
        return [("bfp_search_weight", 7)]
    return [("quantize", fmt)]


def export_head(setup, tmp_path, monkeypatch, fmt, gptq, method, bfp8):
    base, data, model = setup
    spy = Spy(monkeypatch)
    spec = f"all=bf16,lm_head={fmt}" + (",gptq" if gptq else "")
    argv = [
        *("--config", str(config(base, "llama"))),
        *("--model", str(model)),
        *("--out", str(tmp_path / "out")),
        *("--variant", spec),
        *("--device", "cpu"),
        *("--data", str(data)),
        *("--method", method),
        *("--bfp8", bfp8),
    ]
    assert E.main(argv) == 0
    with safe_open(str(tmp_path / "out" / "head.safetensors"), framework="pt") as f:
        head = f.get_tensor("lm_head.weight")
    return spy, head, json.loads((tmp_path / "out" / "quant4.json").read_text())


COMBINATIONS = list(
    itertools.product(
        ("bf16", "bfp8", "bfp4"), (True, False), ("exp", "gptq"), ("rtn", "exp", "gptq")
    )
)


@pytest.mark.parametrize("fmt,gptq,method,bfp8", COMBINATIONS)
def test_head_rounding(setup, tmp_path, monkeypatch, fmt, gptq, method, bfp8):
    spy, head, man = export_head(setup, tmp_path, monkeypatch, fmt, gptq, method, bfp8)

    # the head went through the routine the table names, with its method and mantissa bits
    want = expected_calls(fmt, gptq, method, bfp8)
    assert spy.calls == want
    assert on_grid(head.float(), fmt)

    # the record describes the head's rounding in the words used for layers
    rec = man["rounding"]
    if gptq and fmt == "bfp4":
        assert rec["method"] == method and man["exponent_search"] == (method == "exp")
        assert ("exponent_offsets" in rec) == (method == "exp")
    elif gptq and fmt == "bfp8" and bfp8 == "gptq":
        assert rec["method"] == method and rec["bfp8"] == "gptq"
    elif fmt == "bfp8" and bfp8 == "exp":
        assert rec["bfp8"] == "exp"
    if not gptq:
        assert rec["method"] == "rtn" and man["exponent_search"] is False
    if bfp8 == "rtn":
        assert "bfp8" not in rec


@pytest.mark.parametrize(
    "fmt,gptq",
    [
        ("bf16", True),
        ("bf16", False),
        ("bfp8", True),
        ("bfp8", False),
        ("bfp4", True),
        ("bfp4", False),
    ],
)
def test_head_matches_the_older_behavior(setup, tmp_path, monkeypatch, fmt, gptq):
    """With `--method gptq --bfp8 rtn` the head is what plain GPTQ (bfp4) or round to nearest
    (bf16, bfp8, or GPTQ off) always gave: the same weight and Hessian, the same bits."""
    spy, head, _ = export_head(setup, tmp_path, monkeypatch, fmt, gptq, "gptq", "rtn")
    w = spy.weights[0]
    if gptq and fmt == "bfp4":
        ((h, act_order, damp),) = spy.hessians
        assert act_order is True and damp == 0.1
        old = G.gptq(w, h, fmt, damp=damp)[0]
    else:
        assert spy.hessians == []
        old = quantize(w, fmt)
    assert torch.equal(head.float(), old.bfloat16().float())
