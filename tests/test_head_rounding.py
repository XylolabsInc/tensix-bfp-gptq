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


@pytest.mark.parametrize("manifest_format", [1, 2])
def test_prefixed_head_uses_selected_name(tmp_path, manifest_format):
    from types import SimpleNamespace

    from safetensors.torch import save_file

    src = tmp_path / "src"
    src.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    name = "wrapper.lm_head.weight"
    original = torch.ones(32, 32, dtype=torch.bfloat16)
    save_file(
        {name: original, "model.norm.weight": torch.ones(32)}, str(src / "weights.safetensors")
    )
    md = SimpleNamespace(
        head_name=name,
        fam=SimpleNamespace(prefix="model.", final_norm="norm", embed="embed_tokens"),
        ckpt=SimpleNamespace(
            dir=src, index={name: "weights.safetensors", "model.norm.weight": "weights.safetensors"}
        ),
        untied=lambda: True,
    )
    variant = SimpleNamespace(of=lambda group: "bfp4", tag=lambda: "test")
    manifest = {}
    rounded = torch.full((32, 32), 0.5)
    E.save_head(md, variant, out, torch.ones(32), rounded, manifest, manifest_format)
    with safe_open(str(out / "head.safetensors"), framework="pt") as f:
        assert torch.equal(f.get_tensor(name), rounded)
        if manifest_format == 1:
            assert torch.equal(f.get_tensor("lm_head.weight"), rounded)
    index = json.loads((out / "model.safetensors.index.json").read_text())["weight_map"]
    assert index[name] == "head.safetensors"
    if manifest_format == 2:
        assert manifest["lm_head_tensor"] == name


def test_prefixed_head_export_from_two_shards(tmp_path):
    from types import SimpleNamespace

    from safetensors.torch import load_file, save_file

    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    out.mkdir()
    head, norm, other = "wrapper.lm_head.weight", "model.norm.weight", "model.other.weight"
    save_file(
        {head: torch.ones(32, 32), other: torch.full((32, 32), 2.0)}, str(src / "a.safetensors")
    )
    save_file({norm: torch.ones(32)}, str(src / "b.safetensors"))
    source_index = {head: "a.safetensors", other: "a.safetensors", norm: "b.safetensors"}
    (src / "model.safetensors.index.json").write_text(json.dumps({"weight_map": source_index}))
    md = SimpleNamespace(
        head_name=head,
        fam=SimpleNamespace(prefix="model.", final_norm="norm", embed="embed_tokens"),
        ckpt=SimpleNamespace(dir=src, index=source_index),
        untied=lambda: True,
    )
    variant = SimpleNamespace(of=lambda group: "bfp4", tag=lambda: "test")
    rounded = torch.full((32, 32), 0.5)
    manifest = {}
    E.save_head(md, variant, out, torch.full((32,), 3.0), rounded, manifest, 2)
    index = json.loads((out / "model.safetensors.index.json").read_text())["weight_map"]
    state = {name: load_file(str(out / shard))[name] for name, shard in index.items()}
    model = torch.nn.Module()
    model.wrapper = torch.nn.Module()
    model.wrapper.lm_head = torch.nn.Linear(32, 32, bias=False)
    model.model = torch.nn.Module()
    model.model.norm = torch.nn.LayerNorm(32, bias=False)
    model.model.other = torch.nn.Linear(32, 32, bias=False)
    model.load_state_dict(state)
    assert manifest["lm_head_tensor"] == head
    assert torch.equal(model.wrapper.lm_head.weight, rounded)
    assert torch.equal(model.model.norm.weight, torch.full((32,), 3.0))
    assert torch.equal(model.model.other.weight, torch.full((32, 32), 2.0))
