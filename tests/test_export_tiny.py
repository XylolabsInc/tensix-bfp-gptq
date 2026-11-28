# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Xylolabs
"""End-to-end export of tiny random Qwen3 and Llama models built in memory (torch and
transformers, CPU): every rounded matrix on its grid by verify_grid's rule, the manifest and a
self-contained index written, the export loadable by transformers, and a second run identical."""

import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from tensix_bfp import answers
from tensix_bfp import export as E
from tensix_bfp.bfp import on_grid

ROOT = Path(__file__).resolve().parents[1]
WORDS = [f"w{i}" for i in range(200)]


def tokenizer_json(path):
    from tokenizers import Tokenizer, models, pre_tokenizers

    vocab = {"[UNK]": 0, "<s>": 1, "</s>": 2, **{w: i + 3 for i, w in enumerate(WORDS)}}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.save(str(path))


def tiny_model(d, family):
    import transformers as tf

    torch.manual_seed(0)
    common = dict(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=256,
        bos_token_id=1,
        eos_token_id=2,
    )
    if family == "qwen3":
        cfg = tf.Qwen3Config(tie_word_embeddings=True, **common)
        model = tf.Qwen3ForCausalLM(cfg)
    else:
        cfg = tf.LlamaConfig(tie_word_embeddings=False, **common)
        model = tf.LlamaForCausalLM(cfg)
    with torch.no_grad():  # norms away from 1, so that the norm weights matter
        for n, p in model.named_parameters():
            if "norm" in n:
                p.uniform_(0.5, 1.5)
    model.to(torch.bfloat16).save_pretrained(d)
    tokenizer_json(d / "tokenizer.json")


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    base = tmp_path_factory.mktemp("tiny")
    data = base / "data"
    data.mkdir()
    rnd = random.Random(0)
    (data / "calib_en.txt").write_text(" ".join(rnd.choice(WORDS) for _ in range(3000)))
    models = {}
    for family in ("qwen3", "llama"):
        d = base / family
        d.mkdir()
        tiny_model(d, family)
        models[family] = d
    return base, data, models


def config(base, family):
    p = base / f"{family}.toml"
    p.write_text(
        f'[model]\nname = "tiny-{family}"\nfamily = "{family}"\n'
        '[calib]\ntexts = ["calib_en"]\ntokens = 512\nseq = 64\n'
        "[quant]\ndamp = 0.1\n"
    )
    return p


def run(base, data, models, family, variant, out, *extra):
    argv = [
        "--config",
        str(config(base, family)),
        "--model",
        str(models[family]),
        "--out",
        str(out),
        "--variant",
        variant,
        "--device",
        "cpu",
        "--data",
        str(data),
        *extra,
    ]
    assert E.main(argv) == 0
    return out


def digest(d):
    return {f.name: hashlib.md5(f.read_bytes()).hexdigest() for f in sorted(d.iterdir())}


@pytest.mark.parametrize(
    "family,variant",
    [
        ("qwen3", "all=bfp4,lm_head=bf16,gptq"),
        ("llama", "attn_qkv=bfp8,attn_o=bfp8,mlp=bfp4,lm_head=bfp4,gptq"),
        ("llama", "all=bfp8"),
    ],
)
def test_export(setup, tmp_path, family, variant):
    base, data, models = setup
    out = run(base, data, models, family, variant, tmp_path / "out")
    v = E.parse(variant, E.FAMILIES[family], 0.1)

    # every rounded matrix on its grid, by the shipped checker
    r = subprocess.run(
        [sys.executable, str(ROOT / "tools/verify_grid.py"), str(out)],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    expected = 14 + (v.of("lm_head") in ("bfp4", "bfp8"))
    assert f"{expected} matrices checked, 0 off the grid" in r.stdout

    # the manifest
    man = json.loads((out / "quant4.json").read_text())
    assert list(man) == [
        "layers",
        "online_signs",
        "variant",
        "model",
        "online_block",
        "exponent_search",
        "rounding",
        "lm_head",
        "lm_head_tensor",
    ]
    assert man["variant"] == v.tag() and man["model"] == f"tiny-{family}"
    for i in ("0", "1"):
        assert list(man["layers"][i]) == list(E.FAMILIES[family].linears)
        for name, p in E.FAMILIES[family].linears.items():
            assert man["layers"][i][name] == {"format": v.of(p.group, int(i)), "online": False}
    if v.gptq:
        assert man["rounding"] == {
            "method": "exp",
            "damp": 0.1,
            "formats": ["bfp4"],
            "exponent_offsets": [-1, 0],
        }
        assert man["exponent_search"] is True
    else:
        assert man["rounding"] == {"method": "rtn"} and man["exponent_search"] is False
    tied = family == "qwen3"
    assert man["lm_head_tensor"] == ("model.embed_tokens.weight" if tied else "lm_head.weight")

    # a self-contained index that names every tensor of the original
    wm = answers.export_files_clean(out)
    orig = answers.weight_map(models[family])
    assert set(orig) <= set(wm) and "lm_head.weight" in wm
    assert not any(f.is_symlink() for f in out.iterdir())

    # transformers loads the export; its weights are the files' and differ from the original's
    m, _ = answers.load_hf(out, "cpu")
    rec = answers.weights_check(m.state_dict(), out, models[family])
    assert rec["differ_from_bf16"] > 0
    q = m.state_dict()["model.layers.0.self_attn.q_proj.weight"].float()
    assert on_grid(q, v.of("attn_qkv", 0))


def test_deterministic(setup, tmp_path):
    base, data, models = setup
    spec = "all=bfp4,lm_head=bf16,gptq"
    a = run(base, data, models, "qwen3", spec, tmp_path / "a", "--manifest-format", "1")
    b = run(base, data, models, "qwen3", spec, tmp_path / "b", "--manifest-format", "1")
    assert digest(a) == digest(b)
    man = json.loads((a / "quant4.json").read_text())
    assert "lm_head_tensor" not in man


def test_refuses_nonempty_out(setup, tmp_path):
    base, data, models = setup
    (tmp_path / "x").write_text("x")
    with pytest.raises(SystemExit):
        run(base, data, models, "qwen3", "all=bfp8", tmp_path)


def test_variant_parse():
    fam = E.FAMILIES["qwen3_5"]
    v = E.parse("all=bfp4,ab=bfp4,lm_head=bf16,gptq,mlp@1-2+5=bfp8", fam, 0.1)
    assert v.of("gdn_in", 0) == "bfp4" and v.of("ab") == "bfp4" and v.of("lm_head") == "bf16"
    assert v.of("mlp_down", 2) == "bfp8" and v.of("mlp_down", 3) == "bfp4"
    assert v.tag() == (
        "ab=bfp4,attn_o=bfp4,attn_qkv=bfp4,gdn_in=bfp4,gdn_out=bfp4,lm_head=bf16,"
        "mlp_down=bfp4,mlp_gate_up=bfp4,router=bf16 gptq+damp0.1+mlp@1-2+5=bfp8"
    )
    with pytest.raises(ValueError):
        E.parse("all=int4", fam, 0.1)


def test_chat_chunks_cover_text_without_gaps():
    md = object.__new__(E.Model)
    md.cfg = {"eval": {"texts": ["text"], "formats": ["chat"], "tokens": 6, "chunks": 3}}
    md.bos_ids = lambda: []
    md.chat_prefix = lambda: [90, 91]
    md._text_ids = lambda: lambda text, prefix: list(range(20))
    _, seqs, first = md.eval_tokens()
    assert first == [2, 2, 2]
    assert [t for seq in seqs for t in seq[2:]] == list(range(12))
    md.cfg["eval"]["tokens"] = 2
    with pytest.raises(ValueError, match="prefix"):
        md.eval_tokens()


@pytest.mark.parametrize("fmt", ["bfp4", "bfp8"])
def test_tied_head_export_matches_loaded_embedding(setup, tmp_path, fmt):
    from safetensors.torch import load_file
    from transformers import Qwen3ForCausalLM

    base, data, models = setup
    out = tmp_path / "out"
    run(base, data, models, "qwen3", f"all=bf16,lm_head={fmt}", out)
    man = json.loads((out / "quant4.json").read_text())
    tensors = load_file(str(out / "head.safetensors"))
    name = man["lm_head_tensor"]
    assert name == "model.embed_tokens.weight"
    assert torch.equal(tensors[name], tensors["lm_head.weight"])
    assert on_grid(tensors[name].float(), fmt)
    model = Qwen3ForCausalLM.from_pretrained(out)
    assert torch.equal(model.lm_head.weight, tensors[name])
    assert torch.equal(model.model.embed_tokens.weight, tensors[name])


def test_hessian_normalization_uses_model_epsilon(setup, tmp_path, monkeypatch):
    import shutil

    base, data, models = setup
    model_dir = tmp_path / "llama"
    shutil.copytree(models["llama"], model_dir)
    config_path = model_dir / "config.json"
    cfg = json.loads(config_path.read_text())
    cfg["rms_norm_eps"] = 0.002
    config_path.write_text(json.dumps(cfg))
    seen = []
    original = E.plain_norm

    def record(x, eps=1e-6):
        seen.append(eps)
        return original(x, eps)

    monkeypatch.setattr(E, "plain_norm", record)
    run(
        base,
        data,
        {"llama": model_dir},
        "llama",
        "all=bf16,lm_head=bfp8,gptq",
        tmp_path / "out",
        "--bfp8",
        "gptq",
    )
    assert seen and set(seen) == {0.002}
    x = torch.tensor([[0.0001, 0.0002]])
    expected = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 0.002)
    assert torch.equal(original(x, 0.002), expected)
