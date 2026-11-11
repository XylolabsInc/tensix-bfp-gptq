# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Xylolabs
"""Export a Hugging Face checkpoint whose linear projections lie on TT-NN's block floating point
grids (bfloat8_b, bfloat4_b), rounded to nearest or by GPTQ, as bf16 safetensors.

  python -m tensix_bfp.export --config CONFIG.toml --model BASE_DIR --out OUT_DIR \\
      [--variant SPEC] [--device cuda:0] [--data DIR] [--method exp|gptq] [--bfp8 rtn|exp|gptq]

The model runs one decoder layer at a time. For a GPTQ variant each layer first runs on the
calibration sequences with every projection as a bf16 matmul with float32 accumulation (the
residual stream in float32), recording the Hessian of every projection input: the plain-normed
residual before the token mixer ("in") and before the MLP ("mlp"), multiplied by the norm weight
on both sides, and the inputs of the output and down projections as they are. The layer's
projections are then rounded with those Hessians and written out, and the calibration stream
moves on through the original layer.

A VARIANT is a comma list: GROUP=FMT for the family's groups (attn_qkv, attn_o, mlp_gate_up,
mlp_down, lm_head; Qwen3.5 adds gdn_in, gdn_out and ab), all=FMT (every group), mix=FMT (the
token mixers), mlp=FMT (the MLP), GROUP@LAYERS=FMT (those layers only, LAYERS as 36-39+50), gptq,
and damp=X. Formats: bf16, bfp8, bfp4. Unnamed groups stay bf16, Qwen3.5's ab group bfp8.

The output directory holds:
  layers-NN.safetensors          decoder layer NN: every rounded matrix as a bf16 tensor whose
                                 values lie exactly on its grid, the other tensors as stored
  head.safetensors               the LM head (rounded the same way) and the final norm
  rest.safetensors               every other tensor of the original (embedding, MTP, vision ...)
  model.safetensors.index.json   the weight map
  quant4.json                    the variant and every projection's format per layer
  config.json, tokenizer files   copies of the original's
"""

import argparse
import json
import shutil
import time
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path

import torch

from . import data as datamod
from .bfp import EXP_OFFSETS, bfp_search_weight, bfp_weight, parse_fmt, quantize, rel
from .gptq import Hessian, gptq, gptq_prep, round_proj

DEFAULTS = {
    "calib": {"texts": ["calib_en", "calib_ko"], "tokens": 24576, "seq": 1024},
    "eval": {
        "texts": ["eval_en", "eval_ko"],
        "tokens": 2048,
        "chunks": 1,
        "formats": ["raw"],
        "chat_user": "Write an encyclopedia article.",
    },
    "quant": {"damp": 0.1, "variants": [], "prep_cache": False},
}
MANIFEST = "quant4.json"
FORMATS = ("bf16", "bfp8", "bfp4")


def load_config(path):
    """A TOML config with the defaults filled in. [quant] prep_cache is accepted and ignored."""
    cfg = tomllib.loads(Path(path).read_text())
    for k, v in DEFAULTS.items():
        cfg[k] = {**v, **cfg.get(k, {})}
    return cfg


# --- model families -----------------------------------------------------------------------------


@dataclass
class Proj:
    group: str  # the variant's key
    site: str  # the input whose Hessian GPTQ reads


@dataclass
class Family:
    name: str
    prefix: str  # of the text model's tensors in the checkpoint
    norm_offset: float  # the RMSNorms multiply by norm_offset + w
    linears: dict  # module name in a decoder layer -> Proj
    groups: tuple = ()
    mlp_norm: str = "post_attention_layernorm"  # the norm before the MLP
    in_norm: str = "input_layernorm"  # the norm before the token mixer
    final_norm: str = "norm"
    embed_scale: bool = False  # the embedding times sqrt(hidden), in the weight's dtype
    rope_per_type: bool = False  # one rotary embedding per layer type
    bos_first: bool = False  # every real input starts with the begin-of-sequence token
    embed: str = "embed_tokens"

    def site_norm(self, site):
        return {"in": self.in_norm, "mlp": self.mlp_norm}.get(site)

    # --- transformers classes -------------------------------------------------------------

    def config(self, hf_dir):
        if self.name == "qwen3_5":
            from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig as C
        elif self.name == "gemma4_unified":
            from transformers.models.gemma4_unified.configuration_gemma4_unified import (
                Gemma4UnifiedTextConfig as C,
            )
        elif self.name == "gemma4":
            from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig as C
        elif self.name == "qwen3":
            from transformers.models.qwen3.configuration_qwen3 import Qwen3Config as C
        elif self.name == "qwen2":
            from transformers.models.qwen2.configuration_qwen2 import Qwen2Config as C
        elif self.name == "ministral3":  # the text-only copy (tensix_bfp.ministral3)
            from transformers.models.ministral3.configuration_ministral3 import Ministral3Config

            c = json.loads((Path(hf_dir) / "config.json").read_text())
            cfg = Ministral3Config(**c.get("text_config", c))
            cfg._attn_implementation = "sdpa"
            assert cfg.sliding_window is None, "ministral3: a sliding window is not modeled"
            return cfg
        else:
            from transformers import LlamaConfig as C
        cfg = C.from_pretrained(hf_dir)
        cfg._attn_implementation = "sdpa"
        if self.name.startswith("gemma4"):  # the text config of a multimodal checkpoint
            t = json.loads((Path(hf_dir) / "config.json").read_text()).get("text_config", {})
            for k in ("num_hidden_layers", "hidden_size", "intermediate_size", "vocab_size"):
                assert getattr(cfg, k) == t[k], (
                    f"{self.name} config: {k} {getattr(cfg, k)} != {t[k]}"
                )
        return cfg

    def layer(self, cfg, i):
        """Decoder layer i on the meta device (load_state_dict(assign=True) fills it)."""
        with torch.device("meta"):
            if self.name == "qwen3_5":
                from transformers.models.qwen3_5 import modeling_qwen3_5 as hf

                return hf.Qwen3_5DecoderLayer(cfg, i)
            if self.name == "gemma4_unified":
                from transformers.models.gemma4_unified import modeling_gemma4_unified as hf

                return hf.Gemma4UnifiedTextDecoderLayer(cfg, i)
            if self.name == "gemma4":
                from transformers.models.gemma4 import modeling_gemma4 as hf

                assert not getattr(cfg, "enable_moe_block", False), "gemma4: MoE is not supported"
                return hf.Gemma4TextDecoderLayer(cfg, i)
            if self.name == "ministral3":
                # Mistral's layer: Ministral 3's attention differs only by a query scale that is
                # 1 below original_max_position_embeddings, far beyond every sequence here
                from transformers.models.mistral import modeling_mistral as hf

                return hf.MistralDecoderLayer(cfg, i)
            if self.name == "qwen3":
                from transformers.models.qwen3 import modeling_qwen3 as hf

                return hf.Qwen3DecoderLayer(cfg, i)
            if self.name == "qwen2":
                from transformers.models.qwen2 import modeling_qwen2 as hf

                assert not cfg.use_sliding_window, "qwen2: a sliding window is not modeled"
                return hf.Qwen2DecoderLayer(cfg, i)
            from transformers.models.llama import modeling_llama as hf

            return hf.LlamaDecoderLayer(cfg, i)

    def rotary(self, cfg):
        if self.name == "qwen3_5":
            from transformers.models.qwen3_5 import modeling_qwen3_5 as hf

            return hf.Qwen3_5TextRotaryEmbedding(cfg)
        if self.name == "gemma4_unified":
            from transformers.models.gemma4_unified import modeling_gemma4_unified as hf

            return hf.Gemma4UnifiedTextRotaryEmbedding(cfg)
        if self.name == "gemma4":
            from transformers.models.gemma4 import modeling_gemma4 as hf

            return hf.Gemma4TextRotaryEmbedding(cfg)
        if self.name == "qwen3":
            from transformers.models.qwen3 import modeling_qwen3 as hf

            return hf.Qwen3RotaryEmbedding(config=cfg)
        if self.name == "qwen2":
            from transformers.models.qwen2 import modeling_qwen2 as hf

            return hf.Qwen2RotaryEmbedding(config=cfg)
        if self.name == "ministral3":
            from transformers.models.ministral3 import modeling_ministral3 as hf

            return hf.Ministral3RotaryEmbedding(config=cfg)
        from transformers.models.llama import modeling_llama as hf

        return hf.LlamaRotaryEmbedding(config=cfg)

    def rope_kind(self, cfg, i):
        """Layer i's rotary embedding: its layer type where the family has one per type, None for
        one shared by every layer, False for a layer without one."""
        if not self.rope_per_type:
            return None
        kind = cfg.layer_types[i]
        return kind if kind in cfg.rope_parameters else False

    @staticmethod
    def mixer(m, a_in, rope):
        """The layer's token mixer on its normed input, causal, without a cache."""
        if getattr(m, "block_type", "full_attention") == "linear_attention":
            return m.linear_attn(hidden_states=a_in, cache_params=None)
        return m.self_attn(
            hidden_states=a_in, position_embeddings=rope, attention_mask=None, past_key_values=None
        )[0]

    def unused(self, cfg, i):
        """Prefixes of layer i's tensors its module does not take: Gemma 4 E2B/E4B's KV-sharing
        layers keep k/v weights they never use."""
        if self.name.startswith("gemma4"):
            shared = getattr(cfg, "num_kv_shared_layers", 0) or 0
            if shared and i >= cfg.num_hidden_layers - shared:
                return ("self_attn.k_", "self_attn.v_")
        return ()

    def per_layer_inputs(self, md, seqs, embeds):
        """Gemma 4 E2B/E4B's per-layer inputs, one [1, T, layers, d] float32 tensor per sequence:
        the token's rows of embed_tokens_per_layer times sqrt(d), plus the scaled embedding
        projected per layer, times hidden^-0.5 and RMS-normed with per_layer_projection_norm; the
        sum times 2^-0.5. None for every other model."""
        d = (
            getattr(md.hf, "hidden_size_per_layer_input", 0)
            if self.name.startswith("gemma4")
            else 0
        )
        if not d:
            return [None] * len(seqs)
        cfg, p, n = md.hf, self.prefix, md.hf.num_hidden_layers
        table = p + "embed_tokens_per_layer.weight"
        w = md.ckpt.get(p + "per_layer_model_projection.weight").float()  # [n * d, hidden]
        nw = md.ckpt.get(p + "per_layer_projection_norm.weight").float()
        tok_scale = float(torch.tensor(d**0.5).bfloat16())  # in the table's dtype
        out = []
        for ids, x in zip(seqs, embeds, strict=True):
            t = len(ids)
            tok = md.gather(table, ids).float().reshape(1, t, n, d) * tok_scale
            proj = ((x.float() @ w.T) * cfg.hidden_size**-0.5).reshape(1, t, n, d)
            proj = proj * torch.pow(proj.pow(2).mean(-1, keepdim=True) + cfg.rms_norm_eps, -0.5)
            out.append((proj * nw + tok) * 2**-0.5)
        return out

    def run_layer(self, m, h, rope, parts=False, ctx=None):
        """h [B, T, dim] through layer m: the output, or (a_in, a_out, mid, f_in, f_out, out),
        mid being the residual the MLP's norm reads. ctx: what a sequence's layers share beyond
        the residual (Gemma 4: the KV-sharing layers' keys and values, the per-layer input)."""
        if self.name.startswith("gemma4"):
            return self._run_gemma4(m, h, rope, parts, ctx or {"kv": {}, "ple": None})
        a_in = m.input_layernorm(h)
        a_out = self.mixer(m, a_in, rope)
        mid = h + a_out
        f_in = m.post_attention_layernorm(mid)
        f_out = m.mlp(f_in)
        out = mid + f_out
        return (a_in, a_out, mid, f_in, f_out, out) if parts else out

    @staticmethod
    def _run_gemma4(m, h, rope, parts, ctx):
        """Gemma 4's decoder layer without a cache: causal, the sliding window as an explicit
        mask where the sequence is longer than it, the per-layer input after the MLP (E2B/E4B),
        the KV-sharing layers reading what the last earlier layer of their type stored."""
        att = m.self_attn
        t = h.shape[1]
        mask = (
            window_mask(t, att.sliding_window, h.device)
            if att.is_sliding and t > att.sliding_window
            else None
        )
        a_in = m.input_layernorm(h)
        a = m.self_attn(
            hidden_states=a_in,
            position_embeddings=rope,
            attention_mask=mask,
            shared_kv_states=ctx["kv"],
            past_key_values=None,
        )[0]
        a_out = m.post_attention_layernorm(a)
        mid = h + a_out
        f_in = m.pre_feedforward_layernorm(mid)
        f = m.mlp(f_in)
        f_out = m.post_feedforward_layernorm(f)
        out = mid + f_out
        if getattr(m, "hidden_size_per_layer_input", 0):
            g = m.act_fn(m.per_layer_input_gate(out)) * ctx["ple"]
            out = out + m.post_per_layer_input_norm(m.per_layer_projection(g))
        out = out * m.layer_scalar
        return (a_in, a_out, mid, f_in, f_out, out) if parts else out


def window_mask(t, window, device=None):
    """[1, 1, t, t] additive mask: causal, each query seeing at most `window` keys, itself
    included."""
    dev = device or torch.get_default_device()
    q, k = torch.arange(t, device=dev)[:, None], torch.arange(t, device=dev)[None, :]
    return torch.where((k <= q) & (q - k < window), 0.0, float("-inf"))[None, None]


ATTENTION = {
    "self_attn.q_proj": Proj("attn_qkv", "in"),
    "self_attn.k_proj": Proj("attn_qkv", "in"),
    "self_attn.v_proj": Proj("attn_qkv", "in"),
    "self_attn.o_proj": Proj("attn_o", "o"),
}
DENSE_MLP = {
    "mlp.gate_proj": Proj("mlp_gate_up", "mlp"),
    "mlp.up_proj": Proj("mlp_gate_up", "mlp"),
    "mlp.down_proj": Proj("mlp_down", "down"),
}
LLAMA_GROUPS = ("attn_qkv", "attn_o", "mlp_gate_up", "mlp_down", "lm_head")

FAMILIES = {
    # Qwen3.5 / Qwen3.8 dense: Gated DeltaNet and gated attention layers, (1 + w) RMSNorms
    "qwen3_5": Family(
        "qwen3_5",
        "model.language_model.",
        1.0,
        {
            "linear_attn.in_proj_qkv": Proj("gdn_in", "in"),
            "linear_attn.in_proj_z": Proj("gdn_in", "in"),
            "linear_attn.in_proj_a": Proj("ab", "in"),
            "linear_attn.in_proj_b": Proj("ab", "in"),
            "linear_attn.out_proj": Proj("gdn_out", "out"),
            **ATTENTION,
            **DENSE_MLP,
        },
        groups=("gdn_in", "gdn_out", "attn_qkv", "attn_o", "mlp_gate_up", "mlp_down", "lm_head"),
    ),
    # Llama 3.x: the tokenizer starts every input with the begin-of-sequence token
    "llama": Family(
        "llama", "model.", 0.0, {**ATTENTION, **DENSE_MLP}, LLAMA_GROUPS, bos_first=True
    ),
    # Ministral 3 (text-only copy): Llama's tensors, YaRN rotary, an untied head
    "ministral3": Family(
        "ministral3", "model.", 0.0, {**ATTENTION, **DENSE_MLP}, LLAMA_GROUPS, bos_first=True
    ),
    # Qwen3 dense: Llama's projections, a per-head q_norm and k_norm inside the attention
    "qwen3": Family("qwen3", "model.", 0.0, {**ATTENTION, **DENSE_MLP}, LLAMA_GROUPS),
    # Qwen2 / Qwen2.5: Llama's projections with a bias on q, k and v (kept as stored)
    "qwen2": Family("qwen2", "model.", 0.0, {**ATTENTION, **DENSE_MLP}, LLAMA_GROUPS),
    # Gemma 4 dense text decoder: sliding and full attention, norms before and after each block
    "gemma4": Family(
        "gemma4",
        "model.language_model.",
        0.0,
        {**ATTENTION, **DENSE_MLP},
        LLAMA_GROUPS,
        mlp_norm="pre_feedforward_layernorm",
        embed_scale=True,
        rope_per_type=True,
        bos_first=True,
    ),
}
# Gemma 4 12B: the same layer under transformers' gemma4_unified classes
FAMILIES["gemma4_unified"] = replace(FAMILIES["gemma4"], name="gemma4_unified")
# variant shorthands: name -> the groups it sets
SHORTHANDS = {
    "mix": ("gdn_in", "gdn_out", "attn_qkv", "attn_o"),
    "mlp": ("mlp_gate_up", "mlp_down"),
}


# --- the checkpoint -----------------------------------------------------------------------------


class Checkpoint:
    """Lazy access to a checkpoint's safetensors shards: one tensor, or one decoder layer, read
    on the CPU and moved to `device`."""

    def __init__(self, hf_dir, prefix, embed="embed_tokens", device="cpu"):
        self.dir = Path(hf_dir)
        self.prefix = prefix
        self.embed_name = embed
        self.device = device
        index = self.dir / "model.safetensors.index.json"
        if index.exists():
            self.index = json.loads(index.read_text())["weight_map"]
        else:  # a single-file checkpoint
            from safetensors import safe_open

            with safe_open(str(self.dir / "model.safetensors"), framework="pt") as f:
                self.index = dict.fromkeys(f.keys(), "model.safetensors")

    def _read(self, names):
        from safetensors import safe_open

        out = {}
        with torch.device("cpu"):
            for shard in sorted({self.index[n] for n in names}):
                with safe_open(str(self.dir / shard), framework="pt", device="cpu") as f:
                    for n in names:
                        if self.index[n] == shard:
                            out[n] = f.get_tensor(n)
        return {n: t.to(self.device) for n, t in out.items()}

    def get(self, name):
        return self._read([name])[name]

    def raw_layer(self, i):
        """Text layer i's tensors, keys as in the text model ('layers.i....')."""
        p = f"{self.prefix}layers.{i}."
        return {
            k[len(self.prefix) :]: v
            for k, v in self._read([k for k in self.index if k.startswith(p)]).items()
        }

    def embed(self, tokens):
        """The embedding rows [1, T, dim] of a list of token ids, in the checkpoint's dtype."""
        from safetensors import safe_open

        name = f"{self.prefix}{self.embed_name}.weight"
        with torch.device("cpu"):
            with safe_open(str(self.dir / self.index[name]), framework="pt", device="cpu") as f:
                table = f.get_slice(name)
                e = torch.cat([table[t : t + 1] for t in tokens]).unsqueeze(0)
        return e.to(self.device)


# --- variants -----------------------------------------------------------------------------------


@dataclass
class Variant:
    fam: object
    fmt: dict = field(default_factory=dict)
    gptq: bool = False
    damp: float = 0.1
    layers: list = field(default_factory=list)  # (groups, layer set, format, item) overrides

    def of(self, group, layer=None):
        """group's format, in layer `layer` if given (GROUP@LAYERS=FMT items, the last wins)."""
        fmt = self.fmt.get(group, "bf16")
        for groups, ls, f, _ in self.layers:
            if layer is not None and group in groups and layer in ls:
                fmt = f
        return fmt

    def tag(self):
        f = ",".join(f"{g}={self.fmt[g]}" for g in sorted(self.fmt))
        t = ["gptq" if self.gptq else "rtn"]
        if self.gptq:
            t.append(f"damp{self.damp:g}")
        t += [spec for *_, spec in self.layers]
        return f"{f} {'+'.join(t)}"


def layer_set(text):
    """'36-39+50' -> {36, 37, 38, 39, 50}."""
    out = set()
    for part in text.split("+"):
        a, _, b = part.partition("-")
        out.update(range(int(a), int(b or a) + 1))
    return frozenset(out)


def groups_of(k, fam):
    if k == "all":
        return [g for g in fam.groups if g != "router"]
    if k in SHORTHANDS:
        return [g for g in SHORTHANDS[k] if g in fam.groups]
    if k in fam.groups or k in ("ab", "router"):
        return [k]
    return None


def check_fmt(item, fmt):
    if fmt not in FORMATS:
        raise ValueError(f"variant item {item!r}: format {fmt!r} is not one of {FORMATS}")


def parse(spec, fam, damp):
    """A Variant from its spec (see the module's help)."""
    v = Variant(fam, damp=damp)
    v.fmt = {**dict.fromkeys(fam.groups, "bf16"), "ab": "bfp8", "router": "bf16"}
    for item in [x for x in spec.split(",") if x]:
        k, _, val = item.partition("=")
        g, at, where = k.partition("@")
        if at:
            gs = groups_of(g, fam)
            if gs is None or g == "lm_head":
                raise ValueError(f"variant item {item!r}: no layer group {g!r}")
            check_fmt(item, val)
            v.layers.append((frozenset(gs) - {"lm_head"}, layer_set(where), val, item))
        elif groups_of(k, fam) is not None:
            check_fmt(item, val)
            v.fmt.update(dict.fromkeys(groups_of(k, fam), val))
        elif k == "gptq":
            v.gptq = True
        elif k == "damp":
            v.damp = float(val)
        else:
            raise ValueError(f"variant item {item!r} ({fam.name}: groups {fam.groups})")
    return v


# --- the model ----------------------------------------------------------------------------------


class Model:
    def __init__(self, cfg, hf_dir, device="cpu", data_dir=None):
        self.cfg = cfg
        if cfg["model"]["family"] not in FAMILIES:
            raise SystemExit(f"family {cfg['model']['family']!r}: one of {sorted(FAMILIES)}")
        self.fam = FAMILIES[cfg["model"]["family"]]
        self.ckpt = Checkpoint(hf_dir, self.fam.prefix, self.fam.embed, device)
        self.hf = self.fam.config(self.ckpt.dir)
        self.rotary = self.fam.rotary(self.hf)
        self.n_layers = cfg["model"].get("layers", self.hf.num_hidden_layers)  # fewer: a test
        self.dim = self.hf.hidden_size
        self.data_dir = Path(data_dir) if data_dir else datamod.default_dir()
        self._rope = {}

    def rope(self, t, i=0):
        """cos/sin for T = t positions, for layer i's type where the family has one per type."""
        kind = self.fam.rope_kind(self.hf, i)
        if kind is False:
            return None
        if (t, kind) not in self._rope:
            args = (torch.zeros(1, t, 1), torch.arange(t).unsqueeze(0))
            r = self.rotary(*args, kind) if kind else self.rotary(*args)
            dev = torch.get_default_device()
            self._rope[t, kind] = tuple(x.to(dev) if torch.is_tensor(x) else x for x in r)
        return self._rope[t, kind]

    def _text_ids(self):
        """(text name, prefix tokens) -> the prefix and the text's token ids, cached."""
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(str(self.ckpt.dir / "tokenizer.json"))
        texts = {}

        def ids_of(t, prefix):
            if (t, len(prefix)) not in texts:
                text = (self.data_dir / f"{t}.txt").read_text(encoding="utf-8")
                texts[t, len(prefix)] = [*prefix, *tok.encode(text, add_special_tokens=False).ids]
            return texts[t, len(prefix)]

        return ids_of

    def bos_ids(self):
        return [self.hf.bos_token_id] if self.fam.bos_first else []

    def calib_tokens(self):
        """The calibration sequences: [calib] tokens split evenly over [calib] texts (a short
        text's remainder going to the next), cut into sequences of [calib] seq tokens. Texts are
        encoded without special tokens; for a family whose input starts with the
        begin-of-sequence token, a text is that token + its tokens, and each sequence starts with
        it in place of its first token, as llama.cpp's perplexity tool cuts its chunks."""
        c = self.cfg["calib"]
        bos_ids = self.bos_ids()
        ids_of = self._text_ids()
        calib, left = [], c["tokens"]
        for j, t in enumerate(c["texts"]):
            ids = ids_of(t, bos_ids)
            take = min(left // (len(c["texts"]) - j), len(ids)) // c["seq"] * c["seq"]
            calib += [
                [*bos_ids, *ids[a + len(bos_ids) : a + c["seq"]]] for a in range(0, take, c["seq"])
            ]
            left -= take
        return calib

    def eval_tokens(self):
        """(names, sequences, first scored position of each) of the evaluation: per [eval] text,
        format and chunk, [eval] tokens tokens from where the chunk before ended. "raw" is the
        text as it is (with the begin-of-sequence token first where the family takes it, as for
        calibration); "chat" is the text as the model's reply to [eval] chat_user in its chat
        template, scored on the text's tokens only."""
        e = self.cfg["eval"]
        bos_ids = self.bos_ids()
        ids_of = self._text_ids()

        def piece(t, n, off, prefix):
            ids = ids_of(t, prefix)
            assert len(ids) >= off + n, f"{t}: {len(ids)} tokens, need {off + n}"
            return ids[off : off + n]

        n, names, ev, first = e["tokens"], [], [], []
        pre = self.chat_prefix() if "chat" in e["formats"] else []
        for t in e["texts"]:
            for fmt in e["formats"]:
                for k in range(e["chunks"]):
                    if fmt == "chat":
                        ev.append([*pre, *piece(t, n - len(pre), k * n, [])])
                        first.append(len(pre))
                    elif fmt == "raw":
                        ev.append([*bos_ids, *piece(t, n, k * n, bos_ids)[len(bos_ids) :]])
                        first.append(max(1, len(bos_ids)))
                    else:
                        raise ValueError(f"[eval] formats: {fmt!r} is neither 'raw' nor 'chat'")
                    names.append((t if fmt == "raw" else f"{t}_{fmt}", k))
        return names, ev, first

    def chat_prefix(self):
        """The chat template's tokens up to the model's reply to [eval] chat_user, thinking off
        where the template has the switch. A template that prints the date needs [eval]
        chat_date, so that every run has the same prefix."""
        from transformers import AutoTokenizer

        at = AutoTokenizer.from_pretrained(self.ckpt.dir)
        e = self.cfg["eval"]
        extra = {"current_date": e["chat_date"]} if e.get("chat_date") else {}
        template = at.chat_template if isinstance(at.chat_template, str) else ""
        if "strftime_now" in template and not extra:
            raise SystemExit(
                f"{self.ckpt.dir}: the chat template prints today's date; set [eval] chat_date"
            )
        msg = [{"role": "user", "content": e["chat_user"]}]
        ids = at.apply_chat_template(
            msg, add_generation_prompt=True, tokenize=True, enable_thinking=False, **extra
        )
        return list(ids["input_ids"] if hasattr(ids, "keys") else ids)

    def embed(self, toks):
        e = self.ckpt.embed(toks)
        if self.fam.embed_scale:  # in the weight's dtype, as transformers scales it
            e = e * torch.tensor(self.dim**0.5).to(e.dtype)
        return e.float()

    def head(self):
        """(final norm weight, LM head [vocab, dim] float32). The head is the checkpoint's own
        lm_head tensor (also under a wrapper's prefix), or the embedding when the head is tied.
        A config that says the head is untied without a head tensor is refused. self.head_name
        records the tensor used."""
        nw = self.ckpt.get(f"{self.fam.prefix}{self.fam.final_norm}.weight").float()
        heads = sorted(k for k in self.ckpt.index if k.split(".")[-2:] == ["lm_head", "weight"])
        if "lm_head.weight" in heads:
            name = "lm_head.weight"
        elif len(heads) == 1:
            name = heads[0]
        elif heads:
            raise SystemExit(f"{self.ckpt.dir}: several LM head tensors {heads}")
        else:
            name = None
            if self.untied():
                raise SystemExit(
                    f"{self.ckpt.dir}: the config says the LM head is untied "
                    "(tie_word_embeddings false), but the checkpoint holds no lm_head tensor"
                )
        self.head_name = name or f"{self.fam.prefix}{self.fam.embed}.weight"
        return nw, self.ckpt.get(self.head_name).float()

    def untied(self):
        """Whether config.json, its text_config or the loaded config says tie_word_embeddings
        false."""
        try:
            c = json.loads((self.ckpt.dir / "config.json").read_text())
        except (OSError, ValueError):
            c = {}
        says = (
            c.get("tie_word_embeddings"),
            (c.get("text_config") or {}).get("tie_word_embeddings"),
            getattr(self.hf, "tie_word_embeddings", None),
        )
        return any(s is False for s in says)

    def raw_layer(self, i):
        """Layer i's checkpoint tensors ('layers.i....') without those its module does not take."""
        drop = tuple(f"layers.{i}.{u}" for u in self.fam.unused(self.hf, i))
        return {k: v for k, v in self.ckpt.raw_layer(i).items() if not k.startswith(drop)}

    def gather(self, name, ids):
        """Rows `ids` of a checkpoint table, [len(ids), width] (each distinct row read once)."""
        from safetensors import safe_open

        uniq = sorted(set(ids))
        where = {t: j for j, t in enumerate(uniq)}
        with torch.device("cpu"):
            with safe_open(str(self.ckpt.dir / self.ckpt.index[name]), framework="pt") as f:
                table = f.get_slice(name)
                rows = torch.cat([table[t : t + 1] for t in uniq])
            rows = rows[torch.tensor([where[t] for t in ids])]
        return rows.to(torch.get_default_device())

    def contexts(self, seqs, embeds):
        """Per sequence, what its layers share beyond the residual stream: the KV-sharing layers'
        keys and values (filled as the layers run) and the per-layer inputs (Gemma 4 E2B/E4B)."""
        return [{"kv": {}, "ple": p} for p in self.fam.per_layer_inputs(self, seqs, embeds)]

    @staticmethod
    def at(c, i):
        """A sequence's context for layer i."""
        return {"kv": c["kv"], "ple": None if c["ple"] is None else c["ple"][:, :, i]}

    def build(self, i, sd):
        """Decoder layer i (float32) from a state dict in text-model names ('layers.i....')."""
        m = self.fam.layer(self.hf, i)
        p = f"layers.{i}."
        m.load_state_dict({k[len(p) :]: v for k, v in sd.items()}, assign=True, strict=True)
        return m.eval()


# --- calibration --------------------------------------------------------------------------------


def pack(store):
    """Each square float matrix of a Hessian store as its upper triangle, row by row."""
    out = {}
    for k, v in store.items():
        if (
            torch.is_tensor(v)
            and v.is_floating_point()
            and v.dim() >= 2
            and v.shape[-1] == v.shape[-2] > 1
        ):
            n = v.shape[-1]
            r, c = torch.triu_indices(n, n, device=v.device)
            rows = v.reshape(-1, n, n)[:, r, c].reshape(-1)
            out[k] = {"triu": n, "shape": tuple(v.shape), "rows": rows}
        else:
            out[k] = v
    return out


def unpack(store):
    """pack()'s inverse: every packed matrix made symmetric from its upper triangle."""
    out = {}
    for k, v in store.items():
        if isinstance(v, dict) and "triu" in v:
            n, rows, per = v["triu"], v["rows"], v["triu"] * (v["triu"] + 1) // 2
            r, c = torch.triu_indices(n, n, device=rows.device)
            m = torch.zeros(rows.numel() // per, n, n, dtype=rows.dtype, device=rows.device)
            m[:, r, c] = rows.reshape(-1, per)
            m += m.triu(1).transpose(1, 2)
            out[k] = m.reshape(v["shape"])
        else:
            out[k] = v
    return out


def symmetric(store):
    """The Hessian store as GPTQ reads it: every matrix taken from its upper triangle."""
    return unpack(pack(store))


def plain_norm(x, eps=1e-6):
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def patch(m, record=None):
    """Every projection of layer m a bf16 matmul with float32 accumulation (a bias added in
    float32 after it); record(name, x) gets every projection's input."""
    for n, lin in list(m.named_modules()):
        if isinstance(lin, torch.nn.Linear):
            wb = lin.weight.data.bfloat16()
            bias = None if lin.bias is None else lin.bias.data.float()

            def fwd(x, wb=wb, n=n, bias=bias):
                if record is not None:
                    record(n, x)
                y = torch.nn.functional.linear(x.bfloat16(), wb).float()
                return y if bias is None else y + bias

            lin.forward = fwd
            lin.weight = None
    return m


class Recorder:
    """The Hessians of one layer's projection inputs that are not a normed residual."""

    def __init__(self, fam):
        self.h = {}
        self.site_of = {n: p.site for n, p in fam.linears.items() if not fam.site_norm(p.site)}

    def __call__(self, name, x):
        site = self.site_of.get(name)
        if site is not None:
            if site not in self.h:
                self.h[site] = Hessian(x.shape[-1])
            self.h[site].add(x)

    def store(self):
        return {k: v.get() for k, v in self.h.items()}


# --- conversion ---------------------------------------------------------------------------------


def convert_layer(md, v, i, raw, store, log, method="exp", bfp8="rtn"):
    """Layer i converted: {name: float32 tensor}. store: the layer's Hessians (GPTQ only)."""
    fam, p = md.fam, f"layers.{i}."
    sd = {k: x.float() for k, x in raw.items()}
    preps, hs = {}, {}

    def site_h(site):
        if site not in hs:
            h = store[site].float()
            norm = fam.site_norm(site)
            if norm:
                d = fam.norm_offset + raw[p + norm + ".weight"].float()
                h = d[:, None] * h * d[None, :]
            hs[site] = h
        return hs[site]

    def prep_of(site):
        if site not in preps:
            preps[site] = gptq_prep(site_h(site), True, v.damp)
        return preps[site]

    def rounded(name, w, fmt, site):
        if v.gptq and (fmt == "bfp4" or (fmt == "bfp8" and bfp8 == "gptq")) and store is not None:
            q = round_proj(w, fmt, prep_of(site), method)
        elif fmt == "bfp8" and bfp8 == "exp":
            q = bfp_search_weight(w, 7)
        else:
            q = quantize(w, fmt)
        log[name] = (fmt, rel(w, q))
        return q

    for n, pr in fam.linears.items():
        k = p + n + ".weight"
        if k not in sd:
            continue
        sd[k] = rounded(n, sd[k], v.of(pr.group, i), pr.site)
    return sd


def rounding_record(v, method, bfp8):
    """How the export's projections were rounded, for the manifest: `exponent_search` (GPTQ
    searched each bfloat4_b block's exponent), `rounding.method` (rtn, or under GPTQ `gptq` or
    `exp`), `rounding.damp`, `rounding.formats` (what the method changes), the exponent offsets
    tried, and `rounding.bfp8` when bfloat8_b projections were not rounded to nearest."""
    method = method if v.gptq else "rtn"
    four = "bfp4" in v.fmt.values() or any(f == "bfp4" for _, _, f, _ in v.layers)
    search = v.gptq and four and method.startswith("exp")
    rec = {"method": method}
    if v.gptq:
        rec["damp"] = v.damp
    if method not in ("rtn", "gptq"):
        rec["formats"] = ["bfp4"]
    if search:
        rec["exponent_offsets"] = list(EXP_OFFSETS)
    eight = "bfp8" in v.fmt.values() or any(f == "bfp8" for _, _, f, _ in v.layers)
    if eight:
        b8 = bfp8 if (bfp8 != "gptq" or v.gptq) else "rtn"
        if b8 != "rtn":
            rec["bfp8"] = b8
    return {"exponent_search": search, "rounding": rec}


# --- writing ------------------------------------------------------------------------------------


def write_tensors(path, tensors):
    from safetensors.torch import save_file

    save_file({k: v.contiguous() for k, v in tensors.items()}, str(path))


def matrix(name, w, fmt):
    """{tensor name: bf16 tensor} of one rounded matrix, checked to lie on its grid."""
    kind, bits, _ = parse_fmt(fmt)
    b = w.to(torch.bfloat16)
    if kind == "bfp":
        assert torch.equal(bfp_weight(b.float(), bits), b.float()), f"{name}: not on the grid"
    return {name + ".weight": b}


def save_layer(md, v, d, i, sd, raw, manifest):
    """layers-NN.safetensors: the rounded matrices and the layer's other tensors as stored."""
    p, pre, t, fmts = f"layers.{i}.", md.fam.prefix, {}, {}
    for n, pr in md.fam.linears.items():
        k = p + n + ".weight"
        if k not in sd:
            continue
        fmt = v.of(pr.group, i)
        fmts[n] = {"format": fmt, "online": False}
        t.update(matrix(pre + p + n, sd[k], fmt))
    for k in sd:
        if pre + k in t or any(pre + k == pre + p + n + ".weight" for n in md.fam.linears):
            continue
        t[pre + k] = raw[k]
    write_tensors(d / f"layers-{i:02d}.safetensors", t)
    manifest["layers"][str(i)] = fmts


def save_head(md, v, d, nw, w, manifest, manifest_format):
    """head.safetensors, rest.safetensors, the index, the config and tokenizer copies."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    t = matrix("lm_head", w, v.of("lm_head"))
    t[f"{md.fam.prefix}{md.fam.final_norm}.weight"] = nw.to(torch.bfloat16)
    write_tensors(d / "head.safetensors", t)
    index = {}
    own = sorted(x for x in d.glob("*.safetensors") if not x.is_symlink())
    for f in own:
        with safe_open(str(f), framework="pt") as g:
            for k in g.keys():  # noqa: SIM118 (a safe_open handle, not a dict)
                index[k] = f.name
    # every tensor of the original the export does not hold, copied into rest.safetensors
    names = {f.name for f in own}
    for k, shard in md.ckpt.index.items():
        if k not in index:
            assert shard not in names, f"{shard}: the original has a shard of an export's name"
            index[k] = shard
    linked = sorted({s for s in index.values() if s not in names})
    if linked:
        rest = {}
        for s in linked:
            with safe_open(str(md.ckpt.dir / s), framework="pt") as h:
                for k in [k for k, x in index.items() if x == s]:
                    rest[k] = h.get_tensor(k)
        save_file(rest, str(d / "rest.safetensors.tmp"), metadata={"format": "pt"})
        (d / "rest.safetensors.tmp").rename(d / "rest.safetensors")
        for k in rest:
            index[k] = "rest.safetensors"
    (d / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"quant4": v.tag()}, "weight_map": index}, indent=1)
    )
    for f in (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "generation_config.json",
        "chat_template.jinja",
    ):
        if (md.ckpt.dir / f).exists():
            shutil.copy(md.ckpt.dir / f, d / f)
    manifest["lm_head"] = v.of("lm_head")
    if manifest_format >= 2:
        manifest["lm_head_tensor"] = md.head_name


# --- the export ---------------------------------------------------------------------------------


def set_tf32(device, on):
    """TF32 for float32 matmuls on a CUDA device (the published exports were made with it on)."""
    if not str(device).startswith("cuda") or not on:
        return
    m = torch.backends.cuda.matmul
    if hasattr(m, "fp32_precision"):
        m.fp32_precision = "tf32"
    else:
        m.allow_tf32 = True


def export(
    cfg,
    model_dir,
    out_dir,
    spec,
    device="cpu",
    data_dir=None,
    method="exp",
    bfp8="rtn",
    manifest_format=2,
    tf32=True,
):
    """Write the export of variant `spec` of the checkpoint in model_dir to out_dir."""
    assert method in ("exp", "gptq"), f"method {method!r}: exp or gptq"
    assert bfp8 in ("rtn", "exp", "gptq"), f"bfp8 {bfp8!r}: rtn, exp or gptq"
    torch.set_grad_enabled(False)
    set_tf32(device, tf32)
    torch.set_default_device(device)
    md = Model(cfg, model_dir, device, data_dir)
    v = parse(spec, md.fam, cfg["quant"]["damp"])
    d = Path(out_dir)
    if d.exists() and any(d.iterdir()):
        raise SystemExit(f"{d}: the output directory is not empty")
    d.mkdir(parents=True, exist_ok=True)
    name = cfg["model"]["name"]
    man = {"layers": {}, "online_signs": None}
    t0 = time.time()
    cal = cctx = None
    if v.gptq:
        calib = md.calib_tokens()
        n_tok = sum(len(t) for t in calib)
        print(f"{name}: calibration {len(calib)} x {len(calib[0])} = {n_tok} tokens", flush=True)
        cal = [md.embed(t) for t in calib]
        cctx = md.contexts(calib, cal)
    for i in range(md.n_layers):
        raw = md.raw_layer(i)
        store = None
        if v.gptq:
            rec = Recorder(md.fam)
            m = patch(md.build(i, {k: x.float() for k, x in raw.items()}), record=rec)
            h_in, h_mlp = Hessian(md.dim), Hessian(md.dim)
            new = []
            for x, c in zip(cal, cctx, strict=True):
                p = md.fam.run_layer(m, x, md.rope(x.shape[1], i), parts=True, ctx=md.at(c, i))
                h_in.add(plain_norm(x))
                h_mlp.add(plain_norm(p[2]))
                new.append(p[5])
            cal = new
            store = symmetric({"in": h_in.get(), "mlp": h_mlp.get(), **rec.store(), "n": n_tok})
            del m, rec, h_in, h_mlp
        log = {}
        sd = convert_layer(md, v, i, raw, store, log, method, bfp8)
        del store
        save_layer(md, v, d, i, sd, raw, man)
        man["variant"], man["model"] = v.tag(), name
        man["online_block"] = 0
        man.update(rounding_record(v, method, bfp8))
        del sd, raw
        worst = max((e for _, e in log.values()), default=0.0)
        print(f"layer {i}: worst relative error {worst:.4f}, {time.time() - t0:.0f} s", flush=True)
    nw, w = md.head()
    fmt = v.of("lm_head")
    if v.gptq and fmt == "bfp4":
        hh = Hessian(md.dim)
        for x in cal:
            hh.add(plain_norm(x))
        hh = symmetric({"head": hh.get()})["head"]
        dn = md.fam.norm_offset + nw
        w = gptq(w, dn[:, None] * hh * dn[None, :], fmt, damp=v.damp)[0]
    else:
        w = quantize(w, fmt)
    del cal, cctx
    save_head(md, v, d, nw, w, man, manifest_format)
    (d / MANIFEST).write_text(json.dumps(man))
    print(f"{d}: {v.tag()}, {time.time() - t0:.0f} s", flush=True)
    return d


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True, help="the model's TOML config")
    ap.add_argument("--model", required=True, help="the original checkpoint's directory")
    ap.add_argument("--out", required=True, help="the export's directory (new or empty)")
    ap.add_argument(
        "--variant", help="the variant spec (default: the config's only [quant] variant)"
    )
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--data", help="the directory of the calibration texts (tensix_bfp.data)")
    ap.add_argument(
        "--method",
        default="exp",
        choices=["exp", "gptq"],
        help="bfloat4_b GPTQ with the exponent search (exp, default) or plain GPTQ",
    )
    ap.add_argument(
        "--bfp8",
        default="rtn",
        choices=["rtn", "exp", "gptq"],
        help="bfloat8_b projections: to nearest (default), to nearest with the exponent search, "
        "or GPTQ (with a gptq variant)",
    )
    ap.add_argument(
        "--manifest-format",
        type=int,
        default=2,
        choices=[1, 2],
        help="2 (default) records the LM head's source tensor in quant4.json; 1 leaves that key "
        "out, as in the earlier published exports",
    )
    ap.add_argument(
        "--no-tf32",
        action="store_true",
        help="do not enable TF32 float32 matmuls on CUDA (the published exports had them on)",
    )
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    spec = a.variant
    if spec is None:
        vs = cfg["quant"]["variants"]
        if len(vs) != 1:
            raise SystemExit(
                f"{a.config} lists {len(vs)} variants; choose one with --variant: {vs}"
            )
        spec = vs[0]
    export(
        cfg,
        a.model,
        a.out,
        spec,
        a.device,
        a.data,
        a.method,
        a.bfp8,
        a.manifest_format,
        not a.no_tf32,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
