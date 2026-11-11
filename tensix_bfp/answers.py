# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Xylolabs
"""The answer check: fixed prompts with known answers (answers.json), answered greedily by the
rounded model and by its bf16 original. The rounded model fails only where it gets wrong an
answer the bf16 model gets right, or where its text is broken on an item the bf16 model gets
right. KL, top-1 and perplexity are reported elsewhere and decide nothing here.

  run MODEL_DIR --tokenizer BF16_DIR --out RUN.json [--device cuda:0] [--thinking]
      greedy answers with transformers in bf16, of an export or of the bf16 checkpoint itself
  prompts --tokenizer BF16_DIR --out PROMPTS.json [--thinking]
      each item's prompt as token ids after the chat template, with max_new and the stop ids,
      for another runner; its answers go into a run file of the same form
      ({"model", "runner", "items": [{"id", "text", "tokens", "margins"}]})
  score RUN.json [--ref BF16_RUN.json] [--think RUN_T.json --think-ref BF16_RUN_T.json]
      per item right or wrong and broken or not; with --ref the verdict (exit status 0 for a
      pass, 1 for a fail)

Checks (check()): "number" (an expected value appears as a number, thousands separators
removed), "regex" (case-insensitive unless "case"), "exact" (the whole answer, quotes and end
punctuation stripped), "lines" (the non-empty lines, bullets and end punctuation stripped,
lowercase), "python" (the answer's code with the item's assertions in a separate interpreter,
python -I, 10 s, its own temporary directory, an empty environment).

Broken text (broken()): empty; U+FFFD (except one as the very last character at the token
limit); control characters; a loop (a piece of 6+ characters 5 times in a row, or of 1-5
characters 20 times); the wrong script (more than a fifth of the letters neither Latin, Hangul
nor Han/kana; for a Korean answer less than half Hangul; for an English one more than a fifth
Hangul/Han/kana; 8 letters at least; bracketed text without Hangul left out of the share).

Lost answers (verdict()): a lost answer counts only where the bf16 run was clear at the first
new token where the two runs part, a top-2 logit gap of at least the set's clear_margin; below it
the item is a near tie, listed and not counted. With thinking-on runs of both models, a lost
reasoning item is pardoned when the rounded model answers it right with thinking on, or when the
bf16 model gets it wrong then too. Thinking is stripped before the checks: a closed <think> block
goes, an unclosed one leaves no answer (wrong, not broken).
"""

import argparse
import datetime
import gc
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import unicodedata
from collections import Counter
from pathlib import Path

SET = Path(__file__).resolve().parent / "answers.json"


def load_set(path=SET):
    s = json.loads(Path(path).read_text(encoding="utf-8"))
    ids = [it["id"] for it in s["items"]]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: an id appears twice")
    return s


# --- the answer's text --------------------------------------------------------------------------

SPECIAL = re.compile(
    r"<\|[^<>|\s]{1,40}\|>|<\|[A-Za-z_]{1,30}>|<[A-Za-z_]{1,30}\|>|</?s>|<(?:[eb]os|pad|end_of_turn|start_of_turn)>"
)
# closing markers of a thinking block: Qwen's </think>, Gemma 4's <channel|>, and a chat turn that
# ends a think turn and opens the answer turn
THINK_CLOSE = ("</think>", "<channel|>", "<|im_end|>\n<|im_start|>assistant\n")
THINK_OPEN = ("<think>", "<|channel>")


def answer_text(raw, thinking=False):
    """(the answer without thinking or special tokens, whether an answer exists). Thinking is cut
    up to its last closing marker; thinking still open at the end, or with thinking on no closing
    marker at all, leaves no answer. In harmony-style replies only the last message for the user
    counts."""
    t = raw
    if "<|message|>" in t:
        keep = []
        for seg in t.split("<|start|>"):
            if "<|message|>" not in seg:
                continue
            head, body = seg.split("<|message|>", 1)
            if "analysis" in head or "commentary" in head or "to=self" in head:
                continue
            keep.append(body)
        if not keep:
            return "", False
        t = keep[-1]
    else:
        close = max((t.rfind(c) + len(c) if c in t else -1) for c in THINK_CLOSE)
        if close >= 0:
            t = t[close:]
        elif thinking or any(o in t for o in THINK_OPEN):
            return "", False
    if any(o in t for o in THINK_OPEN):
        return "", False
    t = SPECIAL.sub("", t)
    return t.strip(), True


# --- broken text --------------------------------------------------------------------------------


def script_of(ch):
    cp = ord(ch)
    if 0xAC00 <= cp <= 0xD7A3 or 0x1100 <= cp <= 0x11FF or 0x3130 <= cp <= 0x318F:
        return "hangul"
    if cp < 0x250:
        return "latin"
    if 0x3400 <= cp <= 0x9FFF or 0x3040 <= cp <= 0x30FF or 0xF900 <= cp <= 0xFAFF:
        return "cjk"
    return "other"


def scripts(text):
    return Counter(script_of(c) for c in text if c.isalpha())


BRACKETED = re.compile("[(（\\[][^()（）\\[\\]가-힣]*[)）\\]]")
LOOP_LONG = re.compile(r"(.{6,}?)\1{4,}", re.DOTALL)
LOOP_SHORT = re.compile(r"(.{1,5}?)\1{19,}", re.DOTALL)


def broken(text, out, at_limit=False):
    """Why the answer's text is broken, or None. out: the language the answer is due in (en,
    ko, code). at_limit: the run stopped at its new-token limit, where a lone U+FFFD as the very
    last character is a character cut in half, not broken text."""
    if not text.strip():
        return "empty"
    if "�" in text and not (at_limit and text.count("�") == 1 and text.endswith("�")):
        return "U+FFFD in the text"
    if any(unicodedata.category(c) == "Cc" and c not in "\n\t\r" for c in text):
        return "control characters"
    m = LOOP_LONG.search(text) or LOOP_SHORT.search(text)
    if m:
        return f"a loop: {m.group(1)[:30]!r} repeated"
    sc = scripts(BRACKETED.sub("", text))
    n = sum(sc.values())
    if n >= 8:
        if sc["other"] / n > 0.2:
            return f"wrong script: {sc['other']} of {n} letters neither Latin, Hangul nor Han/kana"
        if out == "ko" and sc["hangul"] / n < 0.5:
            return (
                f"wrong language: {sc['hangul']} of {n} letters Hangul, the answer is due in Korean"
            )
        if out == "en" and (sc["hangul"] + sc["cjk"]) / n > 0.2:
            return f"wrong language: {sc['hangul'] + sc['cjk']} of {n} letters Hangul/Han/kana, the answer is due in English"
        if out == "code" and sc["cjk"] / n > 0.2:
            return f"wrong script: {sc['cjk']} of {n} letters Han/kana in code"
    return None


# --- right or wrong -----------------------------------------------------------------------------

NUMBER = re.compile(r"(?<![\d.])-?\d+(?:\.\d+)?(?![\d])")
STRIP = " \t\n\"'`*_.,!?。、:;「」『』“”‘’()[]"


def numbers(text):
    t = unicodedata.normalize("NFKC", text)
    t = re.sub(r"(?<=\d)[,٬](?=\d{3}(?!\d))", "", t)
    return [float(x) for x in NUMBER.findall(t)]


def code_of(text):
    blocks = re.findall(r"```[\w+-]*\n(.*?)```", text, re.DOTALL)
    for b in blocks:
        if "def " in b:
            return b
    if blocks:
        return blocks[0]
    # no complete block: an unpaired fence is not part of the code
    return "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("```"))


def run_python(code, tests, timeout=10):
    prog = (
        code
        + "\n\n_r = ["
        + ", ".join(f"bool({t})" for t in tests)
        + "]\nprint('ANSWER-CHECK', all(_r))\n"
    )
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "answer.py"
        p.write_text(prog, encoding="utf-8")
        try:
            r = subprocess.run(
                [sys.executable, "-I", str(p)],
                check=False,
                cwd=d,
                env={"PATH": "/usr/bin:/bin"},
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return False
    return r.returncode == 0 and "ANSWER-CHECK True" in r.stdout


def check(item, text):
    c = item["check"]
    k = c["type"]
    if k == "number":
        got = numbers(text)
        return any(abs(g - v) < 1e-9 for g in got for v in c["values"])
    if k == "regex":
        return (
            re.search(c["pattern"], text, 0 if c.get("case") else re.IGNORECASE | re.MULTILINE)
            is not None
        )
    if k == "exact":
        a, b = text.strip(STRIP), c["text"]
        return a == b if c.get("case") else a.casefold() == b.casefold()
    if k == "lines":
        got = [
            re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", x).strip(STRIP).casefold()
            for x in text.splitlines()
        ]
        return [x for x in got if x] == [x.casefold() for x in c["lines"]]
    if k == "python":
        return run_python(code_of(text), c["tests"])
    raise ValueError(f"{item['id']}: unknown check {k}")


def score_item(item, raw, thinking=False, at_limit=False):
    text, has = answer_text(raw, thinking)
    out = item.get("out", item["lang"])
    why = broken(text, out, at_limit) if has else None
    right = has and why is None and check(item, text)
    return {"id": item["id"], "right": bool(right), "broken": why, "answer": has}


def select(s, thinking=False):
    """The set's items a run holds: all of them, or with thinking on the reasoning items only."""
    return [it for it in s["items"] if not thinking or it["kind"] == "reasoning"]


# mixed script, information only: a Hangul syllable with a Latin letter glued after it, Han or kana
# beside Hangul, or Cyrillic, Greek, Arabic, Thai or Devanagari anywhere
MIXED_SCRIPT = re.compile(
    "[가-힣][A-Za-z]|[A-Za-z][가-힣][A-Za-z]|[가-힣][぀-ヿ一-鿿]|[぀-ヿ一-鿿][가-힣]|[Ѐ-ӿͰ-Ͽ؀-ۿ฀-๿ऀ-ॿ]"
)


def mixed_script(run, s=None):
    """{item id: the pieces} of a run's answers in mixed script, code items left out."""
    s = s or load_set()
    items = {it["id"]: it for it in select(s, run.get("thinking", False))}
    out = {}
    for r in run["items"]:
        it = items.get(r["id"])
        if it is None or it["kind"] == "code" or it.get("out") == "code":
            continue
        text, has = answer_text(r["text"], run.get("thinking", False))
        found = [m.group(0) for m in MIXED_SCRIPT.finditer(text)] if has else []
        if found:
            out[r["id"]] = found[:4]
    return out


def stopped_at_limit(run, item, got, s):
    """Whether this item's run used all the new tokens the set gives it (False when that cannot
    be told: no tokens recorded, or the run's tokenizer directory is not here)."""
    tok = run.get("tokenizer")
    if "tokens" not in got or not tok or not (Path(tok) / "config.json").exists():
        return False
    return len(got["tokens"]) >= max_new_of(item, tok, s, run.get("thinking", False))


def score_run(run, s=None):
    s = s or load_set()
    items = {it["id"]: it for it in select(s, run.get("thinking", False))}
    got = {r["id"]: r for r in run["items"]}
    missing = [i for i in items if i not in got]
    th = run.get("thinking", False)
    res = {
        i: score_item(items[i], got[i]["text"], th, stopped_at_limit(run, items[i], got[i], s))
        for i in items
        if i in got
    }
    return res, missing


def verdict(res, ref, run=None, ref_run=None, margin=None, think=None):
    """{"pass", "regressions", "near_ties", "cleared", "broken", "gains", "ref_broken", ...} of a
    run's scores against the bf16 run's (the rules in the module's help). Without margins every
    lost answer counts. Where the bf16 model gets none of the reasoning items right with thinking
    on (a model without a thinking mode), the thinking-on runs pardon nothing (`think_na`)."""
    lost = [i for i in res if ref.get(i, {}).get("right") and not res[i]["right"]]
    near, reg = [], []
    for i in lost:
        d = (
            divergence(run[i], ref_run[i])
            if run and ref_run and i in run and i in ref_run
            else None
        )
        if margin is not None and d and d[1] is not None and d[1] < margin:
            near.append(i)
        else:
            reg.append(i)
    cleared = []
    think_na = (
        think is not None and bool(think[1]) and not any(r["right"] for r in think[1].values())
    )
    if think is not None and not think_na:
        tres, tref = think
        for i in list(reg):
            if i in tref and (tres.get(i, {}).get("right") or not tref[i]["right"]):
                reg.remove(i)
                cleared.append(i)
    brk = [i for i in res if res[i]["broken"] and ref.get(i, {"right": True})["right"]]
    brk_free = [i for i in res if res[i]["broken"] and i not in brk]
    gains = [i for i in res if res[i]["right"] and i in ref and not ref[i]["right"]]
    return {
        "pass": not reg and not brk,
        "regressions": reg,
        "near_ties": near,
        "cleared": cleared,
        "broken": brk,
        "broken_ref_wrong": brk_free,
        "gains": gains,
        "ref_broken": [i for i in ref if ref[i]["broken"]],
        "counted": len([i for i in res if ref.get(i, {}).get("right")]) - len(near),
        "think_na": think_na,
    }


# --- that a run loaded the export ---------------------------------------------------------------


def tensor_names(path):
    """A safetensors file's tensor names, from its header alone."""
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        head = json.loads(f.read(n))
    head.pop("__metadata__", None)
    return set(head)


def weight_map(model_dir):
    """A checkpoint's tensor name -> file, from its index or its one file."""
    d = Path(model_dir)
    ix = d / "model.safetensors.index.json"
    if ix.exists():
        return json.loads(ix.read_text())["weight_map"]
    return dict.fromkeys(tensor_names(d / "model.safetensors"), "model.safetensors")


def export_files_clean(model_dir):
    """Refuse an export that would not load as its index says: transformers prefers a
    model.safetensors to an index and reads every listed file whole, later files overwriting
    earlier ones."""
    d = Path(model_dir)
    if (d / "model.safetensors").exists():
        raise SystemExit(f"{d}: a model.safetensors beside the export: it would load instead")
    if not (d / "model.safetensors.index.json").exists():
        raise SystemExit(f"{d}: an export without an index")
    wm = weight_map(d)
    gone = sorted({f for f in wm.values() if not (d / f).exists()})
    if gone:
        raise SystemExit(f"{d}: files the index names are missing: {gone[:5]}")
    wrong = {}
    for f in sorted(d.glob("*.safetensors")):
        over = [k for k in tensor_names(f) if wm.get(k) != f.name]
        if over:
            wrong[f.name] = over
    if wrong:
        some = {f: sorted(k)[:3] for f, k in wrong.items()}
        raise SystemExit(
            f"{d}: files hold tensors the index names elsewhere, which would overwrite them: {some}"
        )
    return wm


EXPERT_KEY = re.compile(r"(.*\.experts)\.(\d+)\.(\w+)\.weight")
ALLOWED_MISSING = re.compile(
    r"(^|\.)(mtp|vision|visual|audio|multi_modal_projector|embed_vision|embed_audio)[._]"
)


def loaded_tensor(k, sd):
    """A checkpoint tensor's value in a loaded model's state dict, or None (a text-only class
    drops language_model from the names; a class may fuse per-expert matrices on load)."""
    for c in (
        k,
        k.replace("model.language_model.", "model."),
        k.replace("language_model.", ""),
        k.replace("backbone.", "model.", 1),
    ):
        if c in sd:
            return sd[c]
        m = EXPERT_KEY.fullmatch(c)
        if m and f"{m[1]}.{m[3]}" in sd:
            return sd[f"{m[1]}.{m[3]}"][int(m[2])]
        if m and m[3] in ("gate_proj", "up_proj") and f"{m[1]}.gate_up_proj" in sd:
            gu = sd[f"{m[1]}.gate_up_proj"][int(m[2])]
            half = gu.shape[0] // 2
            return gu[:half] if m[3] == "gate_proj" else gu[half:]
    return None


def weights_check(sd, model_dir, tok_dir):
    """Refuse a load that did not give the export's values: every tensor of the loaded state dict
    must equal the file the index names, and some must differ from the bf16 checkpoint's. Returns
    the record kept in the run file."""
    import torch
    from safetensors import safe_open

    d, tok = Path(model_dir), Path(tok_dir)
    wm, ref = weight_map(d), weight_map(tok)
    by_file = {}
    for k, f in wm.items():
        by_file.setdefault(f, []).append(k)
    compared, absent, differ, wrong = 0, [], 0, []
    # rest.safetensors (what the variant left alone) last: the bf16 test stops at three differences
    for f in sorted(by_file, key=lambda f: (f == "rest.safetensors", f)):
        with safe_open(str(d / f), framework="pt") as h:
            for k in by_file[f]:
                to_ref = differ < 3 and k in ref
                got = None if sd is None else loaded_tensor(k, sd)
                if sd is not None and got is None:
                    absent.append(k)
                    continue
                if got is None and not to_ref:
                    continue
                v = h.get_tensor(k)
                if got is not None:
                    got = got.detach().cpu()
                    if not torch.equal(got, v.to(got.dtype)):
                        wrong.append(k)
                compared += 1
                if to_ref:
                    with safe_open(str(tok / ref[k]), framework="pt") as o:
                        differ += not torch.equal(o.get_tensor(k).to(v.dtype), v)
    if wrong:
        raise SystemExit(f"{d}: {len(wrong)} loaded tensors differ from the export's: {wrong[:5]}")
    unknown = [k for k in absent if not ALLOWED_MISSING.search(k)]
    if not compared or len(unknown) > compared:
        raise SystemExit(
            f"{d}: {compared} tensors compared, {len(unknown)} not found in the loaded model: "
            f"{sorted(unknown)[:5]}"
        )
    if not differ:
        raise SystemExit(f"{d}: the {compared} tensors compared all equal the bf16 checkpoint's")
    rec = {
        "compared": compared,
        "not_in_model": len(absent),
        "differ_from_bf16": differ,
        "loaded": "state dict" if sd is not None else "by the index (the loader's own reads)",
    }
    print(f"answers: weights check {rec}", file=sys.stderr, flush=True)
    return rec


def load_problem(run, ref_run):
    """Why a host run of an export says nothing about that export, or None."""
    if not run.get("runner", "").startswith("host ") or run.get("model") == ref_run.get("model"):
        return None
    if not run.get("weights_check"):
        return "no weights check in this run: nothing shows it loaded the export"
    by = {x["id"]: x for x in ref_run["items"]}
    same = [
        x
        for x in run["items"]
        if x.get("margins") and by.get(x["id"], {}).get("margins") == x["margins"]
    ]
    if run["items"] and len(same) == len(run["items"]):
        return "margins bit-identical to bf16's on every item: bf16 was loaded"
    return None


# --- prompts and generation ---------------------------------------------------------------------


def arch_of(tok_dir):
    cfg = Path(tok_dir) / "config.json"
    return json.loads(cfg.read_text()).get("architectures", [""])[0] if cfg.exists() else ""


def assistant_prefix(tok_dir, s):
    """Text after the chat template's generation prompt for templates without a thinking switch
    (answers.json "assistant_prefix", by architecture)."""
    return s.get("assistant_prefix", {}).get(arch_of(tok_dir), "")


def chat_kwargs(tok_dir, s, thinking=False):
    """The template's variables: the set's "chat", with "chat_by_arch" over it, and strftime_now
    pinned to current_date so that a template printing the date gives the same prompt every day.
    With thinking: enable_thinking on and "chat_think_by_arch" instead of "chat_by_arch"."""
    if thinking:
        arch = s.get("chat_think_by_arch", {}).get(arch_of(tok_dir), {})
        kw = {**s.get("chat", {}), "enable_thinking": True, **arch}
    else:
        kw = {**s.get("chat", {}), **s.get("chat_by_arch", {}).get(arch_of(tok_dir), {})}
    if "current_date" in kw:
        day = datetime.date.fromisoformat(kw["current_date"])
        kw["strftime_now"] = day.strftime
    return kw


def max_new_of(item, tok_dir, s, thinking=False):
    if thinking:
        return s["max_new_think"]
    return item.get("max_new") or s.get("max_new_by_arch", {}).get(arch_of(tok_dir)) or s["max_new"]


def chat_ids(tok, item, s, prefix="", kw=None):
    msg = [{"role": "user", "content": item["prompt"]}]
    kw = s.get("chat", {}) if kw is None else kw
    text = tok.apply_chat_template(msg, add_generation_prompt=True, tokenize=False, **kw)
    return list(tok(text + prefix, add_special_tokens=False)["input_ids"])


def stop_ids(tok, model_dir):
    """The end-of-sequence ids of generation_config.json and the tokenizer, and each of the
    config's stop_strings that is one token."""
    ids = set()
    gcf = Path(model_dir) / "generation_config.json"
    if gcf.exists():
        g = json.loads(gcf.read_text())
        e = g.get("eos_token_id")
        ids |= set(e if isinstance(e, list) else [e] if e is not None else [])
        for st in g.get("stop_strings") or []:
            one = tok(st, add_special_tokens=False)["input_ids"]
            if len(one) == 1:
                ids.add(one[0])
    if tok.eos_token_id is not None:
        ids.add(tok.eos_token_id)
    return sorted(ids)


def decode(tok, ids):
    """The new tokens as text, special tokens kept (answer_text() strips them)."""
    return tok.decode(ids, skip_special_tokens=False)


def prompts_file(tok_dir, s=None, thinking=False):
    """Each item's prompt as token ids after the chat template (with thinking: the reasoning
    items only, enable_thinking on, max_new_think new tokens)."""
    from transformers import AutoTokenizer

    s = s or load_set()
    tok = AutoTokenizer.from_pretrained(tok_dir)
    prefix = "" if thinking else assistant_prefix(tok_dir, s)
    kw = chat_kwargs(tok_dir, s, thinking)
    return {
        "set_version": s["version"],
        "tokenizer": str(tok_dir),
        "thinking": thinking,
        "assistant_prefix": prefix,
        "stop": stop_ids(tok, tok_dir),
        "items": [
            {
                "id": it["id"],
                "ids": chat_ids(tok, it, s, prefix, kw),
                "max_new": max_new_of(it, tok_dir, s, thinking),
            }
            for it in select(s, thinking)
        ],
    }


def run_items(generate, tok_dir, model, runner, s=None, thinking=False):
    """A runner's hook: generate(ids, max_new, stop) -> the new token ids (greedy, ending at a
    stop id or after max_new), or (ids, margins) with per new token [the logit gap of its top
    two, the second token]. Returns the run file's dict."""
    from transformers import AutoTokenizer

    s = s or load_set()
    tok = AutoTokenizer.from_pretrained(tok_dir)
    p = prompts_file(tok_dir, s, thinking)
    items = []
    for it in p["items"]:
        t0 = time.time()
        g = generate(it["ids"], it["max_new"], p["stop"])
        new, margins = (list(g[0]), g[1]) if isinstance(g, tuple) else (list(g), None)
        cut = [k for k, t in enumerate(new) if t in p["stop"]]
        new = new[: cut[0]] if cut else new
        item = {
            "id": it["id"],
            "text": decode(tok, new),
            "tokens": new,
            "s": round(time.time() - t0, 2),
        }
        if margins is not None:
            item["margins"] = [[round(float(m), 3), int(alt)] for m, alt in margins[: len(new) + 1]]
        items.append(item)
        print(f"{it['id']}: {items[-1]['text'][:80]!r}", flush=True)
    return {
        "set_version": s["version"],
        "model": str(model),
        "tokenizer": str(tok_dir),
        "runner": runner,
        "thinking": thinking,
        "items": items,
    }


def load_hf(model_dir, device):
    """The checkpoint as a transformers model in bf16, refusing any weight left uninitialized."""
    import torch
    import transformers as tf

    errs = []
    for n in ("AutoModelForCausalLM", "AutoModelForImageTextToText"):
        cls = getattr(tf, n)
        try:
            m, info = cls.from_pretrained(
                model_dir, dtype=torch.bfloat16, device_map=device, output_loading_info=True
            )
        except (ValueError, KeyError, OSError) as e:
            errs.append(f"{n}: {str(e)[:300]}")
            continue
        miss = [k for k in info.get("missing_keys", []) if not ALLOWED_MISSING.search(k)]
        bad = list(info.get("mismatched_keys", []))
        if miss or bad:
            errs.append(f"{n}: missing {miss[:5]} ({len(miss)}), mismatched {bad[:3]}")
            print(f"answers: {errs[-1]}; trying the next class", file=sys.stderr, flush=True)
            del m
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
        print(f"answers: {model_dir} loaded as {type(m).__name__}", file=sys.stderr, flush=True)
        gcfg = getattr(m, "generation_config", None)
        if gcfg is not None and getattr(gcfg, "stop_strings", None):
            gcfg.stop_strings = None  # stop_ids() takes the one-token ones as ids
        return m.eval(), n
    raise SystemExit(f"{model_dir}: no transformers class loads it whole: " + "; ".join(errs))


def host_run(model_dir, tok_dir, device, max_new=None, only=None, thinking=False):
    import torch
    import transformers
    from transformers import GenerationConfig

    torch.set_grad_enabled(False)
    s = load_set()
    if only:
        s = {**s, "items": [it for it in s["items"] if it["id"] in only]}
    export = os.path.realpath(model_dir) != os.path.realpath(tok_dir)
    if export:
        export_files_clean(model_dir)
    m, cls = load_hf(model_dir, device)
    checked = weights_check(m.state_dict(), model_dir, tok_dir) if export else None
    dev = next(m.parameters()).device
    state = {"manual": False}

    def by_hand(ids, n, stop):
        """Greedy by full forwards, for a model whose generate() cannot return logits."""
        seq, new, margins = list(ids), [], []
        for _ in range(n):
            lg = m(input_ids=torch.tensor([seq], device=dev)).logits[0, -1].float()
            top = lg.topk(2)
            t = int(top.indices[0])
            margins.append((float(top.values[0] - top.values[1]), int(top.indices[1])))
            new.append(t)
            seq.append(t)
            if t in stop:
                break
        return new, margins

    def generate(ids, n, stop):
        if state["manual"]:
            return by_hand(ids, max_new or n, stop)
        gcfg = GenerationConfig(
            do_sample=False, max_new_tokens=max_new or n, eos_token_id=stop, pad_token_id=stop[0]
        )
        x = torch.tensor([ids], device=dev)
        try:
            y = m.generate(
                input_ids=x,
                attention_mask=torch.ones_like(x),
                generation_config=gcfg,
                output_logits=True,
                return_dict_in_generate=True,
            )
        except ValueError as e:
            if "output_logits" not in str(e):
                raise
            state["manual"] = True
            return by_hand(ids, max_new or n, stop)
        margins = []
        for lg in y.logits:
            top = lg[0].float().topk(2)
            margins.append((float(top.values[0] - top.values[1]), int(top.indices[1])))
        return y.sequences[0, len(ids) :].tolist(), margins

    run = run_items(
        generate,
        tok_dir,
        model_dir,
        f"host transformers {transformers.__version__} {cls} bf16",
        s,
        thinking,
    )
    run["device"] = str(dev)
    if checked:
        run["weights_check"] = checked
    return run


def divergence(item, ref_item):
    """Where a run's new tokens first differ from the bf16 run's: (index, the bf16 logit gap there
    between its top two, whether the run's token there was bf16's second choice), or None."""
    a, b = item.get("tokens"), ref_item.get("tokens")
    if a is None or b is None:
        return None
    k = next((j for j, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    if k == len(a) == len(b):
        return None
    m = ref_item.get("margins")
    if not m or k >= len(m):
        return (k, None, None)
    return (k, m[k][0], k < len(a) and a[k] == m[k][1])


def report(res, missing, ref=None, run=None, ref_run=None, v=None):
    lines = []
    for i, r in res.items():
        mark = "right" if r["right"] else "WRONG"
        if r["broken"]:
            mark = f"BROKEN ({r['broken']})"
        elif not r["answer"]:
            mark = "WRONG (no answer: thinking did not end)"
        if ref is not None and i in ref:
            mark += f"   bf16 {'right' if ref[i]['right'] else 'wrong'}"
            if ref[i]["right"] and not r["right"] and run and ref_run:
                d = divergence(run[i], ref_run[i])
                if d and d[1] is not None:
                    if d[0] >= len(run[i].get("tokens", [])):
                        alt = "the run stopped there"
                    else:
                        alt = "its second choice" if d[2] else "not its second choice"
                    mark += f"; differs from new token {d[0]} on, where bf16's top-2 logit gap is {d[1]:.2f} ({alt})"
        lines.append(f"  {i:22s} {mark}")
    n = sum(r["right"] for r in res.values())
    head = f"{n}/{len(res)} right"
    if missing:
        head += f", {len(missing)} missing: {missing}"
    if ref is not None:
        v = v or verdict(res, ref, run, ref_run)
        nr = sum(r["right"] for r in ref.values())
        head = (
            f"answer check: {'PASS' if v['pass'] and not missing else 'FAIL'}: "
            f"{len(v['regressions'])} regressions {v['regressions']}, {len(v['broken'])} broken {v['broken']}; "
            f"{head} against bf16's {nr}/{len(ref)}, {v['counted']} counted (near ties excluded "
            f"{v['near_ties']}; cleared with thinking on {v['cleared']}; gains {v['gains']}; bf16 "
            f"broken {v['ref_broken']})"
        )
        if v.get("think_na"):
            head += "; thinking on not applicable (bf16 gets none of the reasoning items right)"
        if v.get("invalid"):
            head = f"answer check: INVALID, no verdict: {'; '.join(v['invalid'])}. Was: {head}"
        if v.get("mixed_script"):
            found = "; ".join(f"{i}: {' '.join(w)}" for i, w in v["mixed_script"].items())
            lines.append(f"  mixed script (information, it changes no verdict): {found}")
    return head + "\n" + "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("model")
    r.add_argument(
        "--tokenizer", required=True, help="the bf16 checkpoint (its tokenizer and chat template)"
    )
    r.add_argument("--out", required=True, type=Path)
    r.add_argument("--device", default="cuda:0")
    r.add_argument("--max-new", type=int, default=None)
    r.add_argument("--ids", default="", help="a comma-separated subset of item ids")
    r.add_argument("--thinking", action="store_true", help="the reasoning items, thinking on")
    p = sub.add_parser("prompts")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--thinking", action="store_true", help="the reasoning items, thinking on")
    c = sub.add_parser("score")
    c.add_argument("run", type=Path)
    c.add_argument("--ref", type=Path)
    c.add_argument("--think", type=Path, help="the export's thinking-on run")
    c.add_argument("--think-ref", type=Path, help="the bf16 model's thinking-on run")
    c.add_argument("--json", type=Path)
    a = ap.parse_args(argv)
    if a.cmd == "run":
        t0 = time.time()
        run = host_run(
            a.model,
            a.tokenizer,
            a.device,
            a.max_new,
            set(filter(None, a.ids.split(","))),
            a.thinking,
        )
        run["seconds"] = round(time.time() - t0, 1)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(run, ensure_ascii=False, indent=1), encoding="utf-8")
        res, missing = score_run(run)
        print(report(res, missing))
        return 0
    if a.cmd == "prompts":
        a.out.write_text(
            json.dumps(prompts_file(a.tokenizer, thinking=a.thinking), ensure_ascii=False),
            encoding="utf-8",
        )
        return 0
    run = json.loads(a.run.read_text(encoding="utf-8"))
    res, missing = score_run(run)
    ref = ref_run = None
    if a.ref:
        ref_run = json.loads(a.ref.read_text(encoding="utf-8"))
        ref, _ = score_run(ref_run)
    by = {x["id"]: x for x in run["items"]}
    rby = {x["id"]: x for x in ref_run["items"]} if ref_run else None
    think = None
    problems = [load_problem(run, ref_run)] if ref_run else []
    if a.think and a.think_ref:
        trun = json.loads(a.think.read_text(encoding="utf-8"))
        tref = json.loads(a.think_ref.read_text(encoding="utf-8"))
        think = (score_run(trun)[0], score_run(tref)[0])
        problems.append(load_problem(trun, tref))
    v = (
        verdict(res, ref, by, rby, load_set().get("clear_margin"), think)
        if ref is not None
        else None
    )
    problems = [p for p in problems if p]
    if problems:
        v["pass"] = False
        v["invalid"] = problems
    if v is not None:
        v["mixed_script"] = mixed_script(run)
    print(report(res, missing, ref, by, rby, v))
    out = {
        "run": str(a.run),
        "ref": str(a.ref) if a.ref else None,
        "items": res,
        "missing": missing,
    }
    if ref is not None:
        out["verdict"] = v
        out["verdict"]["pass"] = v["pass"] and not missing
        out["think"] = [str(a.think), str(a.think_ref)] if think else None
    if a.json:
        a.json.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    if ref is None:
        return 0
    return 0 if out["verdict"]["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
