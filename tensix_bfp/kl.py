# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Xylolabs
"""KL divergence of an export from its bf16 original, with top-1 agreement and top-5 overlap.

  python -m tensix_bfp.kl --config CONFIG.toml --base BASE_DIR --model EXPORT_DIR \\
      [--device cuda:0] [--data DIR] [--out FILE]

The evaluation sequences are the config's [eval] texts, formats and chunks (export.Model's
eval_tokens): each chunk of [eval] tokens tokens starts where the one before ended, a "raw"
sequence starting with the begin-of-sequence token where the model takes one, as llama.cpp's
perplexity tool cuts its chunks, and a "chat" sequence being the text as the model's reply in its
chat template, scored on the text's tokens only. Both models run in bf16 under transformers on
the same device; at every scored position (from the sequence's first scored token on) this
computes KL(bf16 || export) over the full vocabulary, whether the two argmax tokens agree, and
the overlap of their top-5 sets. One line per sequence, the mean per text, then ALL.

The export is first checked to load as its index says and to hold weights that differ from the
bf16 checkpoint's (answers.export_files_clean, answers.weights_check).
"""

import argparse
import gc
from pathlib import Path

import torch

from . import answers
from .export import Model, load_config


def ref_logprobs(model_dir, seqs, first, device):
    """The bf16 model's log-probabilities at the scored positions of each sequence, float16 on
    the CPU ([n - first, vocab] each: positions first - 1 .. n - 2 predict tokens first .. n - 1)."""
    m, _ = answers.load_hf(model_dir, device)
    out = []
    with torch.no_grad():
        for ids, f in zip(seqs, first, strict=True):
            lg = m(input_ids=torch.tensor([ids], device=device)).logits[0, f - 1 : len(ids) - 1]
            out.append(torch.log_softmax(lg.float(), -1).half().cpu())
    del m
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def kl_rows(model, seqs, first, refs, device):
    """Per sequence (positions, mean KL, top-1 agreement, top-5 overlap) of `model` against the
    reference log-probabilities `refs`."""
    rows = []
    with torch.no_grad():
        for ids, f, ref in zip(seqs, first, refs, strict=True):
            lg = model(input_ids=torch.tensor([ids], device=device)).logits[0, f - 1 : len(ids) - 1]
            lp = torch.log_softmax(lg.float(), -1)
            rf = ref.to(lp.device).float()
            kl = (rf.exp() * (rf - lp)).sum(-1)
            top1 = (rf.argmax(-1) == lp.argmax(-1)).float()
            r5, o5 = rf.topk(5, -1).indices, lp.topk(5, -1).indices
            top5 = (r5[:, :, None] == o5[:, None, :]).any(-1).float().mean(-1)
            rows.append((len(kl), kl.mean().item(), top1.mean().item(), top5.mean().item()))
    return rows


def summary(names, rows):
    """{text name: (positions, KL, top-1, top-5)} over each text's chunks, and "ALL"."""
    acc = {}
    for (name, _), r in zip(names, rows, strict=True):
        acc.setdefault(name, []).append(r)
    acc["ALL"] = rows
    out = {}
    for name, rs in acc.items():
        n = sum(r[0] for r in rs)
        out[name] = (n, *(sum(r[0] * r[j] for r in rs) / n for j in (1, 2, 3)))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--base", required=True, help="the bf16 checkpoint")
    ap.add_argument("--model", required=True, help="the export")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--data", help="the directory of the evaluation texts (tensix_bfp.data)")
    ap.add_argument("--out", type=Path, help="also write the table here")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    md = Model(cfg, a.base, "cpu", a.data)
    names, seqs, first = md.eval_tokens()
    answers.export_files_clean(a.model)
    refs = ref_logprobs(a.base, seqs, first, a.device)
    m, _ = answers.load_hf(a.model, a.device)
    answers.weights_check(m.state_dict(), a.model, a.base)
    rows = kl_rows(m, seqs, first, refs, a.device)
    lines = ["sequence\tpositions\tKL\ttop1\ttop5"]
    for (name, k), (n, kl, t1, t5) in zip(names, rows, strict=True):
        lines.append(f"{name}_{k}\t{n}\t{kl:.6f}\t{t1:.6f}\t{t5:.6f}")
    for name, (n, kl, t1, t5) in summary(names, rows).items():
        lines.append(f"{name}\t{n}\t{kl:.6f}\t{t1:.6f}\t{t5:.6f}")
    text = "\n".join(lines) + "\n"
    print(text, end="")
    if a.out:
        a.out.write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
