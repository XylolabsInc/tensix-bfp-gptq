# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Xylolabs
"""Ministral 3's BF16 release (Mistral3ForConditionalGeneration: language_model.model.* and
language_model.lm_head.weight beside a vision tower) as a text-only Ministral3ForCausalLM
checkpoint (model.*, lm_head.weight), every kept tensor byte for byte, for the export's
"ministral3" family. The config is the release's text_config with an untied head and the
release's begin-of-sequence, end-of-sequence and pad ids; the tokenizer files are copied.

  python -m tensix_bfp.ministral3 SRC DST
"""

import json
import shutil
import sys
from pathlib import Path

TOKENIZER_FILES = (
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "SYSTEM_PROMPT.txt",
    "tekken.json",
)


def rename(key):
    """The text-only name of a release tensor, or None for one left out."""
    if key.startswith("language_model.model."):
        return "model." + key[len("language_model.model.") :]
    if key == "language_model.lm_head.weight":
        return "lm_head.weight"
    return None


def flatten(src, dst):
    """Write the text-only copy of src into dst; returns (tensors, bytes)."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    weight_map = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    new_map, total = {}, 0
    for shard in sorted(set(weight_map.values())):
        out = {}
        with safe_open(str(src / shard), "pt") as f:
            for k in f.keys():  # noqa: SIM118 (a safe_open handle, not a dict)
                r = rename(k)
                if r is not None:
                    out[r] = f.get_tensor(k)
        if out:
            save_file(out, str(dst / shard), metadata={"format": "pt"})
            for r, t in out.items():
                new_map[r] = shard
                total += t.numel() * t.element_size()
    if "lm_head.weight" not in new_map:
        raise SystemExit(f"{src}: no language_model.lm_head.weight")
    index = {"metadata": {"total_size": total}, "weight_map": dict(sorted(new_map.items()))}
    (dst / "model.safetensors.index.json").write_text(json.dumps(index, indent=1))
    cfg = json.loads((src / "config.json").read_text())
    gen_path = src / "generation_config.json"
    gen = json.loads(gen_path.read_text()) if gen_path.exists() else {}
    text = dict(cfg["text_config"])
    text.update(
        architectures=["Ministral3ForCausalLM"],
        model_type="ministral3",
        torch_dtype="bfloat16",
        tie_word_embeddings=False,
        bos_token_id=gen.get("bos_token_id", 1),
        eos_token_id=gen.get("eos_token_id", 2),
        pad_token_id=gen.get("pad_token_id"),
    )
    (dst / "config.json").write_text(json.dumps(text, indent=1))
    for name in TOKENIZER_FILES:
        if (src / name).exists():
            shutil.copy2(src / name, dst / name)
    return len(new_map), total


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        raise SystemExit(__doc__)
    n, total = flatten(argv[0], argv[1])
    print(f"{argv[1]}: {n} tensors, {total} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
