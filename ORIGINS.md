# Origins

Where every file of this repository comes from, and how that was checked.

## Classes

- **ours**: written for this repository.

Every file is ours. No file is copied from another code base: the GPTQ step implements the algorithm of Frantar et
al., 2022 from the paper, and the BFP rounding follows TT-NN's `bfloat8_b` and `bfloat4_b` formats and is
checked against stock ttnn's own packing. The code calls the public APIs of torch, safetensors, tokenizers,
transformers and ttnn.

## How it was checked

1. **Line comparison.** Every line of 30 or more characters (whitespace-normalized) of each text file was looked up
   in tt-metal v0.81.1 (its source checkout at tag `v0.81.1`, 23,859 text files), transformers v5.13.0 (5,885), the
   GPTQ reference implementation [IST-DASLab/gptq](https://github.com/IST-DASLab/gptq) (commit `2d65066`, 37),
   [AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ) (commit `9f7d370`, 132) and
   [llama.cpp](https://github.com/ggml-org/llama.cpp) (commit `d3c3e24`, 2,022). The column "lines found upstream"
   counts the lines found in any of them. They are license headers, import lines and common Python idioms (for
   example `from safetensors import safe_open`, `with tempfile.TemporaryDirectory() as d:`).
2. **Tokens.** Each Python file was compared with every Python, C, C++ and shell file of the five trees as a sequence of
   code tokens (comments dropped): at most 3.2 % of a file's distinct 10-token sequences occur in any one of them
   (`tests/test_data.py`), and Python's difflib ratio against the closest files is at most 0.24 (`tensix_bfp/gptq.py`
   against IST-DASLab/gptq's `gptq.py`, which implements the same algorithm).
3. **When.** On 2026-11-17, on the tree of v0.1.0 (`e889353`), which this file joins unchanged.
4. **v0.2.0.** On 2026-11-19, the files v0.2.0 adds or changes (`tensix_bfp/packed.py`,
   `tests/test_packed.py`, `tests/test_ttnn_pack.py`, `tensix_bfp/__init__.py`, `pyproject.toml`) were
   checked against the same five trees at the same versions: the line comparison as in 1, and the share
   of each Python file's distinct 10-token sequences (comments and docstrings dropped) found in any one
   upstream Python, C, C++ or shell file, at most 1.5 % (`tests/test_ttnn_pack.py`).
5. **Row order.** On 2026-11-24, the files the `qkv_row_order` change adds or changes (`tools/row_order.py`,
   `tools/pack_tensix.py`, `tools/verify_grid.py`, `tests/test_check_tools.py`) were checked against the same five
   trees at the same versions: the line comparison as in 1, and the share of each file's distinct 10-token
   sequences (comments and docstrings dropped) found in any one upstream Python, C, C++ or shell file, at most
   2.4 % (`tools/verify_grid.py`: argument parsing and `sys.path` lines).

6. **Global linears.** On 2026-11-24, `tools/pack_tensix.py`, `tools/verify_grid.py`, `tests/test_check_tools.py` and
   `tests/test_ttnn_pack.py`, as the `global_linears` change leaves them on top of the row-order change, were recounted
   on item 5's basis: every line of 30 or more characters, whitespace-normalized, duplicates included, in both columns
   (the counts for items 5's rows reproduce that way). The lines found upstream were looked up in tt-metal v0.81.1;
   the comparison with the other four trees is the reviewer's.

## Files

| File | Class | Lines checked | Lines found upstream | Notes |
|---|---|---|---|---|
| `pyproject.toml` | ours | 25 | 5 | package metadata |
| `tensix_bfp/__init__.py` | ours | 6 | 1 | package marker |
| `tensix_bfp/answers.py` | ours | 377 | 10 | the answer check (answers.json's items, greedy, against the bf16 model) |
| `tensix_bfp/answers.json` | ours | 47 | 0 | the answer-check items, written for this repository |
| `tensix_bfp/bfp.py` | ours | 69 | 1 | TT-NN's bfloat8_b and bfloat4_b rounding in torch, written from TT-NN's documented format and checked against stock ttnn's own packing (`tests/test_ttnn_pack.py`) |
| `tensix_bfp/data.py` | ours | 86 | 4 | downloads the calibration and evaluation texts at run time; no text is shipped |
| `tensix_bfp/export.py` | ours | 481 | 14 | the exporter and its model families, through transformers' public API |
| `tensix_bfp/gptq.py` | ours | 93 | 1 | GPTQ (Frantar et al., 2022, arXiv:2210.17323), implemented from the paper |
| `tensix_bfp/kl.py` | ours | 58 | 2 | mean KL, top-1 and top-5 against the bf16 model; its sequences are cut as llama.cpp's perplexity tool cuts them (the behavior, not its code) |
| `tensix_bfp/ministral3.py` | ours | 39 | 3 | Ministral 3's release to a text-only checkpoint |
| `tensix_bfp/packed.py` | ours | 281 | 5 | packed checkpoints in TT-NN's tile byte layout, written to a written spec of our own design, checked against tt-metal v0.81.1's public packer source and stock ttnn (`tests/test_ttnn_pack.py`) |
| `tests/test_bfp.py` | ours | 35 | 2 |  |
| `tests/test_check_tools.py` | ours | 158 | 11 | adds the row-order and global-linear tests |
| `tests/test_data.py` | ours | 20 | 1 |  |
| `tests/test_export_tiny.py` | ours | 74 | 7 |  |
| `tests/test_gptq.py` | ours | 39 | 4 |  |
| `tests/test_packed.py` | ours | 176 | 15 |  |
| `tests/test_ttnn_pack.py` | ours | 109 | 10 | bfp.py and packed.py against stock ttnn's packer; the checker on global linears |
| `tools/pack_tensix.py` | ours | 78 | 5 | packs a checkpoint with `ttnn.as_tensor` and checks the round trip; applies a declared `qkv_row_order`; also checks the manifest's `global_linears` |
| `tools/row_order.py` | ours | 46 | 1 | the `spark-llama` row order of a fused q/k/v projection, written to a written spec of our own design from the public Spark-X2.6-4B config and model code; shared by both check tools |
| `tools/verify_grid.py` | ours | 78 | 4 | checks that every quantized matrix is on its grid; applies a declared `qkv_row_order`; also checks the manifest's `global_linears` |
| `configs/*.toml` (15 files) | ours | 1-6 each | 0 | the export configurations of the published checkpoints |
| `assets/xylolabs-lockup.svg`, `assets/xylolabs-lockup-dark.svg` | ours | 1 each | 0 | Xylolabs' logo |

Documentation and other files: `README.md`, `AGENTS.md`, `CLAUDE.md`, `CHANGELOG.md`, `DEVELOPMENT.md`, `NOTICE`, `.gitignore`
and this file are ours; `LICENSE` is the text of the Apache License, Version 2.0.
