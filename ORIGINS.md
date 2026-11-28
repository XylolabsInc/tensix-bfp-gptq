# Origins

## Classes

- **ours**: written for this repository.
- **ours, follows tt-metal**: its structure follows a named tt-metal file (https://github.com/tenstorrent/tt-metal, Apache-2.0), so it is treated as a
  modified version of that file: it carries that file's Tenstorrent SPDX lines above ours, and [NOTICE](NOTICE) names
  the file. No text is copied beyond calls of tt-metal's public APIs.
- **adapted**: taken from a public repository under its declared license (the source, license and changes are in the
  "Source" and "Notes" columns and in the last section); it keeps that source's copyright line beside ours.

Every file is ours. No file is copied from another code base: the GPTQ step implements the algorithm of Frantar
et al., 2022 (arXiv:2210.17323) from the paper, and the BFP rounding follows TT-NN's `bfloat8_b` and
`bfloat4_b` formats and is checked against stock ttnn's own packing. The code calls the public APIs of torch,
safetensors, tokenizers, transformers and ttnn.

## Files

One row per tracked file, except `LICENSE` (the text of the Apache License, Version 2.0), `NOTICE`, this file and
`DEVELOPMENT.md`; `tests/test_origins.py` checks that each appears exactly once. "Source" names the file a row follows
or is adapted from. "Lines checked" counts the lines of 30 or more characters (whitespace-normalized), every line
counted, duplicates included; "Lines in tt-metal" counts those also found in the tt-metal tree described below.

| File | Class | Source | Lines checked | Lines in tt-metal | Notes |
|---|---|---|---|---|---|
| `.gitignore` | ours |  | 0 | 0 | ignore patterns |
| `AGENTS.md` | ours |  | 40 | 0 | instructions for coding agents |
| `CHANGELOG.md` | ours |  | 58 | 0 | documentation |
| `CLAUDE.md` | ours |  | 0 | 0 | symbolic link to AGENTS.md |
| `README.md` | ours |  | 145 | 0 | documentation |
| `assets/xylolabs-lockup-dark.svg` | ours |  | 1 | 0 | Xylolabs' logo, hand-written SVG; no generator script |
| `assets/xylolabs-lockup.svg` | ours |  | 1 | 0 | Xylolabs' logo, hand-written SVG; no generator script |
| `configs/arete-27b.toml` | ours |  | 2 | 0 | the export configuration of a published checkpoint |
| `configs/gemma4-12b.toml` | ours |  | 2 | 0 | the export configuration of a published checkpoint |
| `configs/gemma4-31b.toml` | ours |  | 4 | 0 | the export configuration of a published checkpoint |
| `configs/gemma4-e2b-qat.toml` | ours |  | 2 | 0 | the export configuration of a published checkpoint |
| `configs/gemma4-e2b.toml` | ours |  | 2 | 0 | the export configuration of a published checkpoint |
| `configs/gemma4-e4b.toml` | ours |  | 2 | 0 | the export configuration of a published checkpoint |
| `configs/llama-3.1-8b-cal4x.toml` | ours |  | 6 | 0 | the export configuration of a published checkpoint |
| `configs/llama-3.1-8b.toml` | ours |  | 2 | 0 | the export configuration of a published checkpoint |
| `configs/ministral3-14b.toml` | ours |  | 2 | 0 | the export configuration of a published checkpoint |
| `configs/qwen25-7b.toml` | ours |  | 1 | 0 | the export configuration of a published checkpoint |
| `configs/qwen3-4b.toml` | ours |  | 2 | 0 | the export configuration of a published checkpoint |
| `configs/qwen3-8b.toml` | ours |  | 2 | 0 | the export configuration of a published checkpoint |
| `configs/qwen35-4b.toml` | ours |  | 3 | 0 | the export configuration of a published checkpoint |
| `configs/qwen35-9b.toml` | ours |  | 3 | 0 | the export configuration of a published checkpoint |
| `configs/qwen38-27b.toml` | ours |  | 2 | 0 | the export configuration of a published checkpoint |
| `pyproject.toml` | ours |  | 25 | 0 | package metadata |
| `tensix_bfp/__init__.py` | ours |  | 5 | 1 | package marker |
| `tensix_bfp/answers.json` | ours |  | 47 | 0 | the answer-check items, written for this repository |
| `tensix_bfp/answers.py` | ours |  | 377 | 10 | the answer check (answers.json's items, greedy, against the bf16 model) |
| `tensix_bfp/bfp.py` | ours |  | 69 | 1 | TT-NN's bfloat8_b and bfloat4_b rounding in torch, written from TT-NN's documented format and checked against stock ttnn's own packing (`tests/test_ttnn_pack.py`) |
| `tensix_bfp/data.py` | ours |  | 86 | 4 | downloads the calibration and evaluation texts at run time; no text is shipped |
| `tensix_bfp/export.py` | ours |  | 505 | 13 | the exporter and its model families, through transformers' public API |
| `tensix_bfp/gptq.py` | ours |  | 93 | 1 | GPTQ (Frantar et al., 2022, arXiv:2210.17323), implemented from the paper |
| `tensix_bfp/kl.py` | ours |  | 58 | 2 | mean KL, top-1 and top-5 against the bf16 model; its sequences are cut as llama.cpp's perplexity tool cuts them (the behavior, not its code) |
| `tensix_bfp/ministral3.py` | ours |  | 39 | 3 | Ministral 3's release to a text-only checkpoint |
| `tensix_bfp/packed.py` | ours |  | 384 | 7 | packed checkpoints in TT-NN's tile byte layout, written to a written spec of our own design, checked against tt-metal v0.81.1's public packer source and stock ttnn (`tests/test_ttnn_pack.py`); holds the one list of declared tensors (`declared_tensors`) that both check tools use, so it still runs alone |
| `tests/test_bfp.py` | ours |  | 119 | 4 | adds directed edge vectors frozen from TT-NN 0.81.1 (ties, clamp, subnormal flush, zeros, signs), compared by bit pattern so that a zero's sign counts |
| `tests/test_cache_paths.py` | ours |  | 40 | 7 | cache confinement, directory creation, and a new cache file for a changed weight, with a stand-in for ttnn |
| `tests/test_check_tools.py` | ours |  | 206 | 15 | adds the row-order, global-linear and extra-matrix tests |
| `tests/test_data.py` | ours |  | 20 | 1 |  |
| `tests/test_declared_sections.py` | ours |  | 164 | 4 | every declared section is packed and checked, conflicting formats and qkv_row_order exports are refused |
| `tests/test_export_tiny.py` | ours |  | 121 | 8 |  |
| `tests/test_gptq.py` | ours |  | 39 | 4 |  |
| `tests/test_head_rounding.py` | ours |  | 114 | 7 | the LM head goes through the routine its format and the options name, on its grid, and unchanged for --method gptq --bfp8 rtn |
| `tests/test_origins.py` | ours |  | 21 | 4 | every tracked file is listed in ORIGINS.md exactly once |
| `tests/test_packed.py` | ours |  | 269 | 16 |  |
| `tests/test_shard_names.py` | ours |  | 141 | 9 | shard names that leave their directory, and cache-style snapshots |
| `tests/test_ttnn_pack.py` | ours |  | 150 | 14 | bfp.py and packed.py against stock ttnn's packer; the checker on global linears, extra matrices, conflicting formats and a cache-style snapshot |
| `tools/pack_tensix.py` | ours |  | 64 | 5 | packs a checkpoint with `ttnn.as_tensor` and checks the round trip; applies a declared `qkv_row_order`; checks every declared section through `packed.py`'s `declared_tensors`; names each cache file by the matrix it packs |
| `tools/row_order.py` | ours |  | 46 | 1 | the `spark-llama` row order of a fused q/k/v projection, written to a written spec of our own design from the public Spark-X2.6-4B config and model code; shared by both check tools |
| `tools/verify_grid.py` | ours |  | 59 | 4 | checks that every quantized matrix is on its grid; applies a declared `qkv_row_order`; checks every declared section through `packed.py`'s `declared_tensors` |

## How it was checked

The counts in 1 and 2 are from a run on 2026-11-25 over the files of this tree (on `d214f19` with this change), against
the version this repository pins, tt-metal v0.81.1. The table's two count columns are from the recount in 4.

1. **Line comparison.** Every line of 30 or more characters (whitespace-normalized), every line counted and duplicates
   included, of each file in the table was looked up in tt-metal v0.81.1 (its source checkout at tag `v0.81.1`,
   24,522 text files, plus the runtime tree of the `ttnn` 0.81.1 wheel, 6,665) and in
   transformers v5.13.0 (5,909), the GPTQ reference implementation
   [IST-DASLab/gptq](https://github.com/IST-DASLab/gptq) (commit `2d65066`, 41),
   [AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ) (commit `9f7d370`, 172) and
   [llama.cpp](https://github.com/ggml-org/llama.cpp) (commit `d3c3e24`, 4,152). Of 3,215 lines
   checked, 131 are found in tt-metal's tree and 8 only in one of the other four trees. The lines found
   are license headers, import lines, calls of public APIs and common idioms.
2. **Tokens.** Each of the 23 `.py`, `.cpp`, `.h` and `.sh` files was compared with every Python, C, C++, CUDA and
   shell file of the same trees as a set of distinct 10-token sequences (comments dropped, and docstrings for Python):
   at most 3.70 % of a file's distinct sequences occur in any one upstream file (the highest is `tests/test_data.py`, closest
   to `tt-metal/models/demos/unet_3d/demo/config.py`; a file of 20 lines checked, so few sequences); 6 files reach 2.5 % or more. Among
   the files of class "ours" with 50 or more lines checked, the highest is 3.37 % (`tools/pack_tensix.py`).
3. **v0.3.0.** The release adds or changes `CHANGELOG.md`, `DEVELOPMENT.md`, `ORIGINS.md`, `README.md`,
   `pyproject.toml`, `tensix_bfp/__init__.py`, `tensix_bfp/export.py`, `tensix_bfp/packed.py`,
   `tests/test_bfp.py`, `tests/test_check_tools.py`, `tests/test_declared_sections.py`,
   `tests/test_head_rounding.py`, `tests/test_origins.py`, `tests/test_packed.py`, `tests/test_shard_names.py`,
   `tests/test_ttnn_pack.py`, `tools/pack_tensix.py`, `tools/row_order.py`, and `tools/verify_grid.py`.
   All 19 files are class "ours". The 2026-11-27 comparison used the same six upstream trees at the same
   versions as in 1. Using the line comparison in 1, 106 of 2,356 lines matched at least one tree, including
   99 in the tt-metal source checkout. The share of each Python file's distinct 10-token sequences (comments and
   docstrings dropped) found in any one upstream Python, C, C++ or shell file was at most 3.4 %
   (`tools/pack_tensix.py`). The `ttnn` 0.81.1 wheel's runtime tree contained 5,489 text files in this run
   (6,665 in the earlier inventory). The tt-metal source and transformers counts differed by 13 and 4 files,
   respectively. The other three counts stayed the same.
4. **v0.4.1.** The release changes `CHANGELOG.md`, `ORIGINS.md`, `pyproject.toml`, `tensix_bfp/__init__.py`,
   `tensix_bfp/bfp.py`, `tensix_bfp/packed.py`, `tests/test_bfp.py`, `tests/test_cache_paths.py`,
   `tests/test_packed.py` and `tools/pack_tensix.py`, all class "ours". On 2026-11-28 both count columns were
   recounted for every row with the line comparison in 1, against tt-metal's source tree at tag `v0.81.1` (24,522
   text files) and the `ttnn` 0.81.1 wheel's runtime tree (5,489). The rows of files changed since their last count,
   by v0.4.0 or by this release, now hold current counts; the other rows came out as before. Of 3,543 lines
   checked, 145 are found in tt-metal's tree. One line this release adds is among them: the new test in
   `tests/test_cache_paths.py` writes the index file with the same line as the test before it. The other four
   trees of 1 and the token comparison of 2 were not run again.

## Third-party model code and configs

None. No file is adapted from a third-party model repository or from transformers: the model families are
handled through transformers' public API at run time, and `configs/*.toml` are written for this repository.
