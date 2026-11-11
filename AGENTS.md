# AGENTS.md

Instructions for AI coding agents (Codex, Claude Code and others) working in this repository.
`CLAUDE.md` is a symlink to this file.

## What this is

tensix-bfp-gptq rounds the linear layers of Hugging Face LLMs onto TT-NN's block floating point
grids (`ttnn.bfloat4_b`, `ttnn.bfloat8_b`), by GPTQ or to nearest. It writes bf16 safetensors whose
values TT-NN packs without rounding again. The README covers the method and every command.

## Layout

| Path | Content |
|---|---|
| `tensix_bfp/bfp.py` | the BFP grids, bit for bit with tt-metal's host packer |
| `tensix_bfp/gptq.py` | GPTQ and the per-block exponent search |
| `tensix_bfp/export.py` | the layer-by-layer export and the variant grammar |
| `tensix_bfp/data.py` | the calibration texts: sources, slices and SHA-256 |
| `tensix_bfp/kl.py`, `answers.py`, `answers.json` | the KL and answer checks |
| `tensix_bfp/ministral3.py` | Ministral 3's text-only copy |
| `configs/*.toml` | one config per model |
| `tools/pack_tensix.py`, `tools/verify_grid.py` | export checks, also shipped in each Hugging Face model repo |
| `tests/` | the pytest suite; `test_ttnn_pack.py` needs ttnn and a chip or ttsim |

## Setup and checks

```
pip install -e ".[test]"
pytest
ruff check . && ruff format --check .
```

A change to `bfp.py`, `gptq.py`, `export.py` or `tools/` also runs the ttnn tests: Python 3.12 with
`.[ttnn]` and ttsim, set up as the README's Install section shows, then
`python tests/test_ttnn_pack.py`.

## Rules

- **Exactness is the contract.** `bfp()` reproduces tt-metal's packer bit for bit: round to nearest
  even, the mantissa clamped at its maximum, a zero exponent field flushed to zero. A change to
  rounding passes `test_ttnn_pack.py` against ttnn, not only the torch tests.
- **Published exports stay reproducible.** Don't change a default that alters an existing variant's
  output (`--method`, `--bfp8`, the damping, calibration slicing, `--manifest-format`). Add an
  option and keep the old behavior as the default.
- **Calibration texts are never committed.** They are CC BY-SA and fetched by `tensix_bfp.data`.
  Record a new source there with its SHA-256.
- **A new model family** needs a config in `configs/`, its projection names in `export.py`, a tiny
  random-model test in `tests/test_export_tiny.py` when the architecture is new, and a row in the
  README's table.
- **Dependencies stay few.** A new one needs a reason. The `ttnn` extra is pinned to the version the
  tests ran on.
- **Public material only.** Use public sources (tt-metal, ttnn, ttsim, Tenstorrent's docs) and
  nothing you can't publish under Apache-2.0.

## Style

- Python 3.11 or later; ruff with line length 100 and sorted imports.
- American spelling in code, comments and docs.
- Plain, short docs: say what a thing does and how to run it.
- Commits: Conventional Commits with a gitmoji, imperative mood, header under 72 characters, for
  example `fix(export): 🐛 keep the q, k and v biases for Qwen2.5`. One change per commit, signed.
