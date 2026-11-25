# Changelog

## Unreleased

### Fixed

- Shard names of `model.safetensors.index.json` are checked when the index is read: a name that
  is not a plain `.safetensors` file name (a path separator, `..`, an absolute path) is refused
  with a `ValueError` naming it, by `packed.py`, `tools/pack_tensix.py` and `tools/verify_grid.py`.
  `unpack` and `pack` also refuse a symbolic-link output directory or output shard path, and
  write nothing before every name has passed. A cache-style snapshot (shard files that are links
  to `../../blobs/<hash>`) still reads, also through `tools/pack_tensix.py`.

## 0.2.0 (2026-11-18)

### Added

- `tensix_bfp/packed.py`: packed checkpoints, format `tensix-bfp-tiles` version 1. A packed
  checkpoint stores each BFP4 or BFP8 matrix as the tile bytes TT-NN keeps for it, 4.5 or 8.5 bits
  a weight, and lists those matrices in `tensix_pack.json`. Decoding gives the bf16 export bit for
  bit, except that a `-0.0` loads as `+0.0`. The commands are `verify`, `unpack` and `pack`, and
  `load_bf16` and `iter_bf16` load a packed checkpoint in Python. It needs only torch and
  safetensors and also works copied on its own, as the Hugging Face model repos ship it. The
  console script is `tensix-bfp-packed`.
- `tests/test_packed.py`, and a check in `tests/test_ttnn_pack.py` that the packed bytes are the
  ones TT-NN writes to its weight cache.
- `DEVELOPMENT.md`, how the repository is developed and checked, and `ORIGINS.md`, where every
  file comes from.

### Changed

- `pyproject.toml` names ruff's lint rules (`E4`, `E7`, `E9`, `F`, `I`), so that a newer ruff's
  defaults do not change them.

The exports are unchanged: every option and default writes what 0.1.0 wrote.

## 0.1.0 (2026-11-16)

The first release: BFP4 and BFP8 rounding bit for bit with tt-metal's host packer, GPTQ with the
per-block exponent search, the layer-by-layer export and its variants, the KL and answer checks,
and the export checks `tools/verify_grid.py` and `tools/pack_tensix.py`.
