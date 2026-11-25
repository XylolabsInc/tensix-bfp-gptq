# Changelog

## Unreleased

### Added

- `tests/test_bfp.py`: directed edge vectors for `bfp()` in both formats (ties, the clamp at the top
  code, subnormal inputs flushed to zero, zero blocks, one non-zero value, both signs), with
  expected values frozen from stock TT-NN 0.81.1 and checked again when ttnn is installed.

### Fixed

- The LM head follows `--method` and `--bfp8` like every layer projection. Before, a bfp4 head
  under GPTQ was always rounded with plain GPTQ, even with `--method exp` (the default), and a bfp8
  head was always rounded to nearest, even with `--bfp8 gptq` or `--bfp8 exp`. Exports made with
  v0.2.0 or earlier, or with any release before v0.3.0, which contains this fix, therefore have a head that their `rounding.*` record misdescribes. With
  `--method gptq --bfp8 rtn` the head is bit for bit what it was. Exports made after the fix are
  new weights: accuracy results of earlier exports do not carry over to them.
- Shard names of `model.safetensors.index.json` are checked when the index is read: a name that
  is not a plain `.safetensors` file name (a path separator, `..`, an absolute path) is refused
  with a `ValueError` naming it, by `packed.py`, `tools/pack_tensix.py` and `tools/verify_grid.py`.
  `unpack` and `pack` also refuse a symbolic-link output directory or output shard path, and
  write nothing before every name has passed. A cache-style snapshot (shard files that are links
  to `../../blobs/<hash>`) still reads, also through `tools/pack_tensix.py`.
- `packed.pack`, `tools/pack_tensix.py` and `tools/verify_grid.py` take their tensors from one
  list, `declared_tensors` in `tensix_bfp/packed.py`: layer projections, the LM head,
  `global_linears` and `extra`.
  `pack` now packs the global linears and extras, a tensor declared with two different formats is
  an error, and a format that is neither `bfp4`, `bfp8` nor `bf16` is an error. `pack` refuses an
  export with a `qkv_row_order`, and `tools/row_order.py` checks the fused row count without
  `assert`. `packed.py` still works copied on its own.

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
