# Changelog

## 0.4.1 (2026-11-28)

### Fixed

- `tools/pack_tensix.py --cache-dir` names each TT-NN cache file by the tensor, its format,
  dtype, shape and bytes as packed. It used the tensor name alone, and `ttnn.as_tensor` loads an
  existing file, so a rerun with the same cache directory after a weight or the `qkv_row_order`
  changed packed and checked the old data. Cache files written by 0.4.0 are not reused.
- `pack` and `unpack` of `tensix_bfp/packed.py` give the output directory the destination's
  mode. It was left 0700, the mode of the staging directory, so other users could not read it.
- `bfp()` returns +0.0 for a value that rounds to zero, as tt-metal's packer stores it; a
  negative input gave -0.0. Values do not change, nor do the tile bytes of a packed checkpoint;
  a bf16 export can differ from one made with 0.4.0 in the sign bit of such zeros.
- `tensix_bfp/packed.py` refuses a `tensix_pack.json` tensor record that is not an object with
  a `TypeError` naming the tensor, which `verify` reports as a problem. `verify` raised an
  `AttributeError`.

## 0.4.0 (2026-11-28)

### Changed

- GPTQ exports using a configured `rms_norm_eps` other than the previous helper default can
  produce different packed values from v0.3.0 with the same arguments. Tied quantized-head
  exports also change the embedding values and emitted tensor keys. These corrected exports
  are not bit-for-bit reproducible from v0.3.0; there is no legacy export option. Existing
  v0.3.0 packed files remain readable.
- Chat evaluation now uses contiguous response chunks, so evaluation inputs and results change.
- TT-NN cache filenames now use hashes; caches under the old names are not reused.

## 0.3.0 (2026-11-27)

### Added

- `tests/test_bfp.py`: directed edge vectors for `bfp()` in both formats (ties, the clamp at the top
  code, subnormal inputs flushed to zero, zero blocks, one non-zero value, both signs), with
  expected values frozen from stock TT-NN 0.81.1 and checked again when ttnn is installed.
- `tools/verify_grid.py` and `tools/pack_tensix.py` apply a declared `qkv_row_order` to fused q/k/v
  projections before checking or packing. Both tools now check the quantized matrices listed under
  `global_linears` and `extra` in `quant4.json`.

### Fixed

- A `bf16` LM head is stored without BFP rounding. A `bfloat4_b` head follows `--method` when GPTQ
  is enabled, and a `bfloat8_b` head follows `--bfp8`. In v0.2.0 and earlier, quantized heads used
  plain GPTQ for `bfloat4_b` with GPTQ enabled and round to nearest otherwise, so the `rounding.*`
  record could misdescribe the head. A `bf16` head matches the earlier export with any rounding
  options. A `bfloat4_b` head matches with `--method gptq` or with GPTQ disabled, and a `bfloat8_b`
  head matches with `--bfp8 rtn` or with `--bfp8 gptq` and GPTQ disabled. The other settings
  round the head differently and can change its weights, so reevaluate accuracy if they change.
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
  an error, and a format that is neither `bfp4`, `bfp8` nor `bf16` is an error. `packed.pack` refuses an
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
