# Development

This page says how tensix-bfp-gptq is developed and how a release is checked before it is
published. The Hugging Face model repos that use its exports and its loader link here.

## What the work uses

- Tenstorrent's open software and documentation: tt-metal and TT-NN, ttsim, SFPI, their docs and
  their community. The BFP formats and the tile byte layout follow TT-NN's documented behavior and
  tt-metal's public source (v0.81.1). The tests check them against stock `ttnn` on a Tenstorrent
  card or on ttsim.
- Published papers, such as GPTQ (Frantar et al., 2022, arXiv:2210.17323), implemented from the
  paper.
- The public APIs of torch, safetensors, tokenizers and transformers.
- Public model checkpoints and public texts. The texts are downloaded at run time, not shipped
  (`tensix_bfp/data.py`).

## What it does not use

No SDK, source code, documents, tools, test facilities or hardware of any other hardware vendor
are used to write, design or test this repository. Defaults in the code are generic or chosen on
ttsim. The code and the docs describe Tenstorrent Tensix and ttsim only, and carry no numbers
measured on other hardware.

## How the work is done

- **A clean checkout.** Development happens in a checkout of this repository by developers and
  coding agents who are not shown any third party's confidential material while they work.
- **Written specs only.** When another team contributes a design, such as the packed file layout
  of the Hugging Face model repos, it arrives as a written specification of our own design,
  reviewed before it is used, never as code. The packed layout in v0.2.0 came this way. Its byte
  order was then checked against tt-metal's public packer source.
- **Tests.** `pytest` runs on a CPU. The tests that need `ttnn` run on ttsim or a Tenstorrent card
  (`tests/test_ttnn_pack.py`).

## Before a release is published

1. A reviewer outside the development session reads the release's changes for confidential
   material and for anything that describes non-public hardware.
2. Every file is compared, as a sequence of code tokens, with the non-public third-party source
   trees we hold. The comparison must show nothing copied.
3. `ORIGINS.md` records each file's origin and its comparison with public upstream code.

Only then are the commits and the tag pushed.

## History

- **v0.2.0** was developed as this page describes.
- **v0.1.0** was written from public material, under the rule in `AGENTS.md`. Its files were
  checked after the release, as `ORIGINS.md` records (2026-11-17).
