# How this repository is developed

tensix-bfp-gptq is written for Tenstorrent Tensix NPUs from Tenstorrent's open software stack only.

## Sources

- Use the pinned tt-metal and TT-NN release (v0.81.1), the ttsim simulator, tt-forge and tt-mlir
  where named, and Tenstorrent's public documentation and community answers.
- Use public Hugging Face model code and weights under the licenses declared in their model cards.
  Keep copyright lines in adapted files and record their sources in ORIGINS.md.

## Hardware-specific work

Use only Tenstorrent's public hardware, SDKs, source code, documentation, and test facilities for
hardware-specific work. Read device-dependent defaults such as core counts, memory banks, and L1
sizes at run time. Treat tuned settings as starting points.

## How a change is made

1. Write a specification for the project's own design covering the algorithm, data layout, API, tests,
   and tolerances. Review it for third-party content and remove any third-party code before implementation.
2. Implement the change using only the specification and the public stack. Work in a clean checkout
   containing only this repository, the pinned upstream release, and ttsim.
3. Test each change on ttsim using the checks in [AGENTS.md](AGENTS.md) and the setup in [README.md](README.md).
   Compare against float64 or bf16 references using the tolerances in the specification.
4. Record each file's origin in ORIGINS.md and include SPDX headers. List files that follow tt-metal in NOTICE.
5. Review each commit for third-party content before pushing it. Include a token comparison against code
   outside the permitted sources. Sign each commit.

## Releases before this process

Xylolabs developed v0.1.0 (2026-11-16) using upstream tt-metal, TT-NN, and ttsim. Each release after
v0.1.0 follows the process above.

## Log

| Date | Change | Specification (name, MD5) | Upstream | Checks |
|---|---|---|---|---|
| 2026-11-24 | feat(tools): apply a declared row order of a fused q/k/v projection | 06-qkv-row-order.md, fe3716d0867b828689cf9e6bd879cc5d | tt-metal/TT-NN v0.81.1, ttsim v1.16.0 | pytest (73 passed, 1 skipped), test_ttnn_pack.py on ttsim (6 tests passed; pytest with ttnn: 79 passed), ruff check and format |
| 2026-11-24 | feat(tools): check the manifest's global linears | 07-global-linears.md, 61826f2540bb6ccbde25f1d1034539f6 | tt-metal/TT-NN v0.81.1, ttsim v1.16.0 | pytest (TT-NN installed: 87 passed), test_ttnn_pack.py on ttsim (7 tests passed, 38 s, including test_checker_covers_global_linears), ruff check and format |
| 2026-11-24 | feat(tools): check the manifest's extra matrices | 01-verify-grid-extras.md, fb52fb46b0f78fe54834601c8bc8d643 | tt-metal/TT-NN v0.81.1, ttsim v1.16.0 | pytest (TT-NN installed: 94 passed), test_ttnn_pack.py on ttsim (7 tests passed), ruff check and format |
| 2026-11-24 | docs(origins): record every file's origin | 01-origins-md.md, 0fb963254532e4ff23a9568d4dce868a | tt-metal/TT-NN v0.81.1, ttsim v1.16.0 | pytest (TT-NN installed: 96 passed), tests/test_origins.py (2 passed), ruff check and format |
| 2026-11-25 | fix(packed): reject shard names that leave the checkpoint directory | 01-shard-name-safety.md, bfdab985c6474f2b8b4f8a75ff2ae688 | tt-metal/TT-NN v0.81.1, ttsim v1.16.0 | pytest (TT-NN installed: 141 passed, 1 skipped), test_ttnn_pack.py on ttsim (8 tests passed, including pack_tensix.py on a cache-style snapshot), ruff check and format |
| 2026-11-25 | fix(packed): pack and check every declared section the same way | 02-packers-declared-sections.md, 1c277030f3d3c27851cb48a4c4bdf083 | tt-metal/TT-NN v0.81.1, ttsim v1.16.0 | pytest (TT-NN installed: 158 passed, 1 skipped), test_ttnn_pack.py on ttsim (9 tests passed), ruff check and format |
| 2026-11-25 | test(bfp): pin the core rounding to TT-NN with directed edge vectors | 03-rounding-edge-tests.md, 4370f7274112b89e71107c1af32a691b | tt-metal/TT-NN v0.81.1, ttsim v1.16.0 | pytest (TT-NN installed: 187 passed, 1 skipped; tests/test_bfp.py without ttnn: 40 passed, 2 skipped), test_ttnn_pack.py on ttsim (9 tests passed), ruff check and format |
| 2026-11-25 | fix(export): round the LM head as --method and --bfp8 say | 04-lm-head-rounding-options.md, f96fc331b689e200b7738f2ef29cef16 | tt-metal/TT-NN v0.81.1, ttsim v1.16.0 | pytest (TT-NN installed: 229 passed, 1 skipped), test_ttnn_pack.py on ttsim (9 tests passed), ruff check and format |
