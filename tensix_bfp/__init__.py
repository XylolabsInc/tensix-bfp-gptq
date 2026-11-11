# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Xylolabs
"""BFP4 and BFP8 (TT-NN bfloat4_b and bfloat8_b) weight rounding with GPTQ, for Tenstorrent Tensix
NPUs (Wormhole, Blackhole). Not affiliated with or endorsed by Tenstorrent.

Modules: bfp (the block floating point grids), gptq (GPTQ onto them), export (the checkpoint
export), data (the calibration and evaluation texts), kl (KL divergence against the bf16 model),
answers (the answer check), ministral3 (Ministral 3's text-only copy).
"""

__version__ = "0.1.0"
