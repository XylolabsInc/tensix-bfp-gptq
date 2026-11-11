# tensix-bfp-gptq

BFP4 and BFP8 weight rounding with GPTQ for Tenstorrent Tensix NPUs (Wormhole, Blackhole).

TT-NN stores weights in block floating point: `ttnn.bfloat4_b` (BFP4) and `ttnn.bfloat8_b`
(BFP8). This package rounds the linear projections of a Hugging Face LLM checkpoint onto those
grids, either to nearest or with GPTQ, and exports an ordinary bf16 safetensors checkpoint whose
values already lie on the grid. TT-NN's packer then stores them without rounding again, and any
bf16 runtime (transformers, a CPU or GPU) runs exactly the weights the NPU will run.

Not affiliated with or endorsed by Tenstorrent.

## The grid

TT-NN multiplies `x @ W` with `W` laid out `[in, out]`, the transpose of a PyTorch `Linear`
weight. In tile layout each block of 16 consecutive values of a row (16 output channels of one
input channel) shares one 8-bit exponent, the largest of the 16, and each value keeps a sign and a
mantissa with an explicit leading one: 3 bits for bfloat4_b, 7 bits for bfloat8_b. That is 4.5
and 8.5 bits per weight. `tensix_bfp.bfp.bfp()` reproduces the host packer of tt-metal bit for bit
(round to nearest even, the mantissa clamped at its maximum, values with a zero exponent field
flushed to zero); `tests/test_ttnn_pack.py` checks it against `ttnn.as_tensor`.

A runtime keeps the values exactly when it packs them in the `[in, out]` layout at their own format
or a wider one: TT-NN's `ttnn.as_tensor`, tt-transformers with `--use_hf_rope`, and the tt-metal
demos at matching dtypes. tt-transformers without `--use_hf_rope` reorders the q and k rows and
re-rounds the fused QKV, and tt-forge's `experimental_weight_dtype` packs PyTorch's `[out, in]`
layout and re-rounds them.

GPTQ (Frantar et al., 2022) rounds one input channel at a time and spreads each column's error
over the columns not yet rounded, through the inverse Hessian of the projection's input. A BFP
block lies inside one column, so GPTQ applies unchanged. For bfloat4_b the default method `exp`
also searches each block's exponent while it rounds: the block's largest exponent `E` and `E - 1`
(the largest values clipped at the top code, the rest on a grid twice as fine) are both tried and
the smaller squared error wins. Every code stays within the mantissa, so the packer, which takes
the block's largest exponent, stores the result exactly.

## Install

```
pip install .                 # torch, safetensors, transformers, tokenizers, numpy, accelerate
pip install ".[ttnn]"         # also ttnn 0.81.1, for tools/pack_tensix.py and the ttnn tests
pip install ".[test]"         # pytest
```

Python 3.11 or later; 3.12 for `.[ttnn]` (ttnn 0.81.1 ships wheels for Linux x86_64 with glibc 2.34
or later; on any other Python or platform the extra installs without ttnn).

ttnn packs on the host but still needs a Tenstorrent card or Tenstorrent's simulator, even without
`--device`. With no card, use [ttsim](https://github.com/tenstorrent/ttsim) (tested with v1.15.1);
device round trips (`--device`, the DRAM test) also need SFPI 7.89.0, tt-metal v0.81.1's version:

```
curl -LO https://github.com/tenstorrent/ttsim/releases/download/v1.15.1/libttsim_bh.so
curl -L -o soc_descriptor.yaml https://raw.githubusercontent.com/tenstorrent/tt-metal/v0.81.1/tt_metal/soc_descriptors/blackhole_140_arch.yaml
export TT_METAL_SIMULATOR=$PWD/libttsim_bh.so TT_METAL_SLOW_DISPATCH_MODE=1
curl -LO https://github.com/tenstorrent/sfpi/releases/download/7.89.0/sfpi_7.89.0_x86_64_debian.txz
tar -xJf sfpi_7.89.0_x86_64_debian.txz -C "$(python -c 'import os, ttnn; print(os.path.dirname(ttnn.__file__))')/runtime"
```

`soc_descriptor.yaml` must sit beside the simulator library for tt-metal older than v0.83.

## Supported models

| config `family` | models | configs |
|---|---|---|
| `qwen3` | Qwen3 dense | `qwen3-4b`, `qwen3-8b` |
| `qwen2` | Qwen2.5 (q, k and v biases kept as stored) | `qwen25-7b` |
| `qwen3_5` | Qwen3.5 / Qwen3.8 dense: Gated DeltaNet and gated attention | `qwen35-4b`, `qwen35-9b`, `qwen38-27b`, `arete-27b` |
| `gemma4` | Gemma 4 E2B, E4B (per-layer inputs, KV sharing), 31B | `gemma4-e2b`, `gemma4-e2b-qat`, `gemma4-e4b`, `gemma4-31b` |
| `gemma4_unified` | Gemma 4 12B | `gemma4-12b` |
| `ministral3` | Ministral 3 (text model, from a text-only copy) | `ministral3-14b` |
| `llama` | Llama 3.x | `llama-3.1-8b` |

Each config in `configs/` names the model, its family, the calibration and evaluation texts, the
GPTQ damping and the variants studied for it.

## Calibration texts

```
python -m tensix_bfp.data fetch [--data DIR]     # default $TENSIX_BFP_DATA or ~/.cache/tensix_bfp/data
python -m tensix_bfp.data check [--data DIR]
```

Six texts: WikiText-2 (raw) train and test rows from the Hugging Face datasets server, and
plain-text extracts of Korean Wikipedia articles (two disjoint title lists). `tensix_bfp/data.py`
holds the exact sources, slices and the SHA-256 of each file as used for the published exports.
Both sources are CC BY-SA and are not shipped here. WikiText-2 fetches are stable; Wikipedia
articles change, so a fresh Korean fetch may differ from the recorded hash, and an export
calibrated on it may differ from a published one.

Calibration takes `[calib] tokens` tokens split evenly over `[calib] texts`, cut into sequences of
`[calib] seq` tokens. Texts are encoded without special tokens; for families whose input starts
with the begin-of-sequence token (Llama, Ministral 3, Gemma 4) each sequence starts with it in
place of its first token, as llama.cpp's perplexity tool cuts its chunks.

## Export

```
python -m tensix_bfp.export --config configs/qwen3-4b.toml --model Qwen3-4B --out Qwen3-4B-BFP4 \
    --variant all=bfp4,lm_head=bf16,gptq [--device cuda:0] [--data DIR]
```

`--model` is the original checkpoint directory. The model runs one decoder layer at a time, so
the device needs room for one layer and the calibration activations, not the whole model. For a
GPTQ variant each layer first runs on the calibration sequences with every projection as a bf16
matmul with float32 accumulation, recording the Hessian of each projection input; the layer is
then rounded and written, and the calibration stream moves on through the original layer.

A variant is a comma list:

- `GROUP=FMT` for the groups `attn_qkv`, `attn_o`, `mlp_gate_up`, `mlp_down`, `lm_head`
  (Qwen3.5 adds `gdn_in`, `gdn_out` and `ab`, the Gated DeltaNet's a and b projections);
- `all=FMT` (every group), `mix=FMT` (the token mixers), `mlp=FMT` (the MLP);
- `GROUP@LAYERS=FMT` for some layers only, LAYERS like `36-39+50`;
- `gptq` (GPTQ for the bfloat4_b groups; without it everything is rounded to nearest) and
  `damp=X` (GPTQ's damping as a fraction of the mean Hessian diagonal; default `[quant] damp`).

Formats: `bf16`, `bfp8`, `bfp4`. Groups not named stay bf16, Qwen3.5's `ab` bfp8. With one
variant in the config, `--variant` may be left out. Other options:

- `--method exp|gptq`: bfloat4_b GPTQ with the exponent search (default) or plain GPTQ;
- `--bfp8 rtn|exp|gptq`: bfloat8_b projections rounded to nearest (default), to nearest with the
  exponent search, or by GPTQ (with `gptq`);
- `--manifest-format 1|2`: 2 (default) records the LM head's source tensor in `quant4.json`;
  1 leaves that key out, as in the earlier published exports;
- `--no-tf32`: leave float32 matmuls on CUDA at full precision (see below).

### Output

| file | content |
|---|---|
| `layers-NN.safetensors` | decoder layer NN: each rounded matrix as a bf16 tensor on its grid, the other tensors as stored |
| `head.safetensors` | the LM head (rounded the same way) and the final norm |
| `rest.safetensors` | every other tensor of the original: the embedding, MTP layers, a vision tower |
| `model.safetensors.index.json` | the weight map; its metadata holds the variant |
| `quant4.json` | the variant, each projection's format per layer, the rounding method, the LM head's format |
| `config.json`, tokenizer files | copies of the original's |

The export loads with `transformers` like the original. In `quant4.json`, `layers` maps each
layer to `{projection: {"format": "bfp4" | "bfp8" | "bf16", "online": false}}`; `rounding` says
how the bfloat4_b (and bfloat8_b) projections were rounded.

### Reproducing a published export

The published exports were computed on a CUDA GPU (compute capability 9.0) with TF32 float32
matmuls enabled, which is this tool's default on CUDA (`--no-tf32` turns it off; the Hessians then
differ in their last bits). Byte-identical results also need the same library versions: torch
2.10, transformers 5.22.1, safetensors 0.8.0, tokenizers 0.23.3, and for Qwen3.5
flash-linear-attention 0.6.0; and the calibration texts with the recorded hashes. Take the
variant from the published `quant4.json`: its `variant` field lists every group's format and
`gptq+damp0.1` (for example `ab=bfp8,attn_o=bfp4,attn_qkv=bfp4,lm_head=bf16,mlp_down=bfp4,mlp_gate_up=bfp4,router=bf16 gptq+damp0.1`
is `all=bfp4,lm_head=bf16,gptq`; `ab` and `router` are defaults that apply to no Qwen3 layer).
`rounding.method` `exp` is `--method exp`, `rounding.bfp8` (when present) is `--bfp8`, and a
manifest without `lm_head_tensor` needs `--manifest-format 1`.

### Ministral 3

Ministral 3's BF16 release is a vision-language checkpoint. Make its text-only copy first:

```
python -m tensix_bfp.ministral3 Ministral-3-14B-Instruct-2512-BF16 Ministral-3-14B-text
python -m tensix_bfp.export --config configs/ministral3-14b.toml --model Ministral-3-14B-text \
    --out Ministral-3-14B-BFP4-MLP --variant mix=bfp8,mlp=bfp4,lm_head=bf16,gptq
```

## Check an export

```
python tools/verify_grid.py EXPORT_DIR            # torch only: every listed matrix on its grid
python tools/pack_tensix.py EXPORT_DIR [--device] # ttnn: pack each matrix, unpack, compare
```

## KL divergence

```
python -m tensix_bfp.kl --config configs/qwen3-4b.toml --base Qwen3-4B --model Qwen3-4B-BFP4 [--device cuda:0]
```

Runs the config's `[eval]` sequences through the bf16 original and the export, both in bf16 under
transformers, and prints per sequence, per text and over all of them the mean KL(bf16 || export)
over the full vocabulary, top-1 agreement and top-5 overlap. "raw" sequences are the text as it
is, cut like calibration; "chat" sequences are the text as the model's reply in its chat
template, scored on the text's tokens only. The export is first checked to load as its index
says and to differ from the original.

## Answer check

```
python -m tensix_bfp.answers run Qwen3-4B --tokenizer Qwen3-4B --out bf16.json
python -m tensix_bfp.answers run Qwen3-4B-BFP4 --tokenizer Qwen3-4B --out export.json
python -m tensix_bfp.answers score export.json --ref bf16.json
```

33 fixed prompts in English and Korean (arithmetic, facts, reasoning, instructions, code) with
known answers (`tensix_bfp/answers.json`), answered greedily. The export fails only where it gets
wrong an answer the bf16 model gets right, or where its text is broken (empty, garbage, a loop,
the wrong language) on an item the bf16 model gets right. A lost answer counts only where the bf16
model was clear at the first token where the two runs part (a top-2 logit gap of at least 1.0);
closer cases are listed as near ties. With `--thinking` runs of both models (`score --think ...
--think-ref ...`), a reasoning item lost with thinking off is pardoned when the export answers it
right with thinking on. `prompts` writes the token ids for another runner, such as one on the NPU.

## Tests

```
pytest                                    # torch and transformers; the ttnn tests skip without ttnn or a chip
python tests/test_ttnn_pack.py            # the ttnn tests alone, without pytest
```

`test_bfp.py` and `test_gptq.py` check the grids, the exponent search and GPTQ on random data;
`test_export_tiny.py` exports tiny random Qwen3 and Llama models end to end on the CPU;
`test_ttnn_pack.py` compares the rounding with `ttnn.as_tensor` for both formats and, where a
device opens, a round trip through its DRAM.

## License

Apache-2.0. See `LICENSE` and `NOTICE`.
