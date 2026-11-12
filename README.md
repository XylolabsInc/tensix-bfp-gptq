<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/xylolabs-lockup-dark.svg">
  <img src="assets/xylolabs-lockup.svg" alt="Xylolabs" height="40">
</picture>

# tensix-bfp-gptq

**Quantize Hugging Face LLMs to TT-NN's BFP4 and BFP8 formats with GPTQ, for Tenstorrent Tensix NPUs**

[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-3776AB)](pyproject.toml)
[![ttnn](https://img.shields.io/badge/ttnn-0.81.1-5B2C83)](https://github.com/tenstorrent/tt-metal/releases/tag/v0.81.1)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-models-FFD21E)](https://huggingface.co/Xylolabs)

`tenstorrent` · `tensix` · `tt-metal` · `ttnn` · `quantization` · `gptq` · `block-floating-point`

</div>

This tool quantizes the linear layers of a Hugging Face LLM to TT-NN's block floating point formats,
`ttnn.bfloat4_b` (BFP4) and `ttnn.bfloat8_b` (BFP8), with GPTQ or round to nearest. The output is an
ordinary bf16 safetensors checkpoint whose values already sit on the grid. TT-NN packs it without
quantizing again, and transformers on a CPU or GPU runs exactly the weights the NPU will run.

## How it works

TT-NN computes `x @ W` with `W` laid out `[in, out]`, the transpose of a PyTorch `Linear` weight. In
tile layout, each block of 16 consecutive values in a row shares one 8-bit exponent, the largest of
the 16. Each value keeps a sign and a mantissa with an explicit leading one: 3 bits for BFP4, 7 for
BFP8, so 4.5 and 8.5 bits a weight. `tensix_bfp.bfp.bfp()` matches tt-metal's host packer bit for
bit, and `tests/test_ttnn_pack.py` checks it against `ttnn.as_tensor`.

GPTQ (Frantar et al., 2022) quantizes one input channel at a time and spreads each column's error
over the columns not yet quantized. A block lies inside one column, so GPTQ applies unchanged. For
BFP4 the default method `exp` also tries each block's exponent one step lower and keeps whichever
gives the smaller error. One step lower clips the largest values at the top code and puts the rest
on a grid twice as fine.

The values stay exact wherever a runtime packs them as `[in, out]` at their own format or a wider
one: `ttnn.as_tensor`, tt-transformers with `--use_hf_rope`, and the tt-metal demos at matching
dtypes. tt-transformers without `--use_hf_rope` and tt-forge's `experimental_weight_dtype`
re-quantize them.

## Install

```
pip install .            # torch, safetensors, transformers, tokenizers, accelerate
pip install ".[ttnn]"    # also ttnn 0.81.1, for tools/pack_tensix.py and the ttnn tests
pip install ".[test]"    # pytest
```

This needs Python 3.11 or later and is tested on 3.11 to 3.15. `.[ttnn]` needs Python 3.12: ttnn
0.81.1 ships wheels only for CPython 3.10 and 3.12, on Linux x86_64 with glibc 2.34 or later. On
any other Python or platform the extra installs without ttnn.

It is tested on Ubuntu 24.04. tt-metal v0.81.1 is built for Ubuntu 22.04 and 24.04; Ubuntu 26.04
is not on that list, though its glibc is new enough for the wheel. Its system Python is 3.14, so
`.[ttnn]` there needs a separate Python 3.12, such as `uv venv --python 3.12`.

ttnn packs on the host but still needs a Tenstorrent card or Tenstorrent's simulator. Without a
card, use [ttsim](https://github.com/tenstorrent/ttsim) (tested with v1.16.0). Device round trips
(`--device`, the DRAM test) also need SFPI 7.89.0:

```
curl -LO https://github.com/tenstorrent/ttsim/releases/download/v1.16.0/libttsim_bh.so
curl -L -o soc_descriptor.yaml https://raw.githubusercontent.com/tenstorrent/tt-metal/v0.81.1/tt_metal/soc_descriptors/blackhole_140_arch.yaml
export TT_METAL_SIMULATOR=$PWD/libttsim_bh.so TT_METAL_SLOW_DISPATCH_MODE=1
curl -LO https://github.com/tenstorrent/sfpi/releases/download/7.89.0/sfpi_7.89.0_x86_64_debian.txz
tar -xJf sfpi_7.89.0_x86_64_debian.txz -C "$(python -c 'import os, ttnn; print(os.path.dirname(ttnn.__file__))')/runtime"
```

For tt-metal older than v0.83, `soc_descriptor.yaml` must sit beside the simulator library.

## Supported models

| config `family` | models | configs |
|---|---|---|
| `qwen3` | Qwen3 dense | `qwen3-4b`, `qwen3-8b` |
| `qwen2` | Qwen2.5 (q, k and v biases kept as stored) | `qwen25-7b` |
| `qwen3_5` | Qwen3.5 / Qwen3.8 dense (Gated DeltaNet and gated attention) | `qwen35-4b`, `qwen35-9b`, `qwen38-27b`, `arete-27b` |
| `gemma4` | Gemma 4 E2B, E4B, 31B | `gemma4-e2b`, `gemma4-e2b-qat`, `gemma4-e4b`, `gemma4-31b` |
| `gemma4_unified` | Gemma 4 12B | `gemma4-12b` |
| `ministral3` | Ministral 3 (text model, from a text-only copy) | `ministral3-14b` |
| `llama` | Llama 3.x | `llama-3.1-8b` |

Each config in `configs/` names the model, its family, the calibration and evaluation texts, the
GPTQ damping and the variants studied for it.

## Usage

### Calibration texts

```
python -m tensix_bfp.data fetch [--data DIR] [--extra]   # default $TENSIX_BFP_DATA or ~/.cache/tensix_bfp/data
python -m tensix_bfp.data check [--data DIR] [--extra]
```

`--extra` also fetches two more calibration texts, `calib3_en` and `calib3_ko`, which
`configs/llama-3.1-8b-cal4x.toml` uses for four times the calibration tokens (262,144).

The texts are WikiText-2 (train and test; CC BY-SA 3.0 and GFDL) and extracts of Korean Wikipedia
articles (CC BY-SA 4.0, most text also GFDL). Both are downloaded from their sources, not shipped
here. `tensix_bfp/data.py` records each file's source and SHA-256. Wikipedia articles change, so a
fresh Korean fetch may not match the recorded hash.

Calibration takes `[calib] tokens` tokens, split evenly over `[calib] texts` in sequences of
`[calib] seq`. For Llama, Ministral 3 and Gemma 4, each sequence starts with the begin-of-sequence
token, as in llama.cpp's perplexity tool.

### Export

```
python -m tensix_bfp.export --config configs/qwen3-4b.toml --model Qwen3-4B --out Qwen3-4B-BFP4 \
    --variant all=bfp4,lm_head=bf16,gptq [--device cuda:0] [--data DIR]
```

`--model` is the original checkpoint. One decoder layer runs at a time, so the GPU needs room for
one layer and the calibration activations, not the whole model.

A variant is a comma list:

- `GROUP=FMT` for `attn_qkv`, `attn_o`, `mlp_gate_up`, `mlp_down` and `lm_head` (Qwen3.5 adds
  `gdn_in`, `gdn_out` and `ab`);
- `all=FMT`, `mix=FMT` (the token mixers) and `mlp=FMT`;
- `GROUP@LAYERS=FMT` for some layers only, with LAYERS like `36-39+50`;
- `gptq` to quantize the BFP4 groups with GPTQ (otherwise with round to nearest), and `damp=X` for
  its damping.

The formats are `bf16`, `bfp8` and `bfp4`. Unnamed groups stay bf16 (Qwen3.5's `ab` stays bfp8).

The other options:

- `--method exp|gptq`: BFP4 GPTQ with the exponent search (the default) or without it.
- `--bfp8 rtn|exp|gptq`: BFP8 with round to nearest (the default), with the exponent search, or with
  GPTQ.
- `--manifest-format 1|2`: whether `quant4.json` records the LM head's source tensor (2, the
  default, does).
- `--no-tf32`: turns off TF32 for float32 matmuls on CUDA.

The output loads with transformers like the original:

| file | content |
|---|---|
| `layers-NN.safetensors` | decoder layer NN, each quantized matrix as bf16 on its grid |
| `head.safetensors` | the LM head and the final norm |
| `rest.safetensors` | every other tensor: the embedding, MTP layers, a vision tower |
| `model.safetensors.index.json` | the weight map, with the variant in its metadata |
| `quant4.json` | each projection's format per layer and the quantization method |
| `config.json`, tokenizer files | copied from the original |

Ministral 3's BF16 release is a vision-language checkpoint, so it needs a text-only copy first:

```
python -m tensix_bfp.ministral3 Ministral-3-14B-Instruct-2512-BF16 Ministral-3-14B-text
```

### Check an export

```
python tools/verify_grid.py EXPORT_DIR             # torch only: every listed matrix on its grid
python tools/pack_tensix.py EXPORT_DIR [--device]  # ttnn: pack each matrix, unpack, compare
```

### KL divergence

```
python -m tensix_bfp.kl --config configs/qwen3-4b.toml --base Qwen3-4B --model Qwen3-4B-BFP4 [--device cuda:0]
```

This runs the config's evaluation sequences through the bf16 original and the export. It prints
the mean KL(bf16 || export) over the full vocabulary, top-1 agreement and top-5 overlap, per
sequence, per text and overall. Sequences come raw, or as the model's reply in its chat template.

### Answer check

```
python -m tensix_bfp.answers run Qwen3-4B --tokenizer Qwen3-4B --out bf16.json
python -m tensix_bfp.answers run Qwen3-4B-BFP4 --tokenizer Qwen3-4B --out export.json
python -m tensix_bfp.answers score export.json --ref bf16.json
```

The check runs 33 fixed prompts in English and Korean with known answers, answered greedily
(`tensix_bfp/answers.json`; one paraphrases Frederick's bat-and-ball problem, 2005). An export loses
an item when it gets wrong, or breaks, an answer the bf16 model gets right. Items where bf16's top-2
logit gap is under 1.0 are near ties and do not count. With `run --thinking` for both models
(`score --think ... --think-ref ...`), a reasoning item lost with thinking off is pardoned when the
export gets it right with thinking on. `prompts` writes the token ids for another runner.

## Reproducing a published export

The published exports were made on CUDA (compute capability 9.0) with TF32 on, which is the default
here, using torch 2.10, transformers 5.22.1, safetensors 0.8.0, tokenizers 0.23.3 (and
flash-linear-attention 0.6.0 for Qwen3.5), with calibration texts that match the recorded hashes.
Take the variant from the export's `quant4.json`. Its `rounding.method` maps to `--method` and
`rounding.bfp8` to `--bfp8`. If the manifest has no `lm_head_tensor`, pass `--manifest-format 1`.

## Tests

```
pytest                           # the ttnn tests skip without ttnn or a chip
python tests/test_ttnn_pack.py   # the ttnn tests alone
```

## License

Apache-2.0, see `LICENSE`. This is an independent project, not affiliated with Tenstorrent.
