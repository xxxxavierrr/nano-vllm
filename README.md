<p align="center">
<img width="300" src="assets/logo.png">
</p>

<p align="center">
<a href="https://trendshift.io/repositories/15323" target="_blank"><img src="https://trendshift.io/api/badge/repositories/15323" alt="GeeeekExplorer%2Fnano-vllm | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

# Nano-vLLM

A lightweight vLLM implementation built from scratch.

## Key Features

* 🚀 **Fast offline inference** - Comparable inference speeds to vLLM
* 📖 **Readable codebase** - Clean implementation in ~ 1,200 lines of Python code
* ⚡ **Optimization Suite** - Prefix caching, Tensor Parallelism, Torch compilation, CUDA graph, etc.

## Installation

```bash
pip install git+https://github.com/GeeeekExplorer/nano-vllm.git
```

## Model Download

To download the model weights manually, use the following command:
```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

## Quick Start

See `example.py` for usage. The API mirrors vLLM's interface with minor differences in the `LLM.generate` method:
```python
from nanovllm import LLM, SamplingParams
llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)
outputs[0]["text"]
```

## FP8 Quantization

On GPUs with native FP8 support, linear weights and activations can be quantized
to FP8 while embeddings, normalization, logits, and the KV cache remain in the
model's original dtype:

```python
llm = LLM("/YOUR/MODEL/PATH", quantization="fp8", enforce_eager=True)
```

## Benchmark

`bench.py` provides an in-process synthetic benchmark with offline burst or
rate-controlled arrivals, concurrency limits, reproducible length distributions,
shared prefixes, SLO goodput, and JSON output. It reports request/input/output/
total throughput, TTFT, TPOT, ITL, E2E latency percentiles, prefill/decode
throughput, scheduler behavior, prefix-cache hits, and GPU memory.

On the development GPU, `bench.py` can be run directly without arguments; it
defaults to `/root/autodl-tmp/huggingface/Qwen3-0.6B` and the settings shown by
`python bench.py --help`.

```bash
python bench.py /YOUR/MODEL/PATH \
  --num-requests 64 --input-len 256 --output-len 128 \
  --max-concurrency 16 --output-json results/benchmark.json
```

Use `--request-rate inf` for offline maximum throughput, or a finite request
rate for Poisson arrivals. Add `--quantization fp8` to benchmark FP8 and use
`--ttft-slo-ms`, `--tpot-slo-ms`, or `--e2e-slo-ms` to report SLO goodput.

**Test Configuration:**
- Hardware: RTX 4070 Laptop (8GB)
- Model: Qwen3-0.6B
- Total Requests: 256 sequences
- Input Length: Randomly sampled between 100–1024 tokens
- Output Length: Randomly sampled between 100–1024 tokens

**Performance Results:**
| Inference Engine | Output Tokens | Time (s) | Throughput (tokens/s) |
|----------------|-------------|----------|-----------------------|
| vLLM           | 133,966     | 98.37    | 1361.84               |
| Nano-vLLM      | 133,966     | 93.41    | 1434.13               |


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=GeeeekExplorer/nano-vllm&type=Date)](https://www.star-history.com/#GeeeekExplorer/nano-vllm&Date)
