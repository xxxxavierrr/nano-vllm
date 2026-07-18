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
llm = LLM("/YOUR/MODEL/PATH", tensor_parallel_size=1)
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

## GPTQ W4A16

GPTQ checkpoints are auto-detected from their Hugging Face
`quantization_config`; `quantization="gptq"` and `--quantization gptq` are
also accepted. The v1 path requires 4-bit symmetric INT32 packing,
`group_size=128`, BF16 activations, and tensor parallel size 1. Linear layers
keep `qweight`, `scales`, `qzeros`, and `g_idx` packed on GPU and never
materialize a full BF16 weight.

The v1 Triton kernel reads the original `g_idx` directly. Fused QKV and
gate/up projections are loaded only when every source shard has an identical
`g_idx`; a mismatch is rejected instead of silently overwriting metadata.

Run the implementation-independent kernel comparison with:

```bash
python tools/bench_gptq_kernel.py
```

The current model runtime still implements the Qwen3 architecture. Qwen3.6
uses the Qwen3.5 architecture and will be connected in a later milestone.

## OpenAI-Compatible Serving

Install the optional serving dependencies and start the API server with one
command. The API process tokenizes requests and communicates with a dedicated
GPU engine process over local ZMQ.

```bash
pip install -e ".[serve]"
nano-vllm-serve --model /YOUR/MODEL/PATH
```

The module entrypoint is equivalent:

```bash
python -m nanovllm.serve.api_server --model /YOUR/MODEL/PATH
```

The server listens on `127.0.0.1:8000` by default. Use `--host 0.0.0.0` to
expose it on the network. Chat Completions supports both streaming and
non-streaming requests:

On the GPU server, the convenience script uses the existing miniconda GPU
environment and installs only missing Web dependencies without a pip cache:

```bash
cd /root/nano-vllm
tools/start_server.sh
```

From a Windows checkout, this command opens a temporary SSH tunnel, sends one
prompt, prints the streaming response, and closes the tunnel:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\chat_gpu.ps1 "用一句话介绍你自己"
```

Alternatively, open `ssh -N -L 8000:127.0.0.1:8000 gpu` in one local terminal
and run `python tools/chat_stream.py "Hello"` in another.

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen3-0.6B","messages":[{"role":"user","content":"Hello"}],"stream":true}'
```

## CUDA Graph modes

`FULL_AND_PIECEWISE` is the default execution mode. Uniform decode batches use
the existing full-model CUDA Graph. Prefill and mixed batches up to 512
scheduled tokens use Piecewise CUDA Graphs with attention and KV-cache updates
left eager; larger batches fall back to eager execution.

Use `cudagraph_mode="FULL_DECODE_ONLY"`, `"PIECEWISE"`, or `"NONE"` when
constructing `LLM`. The server and offline benchmark expose the equivalent
`--cudagraph-mode` and `--piecewise-max-tokens` options. `enforce_eager=True`
and `--enforce-eager` remain aliases for `NONE`. The server convenience script
also accepts the `CUDAGRAPH_MODE` and `PIECEWISE_MAX_TOKENS` environment
variables.

Capturing all buckets through 512 deliberately moves compilation work into
startup so the first online request does not compile a new graph. For faster
development restarts, lower `piecewise_max_tokens`; production startup should
allow several minutes for the one-time capture.

## Benchmark

`bench.py` provides an in-process synthetic benchmark with offline burst or
rate-controlled arrivals, concurrency limits, reproducible length distributions,
shared prefixes, SLO goodput, and JSON output. It reports request/input/output/
total throughput, TTFT, TPOT, ITL, E2E latency percentiles, prefill/decode
throughput, scheduler behavior, prefix-cache hits, GPU memory, and per-mode
`FULL`/`PIECEWISE`/`EAGER` step, token, and timing counters.

On the development GPU, `bench.py` can be run directly without arguments; it
defaults to `/root/autodl-tmp/huggingface/Qwen3-0.6B` and the settings shown by
`python bench.py --help`.

```bash
python bench.py /YOUR/MODEL/PATH \
  --num-requests 64 --input-len 256 --output-len 128 \
  --max-concurrency 16 --output-json results/benchmark.json
```

Use `--request-rate inf` for offline maximum throughput, or a finite request
rate for Poisson arrivals. Add `--quantization fp8` or `--quantization gptq`
to select a quantized checkpoint and use `--ttft-slo-ms`, `--tpot-slo-ms`, or
`--e2e-slo-ms` to report SLO goodput.

### Online serving benchmark

`bench_online.py` is a black-box OpenAI Chat Completions benchmark. It does not
import nano-vLLM engine code, so the same client can test any compatible server
and compare models, operators, tensor-parallel settings, or quantization modes.
Install the small client dependency with `pip install -e ".[benchmark]"` and
run the built-in smoke profile:

```bash
python bench_online.py --base-url http://127.0.0.1:8000 \
  --model Qwen3-0.6B --profile smoke
```

Available profiles are `smoke`, `latency`, `throughput`, and `prefix_cache`.
Every setting can be overridden from the command line:

```bash
python bench_online.py --base-url http://127.0.0.1:8000 \
  --model Qwen3-0.6B --profile throughput \
  --num-requests 100 --max-concurrency 16 --request-rate 8 \
  --metadata framework=nano-vllm --metadata quantization=fp8 \
  --output-json results/fp8-online.json --request-details
```

The benchmark reports request and token throughput, TTFT, TPOT, inter-chunk and
end-to-end latency percentiles, client-side queueing, errors, SLO goodput, and
prefix-cache usage when the server returns it. Synthetic input lengths are
approximate across tokenizers; the result always records whether output token
counts came from API usage or an SSE-chunk fallback.

On Windows, run the same benchmark through a temporary SSH tunnel:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\bench_gpu.ps1 `
  -Profile smoke -Model Qwen3-0.6B
```

### Unified benchmark entrypoint

Use `benchmark.py` when automation needs to choose the execution boundary
explicitly. Offline mode runs the nano-vLLM engine in-process and retains
engine-only phase, scheduler, cache, and memory metrics:

```bash
python benchmark.py --mode offline /YOUR/MODEL/PATH \
  --num-requests 64 --input-len 256 --output-len 128 \
  --metadata quantization=bf16 --output-json results/offline.json
```

Online mode runs the implementation-independent OpenAI HTTP/SSE client:

```bash
python benchmark.py --mode online \
  --base-url http://127.0.0.1:8000 --model Qwen3-0.6B \
  --profile throughput --metadata quantization=bf16 \
  --output-json results/online.json
```

Both modes write schema version 2 with common `mode`, `metadata`, `workload`,
and `metrics` sections. Offline-only details are stored under `engine_metrics`.
The original `bench.py` and `bench_online.py` entrypoints remain available.

On the GPU server, the convenience wrapper selects the existing miniconda
Python and the default Qwen3 model automatically:

```bash
tools/bench_server.sh online --profile smoke
tools/bench_server.sh offline --num-requests 64 --input-len 256 --output-len 128
```

Offline mode requires exclusive access to the GPU. The wrapper refuses to
start it while the online service is healthy, instead of unexpectedly killing
the running server.

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
