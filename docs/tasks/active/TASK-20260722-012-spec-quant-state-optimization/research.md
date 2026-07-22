# Research

## Local implementation findings

### W4A16

`nanovllm/layers/gptq_kernel.py` does not launch a dequantization kernel and a
separate GEMM kernel. Its Triton operator loads packed INT32 weights, expands
INT4 values, reads raw `g_idx`, scales and zero points, constructs a BF16 tile
in registers, and calls `tl.dot` in one launch. It does not allocate a complete
BF16 weight on GPU.

The performance issue is nevertheless real: unpacking, group lookup,
dequantization, and BF16 tile construction repeat for every M tile. The loader
keeps checkpoint-oriented GPTQ layout and performs no Marlin-style runtime
repack or activation permutation.

### Speculative sampling

The current proposer returns argmax draft IDs and discards draft logits. The
acceptance policy only compares draft IDs with target greedy IDs, and MTP
rejects nonzero temperature. There is no probability-ratio acceptance,
residual distribution, or per-request sampling RNG.

### GDN speculative state

`HybridStateManager` owns committed and working slabs. A transaction copies
committed state to working state. Full acceptance copies working state back;
partial rejection calls `ModelRunner._replay_rejected_prefixes`, which reruns
the target model on the accepted prefix to reconstruct the committed GDN
frontier. This preserves correctness but repeats projection, GDN, MLP, and
other target work.

## Upstream vLLM findings

Study pinned to vLLM commit
[`d6dbdb9b0d6e77b9ac4ef9b298d6dfd8f308b583`](https://github.com/vllm-project/vllm/commit/d6dbdb9b0d6e77b9ac4ef9b298d6dfd8f308b583)
on 2026-07-22.

- [`gptq_marlin.py`](https://github.com/vllm-project/vllm/blob/d6dbdb9b0d6e77b9ac4ef9b298d6dfd8f308b583/vllm/model_executor/layers/quantization/gptq_marlin.py)
  separates checkpoint-shaped allocation from a one-time
  `process_weights_after_loading` transformation and delegates execution to a
  selected mixed-precision linear kernel.
- [`rejection_sampler.py`](https://github.com/vllm-project/vllm/blob/d6dbdb9b0d6e77b9ac4ef9b298d6dfd8f308b583/vllm/v1/worker/gpu/spec_decode/rejection_sampler.py)
  accepts target and optional draft logits as a separate runner phase.
- [`rejection_sampler_utils.py`](https://github.com/vllm-project/vllm/blob/d6dbdb9b0d6e77b9ac4ef9b298d6dfd8f308b583/vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py)
  implements probability-ratio acceptance in log space and blockwise
  residual/Gumbel sampling without an extra normalized full softmax tensor.
- [`gdn_attn.py`](https://github.com/vllm-project/vllm/blob/d6dbdb9b0d6e77b9ac4ef9b298d6dfd8f308b583/vllm/v1/attention/backends/gdn_attn.py)
  builds multi-column speculative state indices plus accepted-token metadata.
- [`qwen_gdn_linear_attn.py`](https://github.com/vllm-project/vllm/blob/d6dbdb9b0d6e77b9ac4ef9b298d6dfd8f308b583/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py)
  passes those state indices and accepted counts into convolution/recurrent
  updates. State selection is represented by indices instead of replaying the
  whole target model after rejection.

## Interpretation

The reusable principles are lifecycle separation, lossless rejection sampling,
and indexed speculative state branches. nano-vLLM should not copy vLLM's
registry, platform abstraction, or distributed machinery.

## Target hardware correction

The development GPU is an RTX 4090D with 24 GB, Ada compute capability SM89.
The implementation must target Ada instructions and resource limits; it must
not assume RTX 3090/SM86 or Hopper-only WGMMA/TMA features.

## Existing benchmark foundation and gaps

`benchmarks/metrics.py` already reports request/output-token throughput,
TTFT/TPOT distributions, errors, concurrency, and request goodput under
optional SLOs. `tools/bench_mtp_sweep.py` already records speculative counts,
accepted length, closed-loop concurrency, scheduled tokens in optional step
traces, and chooses output throughput or request goodput.

The current framework does not yet provide SLO-good output tokens/s, accepted
tokens/s, a time-weighted average running-request metric, external GPU
telemetry provenance, or an open-loop offered-load search. Its MTP sweep is
closed-loop, so it can expose saturation and capacity but cannot alone claim a
maximum sustainable request arrival rate under an SLO.
