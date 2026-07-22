# Testing and benchmark evidence

Validation reports distinguish:

1. syntax/static checks;
2. CPU unit tests;
3. GPU kernel correctness;
4. model-level greedy/token agreement;
5. CUDA Graph/compile behavior;
6. offline throughput and latency;
7. online TTFT, TPOT, ITL, throughput, and concurrency;
8. memory capacity and peak allocation.

A skipped CUDA test proves import/collection at most; it is never a passing GPU
result. Benchmark records include commit, model, GPU, dtype/quantization, graph
mode, scheduler limits, workload, warmup/repetition counts, and raw result path.

## RTX 4090D baseline evidence

The 2026-07-23 restored server retained PyTorch 2.8.0+cu128, Triton 3.4.0, and
flash-attn 2.8.3.post1. The structural/runtime checkpoint passed an unfiltered
198-test suite; the later native-W4 checkpoint passed 205 tests with the
temporary extension loaded. Model smokes covered Qwen3-0.6B eager and
FULL/PIECEWISE, Qwen3.6-27B GPTQ eager and MTP k=2, sequential BF16/FP8 KV,
and online health/SSE.

These are integration baselines, not goodput results. Short-smoke TTFT/TPOT,
execution-mode labels, or one concurrency point must not be used to select
quantization, speculative `k`, cache dtype, or state dtype.
