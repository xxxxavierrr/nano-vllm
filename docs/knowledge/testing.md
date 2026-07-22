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
