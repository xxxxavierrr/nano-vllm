# Test evidence

| Area | Status |
| --- | --- |
| Greedy acceptance protocol | pending |
| Proposer helper/structure | pending |
| Scheduler regression | pending |
| GPU MTP accuracy/Graph/performance | unavailable locally |
# Test evidence

## Local static and CPU

- Python bytecode compilation passed for `nanovllm`, `tests`, and `tools`.
- 30 focused tests passed, covering proposer slot mapping across KV pages,
  greedy acceptance ownership, k=1/2/3 scheduler semantics, scoped context,
  hybrid state transactions, and CUDA Graph dispatch policy.

## GPU gates still required

- Qwen3.6 MTP k=1/2/3 token parity against the pre-refactor runner.
- Draft-cache writes across block boundaries and rejected-prefix replay.
- FULL/PIECEWISE/EAGER execution-mode evidence and CUDA Graph replay.
- Acceptance, throughput, TTFT, TPOT, and peak-memory comparison.
