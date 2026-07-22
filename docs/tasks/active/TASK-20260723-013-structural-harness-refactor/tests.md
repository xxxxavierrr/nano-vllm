# Tests

## Planned structural evidence

- Checker accepts small synchronous, async, nested, and class methods.
- Checker rejects an over-budget named hot path and a new generic
  orchestration function.
- Explicit numerical-kernel exceptions require a non-empty reason and match
  only the declared path/name.
- Repository report lists legacy debt but all refactored required targets pass.

## Planned behavior evidence

- Benchmark metrics and online MockTransport tests preserve schema-v3 results.
- GPTQ reference/loader/native tests preserve packed checkpoint behavior.
- FP8 DeltaNet reference/capacity/lifecycle tests preserve layouts and errors.
- Scheduler, speculative sampling, hybrid state, API and compile tests pass.
- CUDA/Triton execution is recorded as skipped or pending on this local host,
  never as passed.

## Current evidence

- Structure harness unit tests: `5 passed`.
- Pre-change repository baseline: completed in report-only mode. All seven
  task targets fail their intended final budget; numerical kernels are reported
  as explicit exceptions; unrelated legacy orchestration is being capped at
  its current size rather than treated as compliant.
- Strict post-change hook passes. Required boundaries are: `bench.main=2`,
  `metrics.summarize=33`, `quantize_linear_gptq=9`,
  `ModelRunner.prepare_inputs=6`, `ModelRunner.run=27`,
  `MTPProposer.propose=34`, and `packed_causal_conv1d=27` lines.
- Focused/combined local suite: `87 passed, 1 skipped`; the skip is CUDA and is
  not treated as successful GPU evidence.
- Planner CPU tests cover typed uncached prefill and Piecewise dummy padding
  remaining outside attention metadata. Coordinator tests cover phase-owned
  verification, prefix commit, proposal state, metrics, RNG, and release.
- Failed full collection is environmental evidence only: `flash_attn` is
  absent and one PyTorch Inductor template fails under Windows GBK. A broader
  retry timed out and is not counted as passed.

## Pending GPU evidence

- Compare old/new prepared metadata and outputs on Qwen3.6 for eager,
  Piecewise, Full decode, mixed batches, MTP `k=1..3`, FP8 KV, branch-state
  commit, abort, preemption, and prefix-cache lifecycle.
- Run offline JSON/schema parity and online streaming smoke tests.
- Run DeltaNet and FP8 attention CUDA correctness tests currently unavailable
  locally; the 16 DeltaNet skips observed in a focused command remain pending.
