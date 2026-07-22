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

## GPU evidence

- Existing PyTorch `2.8.0+cu128`, Triton `3.4.0`, and flash-attn `2.8.3.post1` were retained,
  so validation did not replace the known-good environment.
- Focused CPU/Mock regression: `98 passed`; initial CUDA regression exposed two BF16-reference tolerance failures.
- Numeric diagnosis found the Triton kernel within `7.6e-06` of explicit FP32 accumulation while CUDA BF16
  `F.linear` differed by as much as `0.0625`; the test now enforces both baselines with precision-appropriate bounds.
- Final unfiltered suite: `198 passed, 1 Starlette/httpx deprecation warning`.
- Public `nanovllm.LLM` and benchmark imports now have a regression test after GPU smoke exposed a stale moved-type import.
- Qwen3-0.6B eager offline inference completed through the extracted `BatchPlanner` and runner facade.
- `FULL_AND_PIECEWISE` reports 64 prefill tokens in one PIECEWISE step and 14 decode tokens in seven FULL steps,
  with no EAGER fallback after restoring the configured Piecewise token limit.
- Qwen3.6-27B GPTQ loads and completes eager inference with DeltaNet state capacity allocated.
- MTP `k=2` completed three verification rounds with three branch commits, six discarded branch slots,
  and `rejected_prefix_target_replays=0`.
- FP8 KV CUDA tests and engine smoke pass; token capacity increased from 94,720 to 186,624 tokens,
  while the short-run TPOT changed from 32.39 ms to 35.49 ms.
- Online `/health` returned ready and the OpenAI-compatible streaming request emitted SSE content before clean shutdown.

## Remaining limitations outside this refactor

- PIECEWISE dispatch is proven, but Inductor logs still report some regions skipped because CPU arguments reach them;
  per-region captured-key/replay proof remains owned by the semantic CUDA Graph task.
- Small smoke timings are not goodput conclusions; offered-load sweeps and Qwen3.6 SLO performance remain with the
  active optimization task.
