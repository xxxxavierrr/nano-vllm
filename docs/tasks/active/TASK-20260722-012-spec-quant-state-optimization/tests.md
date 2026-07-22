# Tests

## Planned state tests

- Every acceptance length `a in [0,k]` selects exactly the state after `1+a`
  scheduled inputs for both conv and recurrent state.
- Unselected slots are released; abort, preemption, pool exhaustion, prefix
  reuse, and mixed spec/non-spec batches do not alias state.
- Branch-state output matches an eager no-spec target replay baseline, while
  production counters prove zero rejected-prefix model replays.
- Full and Piecewise Graph replay use stable state-index buffers.

## Planned sampling tests

- Greedy behavior remains byte-for-byte compatible.
- Probability-ratio decisions and `(p-q)+` recovery match a PyTorch reference,
  including zero residual entries, identical distributions, tiny probability,
  and all-accepted bonus sampling.
- Seeded results are invariant to request batching order.
- Monte Carlo token frequencies match direct target sampling within a stated
  confidence bound; non-greedy correctness is not judged by token equality.

## Planned W4 tests and benchmark

- Post-load repack is elementwise equivalent to the GPTQ reference for normal
  and non-monotonic `g_idx`; fused shard validation remains strict.
- Repacked Triton and CUDA outputs meet the existing BF16 error thresholds for
  representative Qwen3.6 shapes and `M=1,8,32,128,512`.
- Assert no global full-BF16 weight/scratch allocation in production.
- Compare raw Triton, repacked Triton, CUDA backend, and BF16 on latency,
  effective bandwidth, throughput, workspace, and end-to-end TTFT/TPOT.
- On the RTX 4090D 24 GB target, report weight, Graph, KV, committed GDN state,
  speculative branch state, draft-logit workspace, free-memory margin, and
  maximum stable concurrency separately.

## Planned goodput and capacity evidence

- Aggregation unit tests cover output/accepted/request rates, SLO-good output
  tokens, p50/p99 TTFT/TPOT, errors, and time-weighted running requests from a
  deterministic synthetic event trace.
- Per-step telemetry records scheduled actual and padded tokens, running
  requests, execution mode, speculative work, and preemptions without reading
  mutable `last_*` side channels.
- Closed-loop sweeps cover concurrency `1,2,4,8,...` through capacity/SLO
  failure. Open-loop sweeps increase offered request rate, then refine around
  the highest rate satisfying the declared p99 TTFT/TPOT and error SLO.
- GPU utilization samples record tool, sampling interval, start/end alignment,
  compute utilization, and memory utilization; absence is explicit in JSON.
- W4A8 large-M compares kernel correctness, workspace, prefill/mixed latency,
  and end-to-end goodput against W4A16. Small-M remains W4A16 unless separately
  justified.
- FP8 KV and FP8 DeltaNet state compare both equal-concurrency overhead and
  best-stable-concurrency goodput. Report capacity gained, conversion/kernel
  cost, scheduler occupancy, and accuracy/state error.
- W3A16 has no implementation gate until the post-FP8 memory report shows that
  model weight bytes still prevent the target concurrency.

## DSpark evidence under the 24 GB limit

- Do not schedule an online BF16-draft-plus-target run; it is not a feasible
  cell on the target GPU.
- Persist target-produced DSpark inputs while the target is loaded alone. Load
  BF16 and INT4 drafts sequentially and compare logits, top-1 agreement, KL,
  first divergence, and per-layer reconstruction error on identical inputs.
- Online acceptance/goodput cells hold the INT4 draft fixed and compare paired
  AWQ versus nano GPTQ targets. Separate INT4 group-size/mixed-precision draft
  variants measure the recoverable acceptance-versus-memory tradeoff.
- Published BF16-draft acceptance is labeled external reference data and never
  presented as a locally reproduced baseline.

## Evidence status

Implemented local evidence:

- State manager tests cover request capacity, branch reservation, every commit
  prefix for `k=3`, release, pool exhaustion, and slot reuse.
- Sampling tests cover identical distributions, forced residual recovery,
  seeded reproducibility, and a 6,000-sample target-distribution comparison.
- GPTQ CPU tests cover non-monotonic `g_idx`, argsort permutation, packed
  qweight repack, fused shard loading, and absence of BF16 production weights.
- Scheduler/greedy compatibility and sampling tests pass in the focused CPU
  suite.
- Python compilation and `git diff --check` pass.
- Schema v3 aggregation tests prove planned-arrival TTFT/E2E includes client
  queueing while service latency remains separately reported.
- MockTransport exercises the complete online runner; a framework-neutral fake
  backend exercises the load generator; deterministic sweep tests cover
  exponential growth and binary refinement.
- Fake-reader telemetry tests cover sampling/reporting without requiring
  `nvidia-smi`; the focused goodput suite passes (`12 passed`).
- Native W4 CPU tests cover safe default fallback, explicit extension failure,
  SM89 gating, the `M=64/65` shape boundary, normalized layout rejection, and
  per-row/per-group W4A8 activation quantization (`32 passed, 17 CUDA skipped`
  with the wider GPTQ/config/API regression set).
- Native W4A16 now has real SM89 evidence: the CUDA 12.8 extension compiles,
  repacked non-monotonic `g_idx` uses the fused activation permutation, all
  tested `M=1,8,19,64,65,128,512` shapes meet BF16 tolerance, and both
  `torch.compile(fullgraph=True)` and direct CUDA Graph replay match eager.
  The formal native plus Triton GPU regression passes (`29 passed`). The full
  server suite with the native extension loaded passes (`205 passed, 1
  deprecation warning`); skipped source-only evidence is no longer used to
  claim W4A16 CUDA correctness.
- The first WMMA packed-word prototype is materially faster than the scalar
  scaffold but remains slower than repacked Triton on `K=N=5120`; therefore
  `auto` intentionally remains Triton and native remains explicit opt-in.
- Later tile experiments retained small `16x64x128` and large
  `32x128x32` as the best tested combination. Direct numerical, fullgraph,
  and CUDA Graph evidence remains valid; the rejected `16x32x32` small
  candidate is not retained.
- The final retained source was rebuilt and the native/Triton focused suite
  rerun successfully (`29 passed`). Required structure/compile/diff checks and
  all 126 documentation relative links pass after archival.
- DSpark local tests cover config/forward shapes, strict missing/unknown weight
  mapping, append/resume/cache hashes, FP32 Hessian accumulation, GPTQ packing
  and reconstruction, sharded checkpoint index/config, dry-run projection, and
  a generated linear loaded by the production GPTQ loader (`20 passed, 1 CUDA
  skipped` with GPTQ regressions).
- State/sampler/KV strengthening covers atomic multi-request prefix selection,
  exact branch request sets, zero replay instrumentation, 50 randomized
  sampler shapes, FP8 scale overhead, theoretical 1.984x head-dim-256 target
  KV compression, and reduced combined gain when native MTP KV remains BF16
  (`31 passed, 9 CUDA skipped`).
- FP8 DeltaNet tests cover scale granularity, zero/non-finite safety,
  quantize/dequantize error, payload/scale/branch capacity, committed-prefix
  remap using the same slot IDs, fail-closed experimental kernels, hybrid-only
  config, API regression, and the standalone capacity CLI (`37 passed`).
- Final combined selected regression: `83 passed, 26 CUDA skipped`; skipped
  tests are recorded as unavailable, not successful.

Deferred evidence:

- The calibration model is a strict local shell. Real Avesed tensor mapping,
  DFlash/Markov logits, per-layer reconstruction, 8.8 GB BF16 calibration time,
  final draft size, and online acceptance/goodput are all pending; synthetic
  round-trip is not counted as real-model compatibility.

- New recurrent and causal-conv prefix-state CUDA tests are written but skipped
  locally because CUDA is unavailable.
- Repacked W4 Triton correctness/compile/Graph tests are written but skipped.
- Native W4A8 numerical/Graph/performance validation, profiler-confirmed
  pipeline/tensor-core utilization, true Piecewise integration inside the
  complete model, and a real Marlin-layout/multi-stage W4A16 implementation
  remain pending. Direct fullgraph/CUDA Graph evidence is complete, but the
  native prototype does not beat Triton.
- FP8 DeltaNet conversion kernels are not fused into production conv/recurrent
  execution and the runtime intentionally rejects explicit enablement. SM89
  kernel correctness, error over long sequences, Graph stability, memory
  capacity, scheduler batch, and goodput are pending.
- Full/Piecewise Graph replay, end-to-end Qwen3.6 accuracy, 24 GB capacity,
  online serving, and all performance benchmarks require the RTX 4090D server.
- Full local collection is additionally blocked by missing `flash_attn`; broad
  regression commands that timed out are not counted as successful.
- W4A16 now has a raw-kernel SM89 latency baseline. W4A8, FP8 capacity,
  GPU-utilization, and SLO-throughput measurements remain pending; theoretical
  compression is not a benchmark result.
