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
- DSpark local tests cover config/forward shapes, strict missing/unknown weight
  mapping, append/resume/cache hashes, FP32 Hessian accumulation, GPTQ packing
  and reconstruction, sharded checkpoint index/config, dry-run projection, and
  a generated linear loaded by the production GPTQ loader (`20 passed, 1 CUDA
  skipped` with GPTQ regressions).

Deferred evidence:

- The calibration model is a strict local shell. Real Avesed tensor mapping,
  DFlash/Markov logits, per-layer reconstruction, 8.8 GB BF16 calibration time,
  final draft size, and online acceptance/goodput are all pending; synthetic
  round-trip is not counted as real-model compatibility.

- New recurrent and causal-conv prefix-state CUDA tests are written but skipped
  locally because CUDA is unavailable.
- Repacked W4 Triton correctness/compile/Graph tests are written but skipped.
- Native `.cu` compilation, numerical equivalence, tensor-core utilization,
  Full/Piecewise Graph behavior, and latency are pending on RTX 4090D. Source
  presence and CPU dispatch tests are not CUDA validation.
- Full/Piecewise Graph replay, end-to-end Qwen3.6 accuracy, 24 GB capacity,
  online serving, and all performance benchmarks require the RTX 4090D server.
- Full local collection is additionally blocked by missing `flash_attn`; broad
  regression commands that timed out are not counted as successful.
- No W4A16/W4A8, FP8 capacity, GPU-utilization, or SLO-throughput number exists
  while the RTX 4090D server is unavailable; theoretical compression is not a
  benchmark result.
