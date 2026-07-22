---
subject: benchmarking
title: Inference benchmark and goodput methodology
status: active
created: 2026-07-22
updated: 2026-07-23
owner: Codex
---

# Inference benchmark and goodput methodology

## Motivation

Kernel latency and single-request token throughput do not determine serving
capacity. Weight, KV-cache, recurrent-state, speculative, scheduler, and CUDA
Graph optimizations compete for the same 24 GB memory budget and can trade
per-step latency for more useful concurrency. nano-vLLM therefore chooses
production optimizations by end-to-end goodput under a declared latency SLO.

## Requirements

- The primary comparison is goodput under a fixed workload and latency SLO,
  not an isolated kernel speedup.
- A successful request is SLO-good only when it completes without error and
  satisfies every configured TTFT, TPOT, and optional end-to-end threshold.
- Report both `goodput_request_per_s` and `goodput_output_token_per_s`. Raw
  request/output-token throughput remains visible but cannot select a winner
  when an SLO is configured.
- Every speculative run additionally reports proposed, accepted, rejected,
  and bonus tokens, accepted tokens/s, verification rounds, mean accepted
  length, and conditional acceptance by speculative position. Accepted
  tokens/s is diagnostic and cannot replace output-token goodput.
- Scheduler utilization is time-weighted and reports average/max running
  requests, scheduled tokens per step, actual versus padded tokens, preemption,
  and execution-mode hit rates. A sample average over step boundaries is not a
  substitute for the time integral of active requests.
- Report GPU compute utilization and memory utilization from an external or
  independent sampler with its interval and provenance. When available,
  memory-bandwidth and power samples are recorded separately.
- Report TTFT and TPOT distributions including p50 and p99. Online results also
  retain inter-token, queue, end-to-end, error-rate, and HTTP status evidence.
- Determine maximum SLO throughput with an open-loop offered-load sweep. A
  closed-loop concurrency sweep is retained for saturation diagnosis, capacity
  limits, and controlled component ablations, but does not alone establish
  maximum arrival-rate goodput.
- Offline and online benchmarks share workload descriptions and output schema.
  Offline runs measure engine/kernel ceilings and scheduler behavior; online
  runs include transport, serialization, tokenization, queues, streaming, and
  process boundaries and are authoritative for serving SLO claims.
- Comparisons use the same prompts, input/output length distribution, sampling
  policy, prefix-cache policy, scheduler limits, Graph mode, warmup, repetition,
  and model revision unless the differing field is the declared independent
  variable.
- Each result records commit, model/checkpoint hashes, target and draft
  quantization, KV/state dtype, speculative method and `k`, GPU/software
  environment, memory budget, scheduler limits, arrival process, raw artifact
  path, and unavailable metrics.
- No GPU performance result is inferred from CPU tests, static inspection, or
  theoretical byte ratios. Missing hardware evidence is marked pending.
- Request timing and SLO eligibility are derived once into immutable facts.
  Request goodput, output-token goodput, latency distributions, and reporting
  consume those same facts rather than reimplementing the SLO predicate.
- Offline engine execution, aggregation, result assembly, and presentation are
  separate boundaries; the CLI entry point does not own the benchmark loop.

## Scope

- Offline closed-loop component and engine benchmarks.
- Online closed-loop concurrency and open-loop offered-load sweeps.
- Quantization, speculative decoding, CUDA Graph, KV-cache, and DeltaNet-state
  comparisons on the RTX 4090D 24 GB target.
- Machine-readable JSON summaries plus raw per-request, per-step, and GPU
  telemetry needed to recompute aggregates.

## Non-goals

- Selecting a kernel solely from synthetic peak TFLOPS or bandwidth.
- Treating acceptance rate as user-visible throughput.
- Claiming production capacity from a single prompt, concurrency, or run.

## Metric definitions

For wall-clock measurement interval `T`:

```text
output_token_per_s          = completed output tokens / T
accepted_token_per_s        = accepted speculative draft tokens / T
request_per_s               = successful requests / T
goodput_request_per_s       = SLO-good requests / T
goodput_output_token_per_s  = output tokens belonging to SLO-good requests / T
average_running_requests    = integral(running_requests(t), dt) / T
```

Online SLO TTFT is planned client arrival to first output token, so client-side
concurrency queueing is included. SLO E2E uses the same arrival origin. The
schema also reports transport/service TTFT and E2E from actual HTTP dispatch,
plus client queue latency, so saturation can be localized. TPOT is the
per-request mean interval after the first token unless a result explicitly
declares another definition. The raw event stream is retained so alternative
distributions can be recomputed.

## Workload matrix

At minimum, performance decisions cover:

- input/output shapes: short chat, medium generation, and long-context decode;
- sampling: greedy correctness and temperature sampling where supported;
- concurrency/capacity sweeps up to allocation or SLO failure;
- speculative `k=0` baseline and supported `k` values;
- BF16/FP8 KV and BF16/FP8 DeltaNet state as independent variables;
- target/draft quantization and CUDA Graph mode as explicit dimensions.

The final capacity run increases offered request rate until error rate or a
latency threshold fails, then refines around the highest passing rate. The same
workload seed and sufficient run duration are used on both sides of an
optimization comparison.

## Acceptance criteria

- The benchmark JSON contains all primary metrics and enough raw evidence to
  recompute them.
- An optimizer is enabled by default only when it improves or preserves maximum
  SLO goodput on a representative workload without violating correctness or
  memory safety.
- Capacity-increasing FP8 KV/state modes are judged at their best stable
  concurrency, not only at equal concurrency.
- Kernel microbenchmarks remain required for diagnosis, but every production
  kernel decision has an end-to-end result.

## Constraints

- Current local Windows development has no usable CUDA server. Schema,
  instrumentation, and CPU aggregation tests can proceed; GPU values and
  performance decisions remain pending.
- The production target is one RTX 4090D 24 GB (Ada SM89).

## Open questions

- Which default TTFT/TPOT SLO profiles should be published for short chat and
  long generation?
- What GPU telemetry source is available in the restored server environment?
- How long must each offered-load point run to make p99 stable enough for a
  default-setting decision?

## Change log

- 2026-07-22: Created the goodput-first benchmark contract and made open-loop
  SLO capacity, scheduler occupancy, speculative acceptance, and GPU telemetry
  required evidence for the optimization roadmap.
- 2026-07-22: Fixed online SLO timing at planned arrival, defined separate
  service latency, and standardized schema v3 engine/GPU telemetry provenance.
- 2026-07-23: Required single-source timing/SLO facts and separated offline
  execution, aggregation, result construction, and presentation boundaries.
