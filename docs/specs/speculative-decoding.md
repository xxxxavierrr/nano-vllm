---
subject: speculative-decoding
title: Speculative decoding and Qwen MTP
status: active
created: 2026-07-22
updated: 2026-07-22
owner: Codex
---

# Speculative decoding and Qwen MTP

## Motivation

nano-vLLM needs speculative decoding as a scheduler/model-runner protocol, not
an isolated MTP forward pass: proposals, verification, acceptance, cache/state
commit or rollback, scheduling budgets, graphs, metrics, and serving behavior
must agree.

## Requirements

- Treat proposal, target verification, rejection sampling, state commit, and
  next proposal as distinct phases with explicit ownership.
- Store draft tokens on request state. The unified scheduler schedules ordinary
  and speculative tokens through the same per-request computed-token deficit;
  it does not create a separate decode worker or speculative scheduler.
- Scheduler token budgets and cache reservations include draft verification
  tokens and never expose uncommitted tokens as request output.
- Draft methods implement a common proposer contract over target hidden state,
  accepted/rejected counts, request state, and stable graph buffers.
  MTP-specific model logic does not leak into scheduler correctness.
- Target logits and draft information flow through a distinct acceptance
  component. Greedy prefix matching and probabilistic rejection sampling are
  different declared policies, not branches hidden inside a proposer.
- Probabilistic rejection sampling is lossless with respect to the target
  distribution: a proposed token `x` is accepted with
  `min(1, p(x) / q(x))`; at the first rejection the recovery token is sampled
  from normalized `max(p - q, 0)`; after full acceptance the bonus token is
  sampled from the target distribution.
- Target and draft probabilities use the same declared logits transforms.
  Randomness is per request and reproducible; a fused/blockwise implementation
  must avoid materializing redundant full-vocabulary softmax buffers.
- Acceptance supports an arbitrary valid prefix length from zero through `k`.
- Rejected verification work reduces the request's committed/computed position
  before the next schedule; rejected tokens never become prefix-cache entries or
  observable output.
- KV cache and recurrent/conv state remain correct after full acceptance,
  partial rejection, preemption, abort, and Graph replay.
- The next proposal is generated only after accepted output and model state have
  been post-processed. Any reuse of the target hidden states must preserve that
  ordering.
- Speculative decoding is optional and preserves non-speculative behavior.
- Report proposed/accepted/rejected counts, per-position acceptance, effective
  accepted tokens per target step, latency, throughput, and memory overhead.
- Report accepted tokens/s together with output-token and request goodput,
  time-weighted running requests, scheduled tokens per step, and TTFT/TPOT p50
  and p99. Acceptance metrics diagnose speculation but never select a winner
  without end-to-end SLO goodput.
- DSpark/DFlash draft calibration is a sequential offline workflow on the
  24 GB target: persist target-produced hidden-state/token inputs first, unload
  the target, then load the BF16 draft alone for reference and GPTQ
  calibration. The online proposer consumes only the resulting packed INT4
  draft.
- Calibration caches are resumable, sharded, self-describing, and preserve
  flattened token data, sequence boundaries, positions, target hidden states,
  and source/checkpoint provenance without embedding the target model itself.
- Draft checkpoint conversion owns strict name mapping, Markov/confidence-head
  preservation, per-linear calibration, and output compatibility with the
  production GPTQ loader. Unknown or incomplete weights fail conversion.

## Scope

- Qwen3.6 native MTP with bounded `k`, greedy and probabilistic verification,
  scheduler and
  ModelRunner integration, GDN state handling, CUDA Graphs, and benchmarks.
- Current vLLM V1 speculative-decoding conventions and rejection of obsolete
  V0 worker architecture.
- Offline DSpark/DFlash calibration model, cache, weight mapping, and packed
  INT4 checkpoint generation tooling. Runtime DSpark tree proposal remains a
  later GPU-integrated milestone.

## Non-goals

- Reintroducing vLLM V0 `SpecDecodeWorker`/multi-step-worker architecture.
- Top-k/top-p or other transforms until the same transform is implemented and
  validated identically for target and draft distributions.
- Enabling an unvalidated DSpark tree proposer in the online engine.

## Acceptance criteria

- One V1-style request/scheduler/runner protocol covers non-spec and spec work.
- State and cache correctness is tested for all acceptance lengths and lifecycle
  events.
- Baseline versus each `k` reports acceptance, speed, memory, and concurrency.
- The selected proposer and `k` maximize output-token/request goodput under the
  declared latency SLO; maximum raw acceptance length is not the objective.
- V1 source conventions and intentional nano-vLLM differences are documented.

## Runtime protocol and ownership

```text
request carries committed tokens + pending draft tokens
        -> scheduler budgets target/draft verification together
        -> target runner verifies scheduled draft positions
        -> acceptance policy returns committed prefix + bonus/recovery token
        -> scheduler/runner subtract rejected work and commit accepted state
        -> proposer uses post-processed state/hidden states for next drafts
        -> request stores next drafts for a later scheduler step
```

The scheduler owns request token accounting, cache reservation, preemption, and
which drafts are scheduled. The target runner owns target forward/logits. The
acceptance policy owns token selection and accepted/rejected counts. A proposer
owns draft-model-specific execution. Cache/state owners implement commit,
discard, or replay. Metrics aggregate the protocol outputs and do not drive its
correctness.

## Current vLLM V1 source conventions

The source study is pinned to vLLM commit
[`6e96891ba00d3d61a1eaa9c95bdd8d2663b183bd`](https://github.com/vllm-project/vllm/commit/6e96891ba00d3d61a1eaa9c95bdd8d2663b183bd)
(2026-07-21). That revision contains both Model Runner V1 and Model Runner V2;
the stable protocol below is shared and is more important than either runner's
temporary class layout.

- The [V1 scheduler](https://github.com/vllm-project/vllm/blob/6e96891ba00d3d61a1eaa9c95bdd8d2663b183bd/vllm/v1/core/sched/scheduler.py)
  explicitly has no prefill/decode phase split. It schedules the gap between
  `num_computed_tokens` and prompt/output/draft tokens, emits scheduled draft
  IDs, truncates them to the token budget, and clears stale drafts.
- The same scheduler optimistically advances computed tokens, then subtracts
  rejected draft positions while processing output. Preemption clears draft
  IDs and resets computed state; rejected work is therefore not committed by
  merely having run a forward pass.
- [`BaseSpeculator`](https://github.com/vllm-project/vllm/blob/6e96891ba00d3d61a1eaa9c95bdd8d2663b183bd/vllm/v1/worker/gpu/spec_decode/speculator.py)
  is the current MRv2 proposer boundary. Its `propose` input includes target
  hidden states, sampled/rejected counts, request buffers, sampling state, and
  graph lifecycle hooks.
- [`MTPSpeculator`](https://github.com/vllm-project/vllm/blob/6e96891ba00d3d61a1eaa9c95bdd8d2663b183bd/vllm/v1/worker/gpu/spec_decode/mtp/speculator.py)
  is only one implementation of that boundary; it does not define scheduler or
  acceptance semantics.
- The [MRv2 rejection sampler](https://github.com/vllm-project/vllm/blob/6e96891ba00d3d61a1eaa9c95bdd8d2663b183bd/vllm/v1/worker/gpu/spec_decode/rejection_sampler.py)
  consumes target logits and optional draft logits, then returns sampled,
  accepted, and rejected counts. This separation is required for probabilistic
  losslessness; greedy-only implementations must state their narrower contract.
- The [MRv2 model runner](https://github.com/vllm-project/vllm/blob/6e96891ba00d3d61a1eaa9c95bdd8d2663b183bd/vllm/v1/worker/gpu/model_runner.py)
  performs target sampling/verification, post-processes sampled/rejected state,
  and only then invokes the proposer for the next draft. Proposal may overlap
  host output copy but not state-commit ordering.
- [`SpecDecodingStats`](https://github.com/vllm-project/vllm/blob/6e96891ba00d3d61a1eaa9c95bdd8d2663b183bd/vllm/v1/spec_decode/metrics.py)
  records draft count, drafted/accepted tokens, and per-position acceptance.
  Its conventional mean acceptance length includes the bonus token:
  `1 + accepted_draft_tokens / draft_rounds`.

The current user-facing V1 configuration supports multiple proposal methods,
dynamic speculative lengths, separate draft/target parallel settings, and
multiple rejection policies. nano-vLLM does not need all of them now, but its
core request/scheduler protocol must not hardcode MTP so deeply that another
proposer requires a second engine path.

## Obsolete paths that are not design references

Do not copy the V0 `SpecDecodeWorker`, `MultiStepWorker`, draft-worker/target-
worker orchestration, or legacy speculative-decode documentation describing
those workers. Their worker topology conflicts with V1's unified request state,
scheduler token budget, and integrated runner/speculator flow. MRv1-only
helpers in `vllm/v1/worker/gpu_model_runner.py` may still exist in the pinned
tree for compatibility, but new nano-vLLM interfaces should follow the shared
V1 semantics and the narrower MRv2 proposer/sampler boundaries above.

## nano-vLLM alignment and required follow-ups

Already aligned:

- Draft IDs live on `Sequence`; scheduler reservation includes verification
  tokens and validates that accepted outputs match the proposed prefix.
- Greedy acceptance returns every valid prefix length from zero through `k`.
- Qwen hybrid state uses working/committed slabs and replays a rejected prefix
  before the next proposal.
- Metrics include drafted/proposed/accepted/rejected counts and per-position
  acceptance; benchmark sweeps vary `k` and concurrency.

Follow-up architecture requirements:

- Extract a small proposer interface from `ModelRunner`; MTP model loading,
  proposal execution, and graph warmup should implement it rather than remain
  global conditionals.
- Keep `greedy_accept` as an explicit acceptance policy. Sampling with
  `temperature > 0` uses a separately tested lossless rejection sampler and
  requires the proposer to return draft logits, not only argmax token IDs.
- Represent verification input/output with typed metadata rather than parallel
  lists of accepted counts, outputs, and next drafts.
- Centralize post-verification commit/rollback for target KV, MTP KV, DeltaNet
  recurrent state, and conv state. The scheduler's token count alone cannot
  prove state correctness for a hybrid model.
- Measure proposal time, verification time, rejection replay time, accepted
  throughput, mean acceptance length including bonus, per-position conditional
  and unconditional acceptance, memory, and concurrency. Acceptance rate alone
  cannot select the best `k`.

## Constraints

Qwen hybrid GDN state is mutable and requires explicit working/commit semantics
when target verification rejects part of a draft chain.

On the RTX 4090D 24 GB target, the 8.8 GB BF16 DSpark draft is an offline-only
calibration/reference artifact. Online DSpark validation requires a packed
INT4 draft; published BF16 acceptance cannot be reported as locally measured.

## Open questions

- Can target verification and next MTP proposal share hidden states safely?
- Whether future proposer families require a branch representation beyond the
  current indexed conv/recurrent prefix slots.

## Change log

- 2026-07-22: Created from the existing Qwen MTP implementation and a planned
  study of current vLLM V1 speculative decoding.
- 2026-07-22: Made SLO-goodput the speculative optimization target and retained
  accepted tokens/s, per-position acceptance, and scheduled-token occupancy as
  diagnostic metrics.
- 2026-07-22: Removed the infeasible online BF16 DSpark baseline from the 24 GB
  experiment matrix; required sequential offline BF16/INT4 agreement and a
  paired-target-plus-INT4-draft runnable baseline.
- 2026-07-22: Added the pinned V1 scheduler/MRv2 proposer and sampler protocol,
  state ordering, metrics definitions, local gaps, and explicit V0 exclusions.
- 2026-07-22: Added lossless probability-ratio acceptance, residual
  `(p-q)+` recovery sampling, shared logits transforms, and per-request RNG as
  required probabilistic MTP behavior.
- 2026-07-22: Added DSpark/DFlash offline cache, strict checkpoint mapping, and
  in-repository GPTQ draft conversion to scope; online tree proposal remains
  gated on real checkpoint/GPU validation.
- 2026-07-23: Recorded Qwen3.6 MTP k=2 GPU integration, indexed branch commits,
  and zero rejected-prefix replay; multi-k/DSpark acceptance and offered-load
  goodput remain pending.
