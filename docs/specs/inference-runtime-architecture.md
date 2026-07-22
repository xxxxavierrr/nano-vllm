---
subject: inference-runtime-architecture
title: Qwen3.6 inference runtime architecture
status: active
created: 2026-07-22
updated: 2026-07-23
owner: Codex
---

# Qwen3.6 inference runtime architecture

## Motivation

Qwen3.6 combines full attention, stateful Gated DeltaNet, packed weight and KV
quantization, MTP speculative decoding, and Full/Piecewise CUDA Graphs. These
capabilities currently converge inside a roughly 1,500-line `ModelRunner` that
owns construction, loading, physical memory, metadata, state transactions,
proposal, capture, execution, and metrics.

The engine should remain small and single-process per replica. Extensibility
comes from separating real lifecycle and mutation boundaries inside the runner,
not from introducing an EngineCore/LocalExecutor process split prematurely.

## Architectural principles

- Keep one scheduler and one model execution path for prefill, mixed, ordinary
  decode, and speculative verification.
- Model batches by scheduled tokens and semantic metadata, not mutually
  exclusive request phases.
- Build scheduler-derived metadata outside model layers.
- Resolve capabilities, layouts, and resource requirements at initialization;
  do not select backends or reinterpret formats in the hot path.
- Centralize persistent GPU-state allocation and speculative commit/rollback.
- Make custom ops the boundary around opaque numerical work or mutation, not a
  wrapper for every kernel or an entire model layer.
- Keep checkpoint format, runtime kernel, graph capability, and cache dtype
  separate.
- Add an abstraction only when it owns policy, resources, mutation, or an
  independently testable lifecycle.
- `ModelRunner.run` is an orchestration facade, not the implementation owner of
  speculative verification, acceptance, state commit, proposal bookkeeping,
  or metrics aggregation. Those phases expose focused interfaces and tests.
- Batch preparation delegates attention, GDN, sampling, and graph-buffer
  construction to focused builders; no single preparation method reconstructs
  every metadata family.

## Target runtime flow

```text
SchedulerBatch
  -> BatchPlanner.prepare()
       -> PreparedBatch
          - token/position tensors
          - AttentionMetadata
          - GDNMetadata
          - SamplingMetadata
          - ExecutionSignature
  -> StateManager.begin_step()
       -> StateTransaction / cache and recurrent-state views
  -> GraphManager.dispatch(ExecutionSignature, capabilities)
       -> FULL / PIECEWISE / EAGER execution descriptor
  -> ModelRunner executes target model
  -> Speculator verifies/accepts/proposes when enabled
  -> StateTransaction.commit(accepted prefixes)
  -> Scheduler postprocess + StepMetrics
```

`ModelRunner` remains the facade and owns this order. It delegates preparation,
resources, state, graph policy, and proposal to components that have real
ownership; it does not become an empty pass-through layer.

## Core data contracts

### PreparedBatch

One typed batch object replaces the current broad mutable `Context` populated
through repeated `set_context` calls. It contains:

- real scheduled token and request counts;
- input IDs, positions, logits indices, and sampling metadata;
- typed attention metadata and optional typed GDN metadata;
- cache/state slot references and real/padded dimensions;
- speculative query lengths and accepted-prefix inputs when relevant;
- an immutable `ExecutionSignature` for graph dispatch.

A scoped forward-context manager may expose this object to custom ops during a
model call. The implementation may retain a process-local forward context, as
vLLM does, but callers use `with forward_context(prepared_batch, execution)` so
reset is exception-safe and optional fields are grouped by subsystem.

### ExecutionSignature

The signature contains semantic graph dimensions:

- real tokens and requests;
- uniform query length or `None`;
- model/backend capability key;
- feature dimensions that actually change captured structure.

It must not contain a Boolean `is_uniform_decode`. Ordinary decode is uniform
query length one; fixed-`k` MTP verification is uniform query length `1+k`.

### ResourceSpec and RuntimePlan

Initialization components publish resource/capability facts instead of letting
`allocate_kv_cache` rediscover module types:

- per-layer weight method and persistent bytes;
- per-layer KV payload/scale layout and bytes per block;
- recurrent and convolution state shape/copies per request;
- draft-model cache/state requirements;
- kernel workspace and warmup shapes;
- Graph support, stable buffers, and estimated persistent pool bytes.

A small `RuntimePlan` validates compatibility and feeds one capacity planner.
This is a data contract, not a generic plugin framework.

## Component ownership

### ModelRunner

Retain responsibility for rank-local initialization order and step
orchestration. Remove direct ownership of:

- GDN slot dictionaries and slab copy algorithms;
- MTP-specific batch construction and recursive proposal loops;
- Full/Piecewise key generation and stable Graph buffers;
- cache tensor discovery by walking model modules;
- quantization-format branches and post-load conversion;
- broad `last_*` dictionaries used as an implicit metrics channel.

TP transport and rank coordination may remain here until Qwen3.6 TP is actually
implemented. A separate LocalExecutor is not required for the current TP=1
runtime.

### BatchPlanner

Own CPU-side translation from `SchedulerBatch`/request state to a
`PreparedBatch`. Internally use focused metadata builders:

- attention/KV metadata builder;
- GDN partition/chunk/state metadata builder;
- sampling/logits metadata builder;
- proposer-specific input builder supplied by the active `Speculator`.

Metadata is constructed once per step, not once per layer. Builders may reuse
pinned host buffers and stable graph buffers. Model layers never reconstruct
request slices, chunk partitions, or state-slot mappings.

### CacheManager and HybridStateManager

Logical KV block allocation remains scheduler-side. A rank-local
`CacheManager` owns physical KV tensors, scale tensors, layer bindings, cache
specs, byte accounting, and lifecycle.

`HybridStateManager` owns GDN convolution/recurrent slabs, slot allocation,
release/reset, committed and working views, and state byte accounting. The
Qwen model supplies declarative state shapes; it does not allocate request
state or expose state-copy orchestration.

Both participate in a `StateTransaction` for speculative execution:

- target verification writes to working recurrent state;
- accepted prefixes are committed centrally;
- rejected suffix KV entries become invalid through the committed frontier and
  are overwritten later;
- recurrent state is committed, restored, or replayed according to one
  documented protocol;
- abort, preemption, and slot reuse use the same release/reset API.

This transaction is the correctness boundary shared by MTP, future proposers,
GDN, and KV cache.

### Speculator

The scheduler remains proposer-agnostic: it budgets verification tokens and
stores draft IDs on request state. Extract runner-specific MTP code behind:

```text
Speculator.prepare_inputs(...)
Speculator.propose(target_output, accepted_prefixes, state_views)
AcceptancePolicy.accept(target_logits, draft)
```

The initial `MTPProposer` retains greedy-only semantics. `GreedyAcceptance`
owns exact prefix acceptance. A future DSpark/tree proposer implements the same
proposal contract without adding branches to `Scheduler` or `ModelRunner`.

The target model and proposer each declare:

- weight quantization method;
- cache/state requirements;
- graph support and capture signatures;
- warmup shapes;
- emitted proposal/acceptance metrics.

MTP must not implicitly remain BF16 because its current constructor passes
`quant_config=None`. Draft precision is an explicit policy and is reported,
even when BF16 is intentionally chosen.

### GraphManager

Own:

- capability resolution across full attention, FP8 paged attention, GDN,
  quantized custom ops, target model, and proposer;
- finite Full/Piecewise candidate keys;
- compile, autotune, eager warmup, capture, and readiness;
- stable input/metadata buffers and replay copies;
- graph-memory estimates/reservation and actual memory reporting;
- per-key capture/replay/fallback metrics.

It consumes `ExecutionSignature` and `PreparedBatch`; it does not rebuild model
metadata. Full keys include request count and uniform query length so MTP
`1+k` can use Full Graph. GDN's current one-request Full restriction is a
declared capability limit rather than an ad hoc capture-size calculation.

### Quantization methods and checkpoint loader

Replace hard-coded GPTQ/FP8 branches in `Config`, `LinearBase`, model
constructors, and global post-load traversal with a lightweight resolved plan:

```text
checkpoint metadata + explicit policy
  -> QuantizationPlan
  -> method for (module type, module prefix)
  -> create parameters
  -> load logical/fused shards
  -> finalize/validate/repack
  -> apply
```

A method owns `create_parameters`, `load/finalize`, `apply`, graph capability,
workspace, and supported sharding. The first map may contain only BF16, FP8,
and GPTQ; a full plugin registry is unnecessary.

Unify target and MTP checkpoint loading behind one `CheckpointLoader` with
namespace/prefix and fused-shard mapping. Preserve strict fused GPTQ `g_idx`
comparison. Keep weight quantization independent from a per-layer KV-cache
specification.

### CapacityPlanner and metrics

One capacity calculation consumes all `ResourceSpec`s, configured utilization,
and measured/estimated compiler/Graph memory. It jointly selects KV blocks and
GDN request-state capacity rather than letting `allocate_kv_cache` discover and
allocate every feature.

Replace `last_execution_mode` and `last_speculative_stats` side channels with a
typed `StepMetrics` returned from execution. Offline and online reporters
aggregate the same fields: scheduler tokens, accepted/proposed tokens,
execution key/mode, real/padded tokens, fallback reason, cache/state capacity,
and memory.

## Qwen3.6 model and Gated DeltaNet boundary

`qwen3_5.py` should contain model architecture, fused projection layout,
declarative state/cache specs, and checkpoint mapping. It should not build
scheduler metadata or implement transient request orchestration.

The GDN layer remains:

```text
fused QKV+Z / B+A projection
  -> one opaque stateful GDN core custom op
       -> packed convolution
       -> private recurrent or chunk numerical kernels
  -> gated norm and output projection
```

Keep the one formal packed API and the three private numerical responsibilities
already identified. Do not merge recurrent scan with chunk construction merely
to reduce kernel count. Remove `_build_packed_metadata`, request Python loops,
and public fallback paths once the runner-style metadata builder covers warmup
and tests.

## Parallelism extension

- Data parallelism remains multiple independent EngineProc replicas routed by
  the existing serving client. It does not enter model state or the scheduler.
- Tensor parallelism is added through layer sharding/collective capability,
  per-rank resource specs, and quantization-method support. Only then is a
  rank executor boundary justified.
- The batch, speculator, state, graph, and metrics contracts must not assume
  TP=1, but the implementation continues to reject unsupported Qwen3.6/GPTQ
  TP combinations explicitly.

## Staged refactoring order

### Stage 0: establish the GPU baseline

Finish the active GDN refactor and run its existing correctness, real loader,
state, Graph, accuracy, and performance gates. Do not rearrange working kernel
code before this baseline exists.

### Stage 1: typed batch and scoped context

Introduce `PreparedBatch`, typed attention/GDN metadata, semantic
`ExecutionSignature`, and a scoped forward context. Move all GDN metadata
construction out of the model. Preserve numerical calls byte-for-byte where
possible.

### Stage 2: physical cache/state ownership

Extract `CacheManager`, `HybridStateManager`, and a speculative
`StateTransaction`. Move allocation, slots, working-state copy/commit, replay,
abort, and preemption integration out of `ModelRunner`. Introduce resource
specs and one capacity calculation.

### Stage 3: proposer and acceptance extraction

Move `_run_mtp_proposal`, proposer metadata, recursive `k` steps, acceptance,
and proposal metrics into `MTPProposer` and `GreedyAcceptance`. Keep the current
scheduler protocol and token results unchanged. Make draft precision explicit.

### Stage 4: capability-aware GraphManager

Use the new execution signature and resource plan to fix `1+k` Full keys,
backend capability resolution, explicit Piecewise capture evidence, stable
buffers, Full memory reservation, and per-key metrics.

### Stage 5: quantization lifecycle and unified loading

Extract BF16/FP8/GPTQ methods and unify target/MTP loading. This is intentionally
after state/graph boundaries because current W4A16 is functional and should not
be destabilized while those correctness contracts are moving.

### Stage 6: performance specialization

Only after parity gates pass, tune GDN crossover/kernels, W4 layout/autotune,
FP8 attention, MTP `k`, graph buckets, and concurrency. New DSpark/tree
proposers or quantization formats start here as implementations of the new
contracts, not new runner branches.

Each stage is a separate task/commit and retains a runnable online/offline
engine.

## What should not be refactored now

- Do not split EngineCore and LocalExecutor solely for code cleanliness.
- Do not add a generic operator/plugin registry beyond the three present
  quantization methods and actual backend choices.
- Do not split prefill and decode into separate runners or schedulers.
- Do not expose one public wrapper per Triton kernel.
- Do not move DP into `ModelRunner`; the existing whole-request replica routing
  is the right boundary.
- Do not combine GDN recurrent and chunk algorithms into one kernel without
  numerical/performance evidence.
- Do not rewrite working W4A16/FP8 kernels during ownership refactors.

## Acceptance criteria

- `ModelRunner` is an orchestration facade rather than the owner of all feature
  implementations, while the process topology and public API stay unchanged.
- Model layers receive typed prebuilt metadata and contain no request-level
  orchestration.
- Target and speculative state mutations have one commit/rollback protocol.
- New quantization methods, proposers, and graph-capable stateful layers can be
  added through their declared lifecycle/capability contracts.
- Full Graph supports compatible uniform `1+k` verification; Piecewise capture
  and memory are observable.
- Online and offline correctness/benchmark results remain comparable across
  every stage.
- Stage 1 preserves the existing model-call surface while replacing manually
  assembled optional context fields with typed batch metadata and an
  exception-safe scoped context, so later stages can migrate independently.

## Constraints

- The local machine cannot validate CUDA kernels, persistent state mutation,
  or CUDA Graph replay.
- Existing uncommitted GDN work is user work and must be preserved.
- GPU validation gates precede performance claims and broad file movement.

## Non-goals

- Recreating vLLM's complete registry, platform, LoRA, distributed worker, or
  compilation-pass infrastructure.
- Adding DP/TP, a new quantization format, or another proposer as part of the
  ownership refactor itself.
- Performing a large flag-day rewrite.

## Open questions

- Whether Inductor CUDAGraph Trees can expose reliable captured-key/replay
  evidence or should be replaced by explicit Piecewise wrappers.
- Whether future probabilistic speculative decoding requires a sampler-owned
  acceptance interface beyond the initial greedy policy.
- The measured Graph/workspace reserve for Qwen3.6-27B W4A16 + FP8 KV on the
  24 GiB target GPU.

## Change log

- 2026-07-22: Created for the cross-capability Qwen3.6 runtime refactoring
  assessment.
- 2026-07-22: Defined target data contracts, state/resource ownership,
  component boundaries, non-goals, and a six-stage refactoring sequence.
- 2026-07-22: Authorized Stage 1 implementation of typed prepared-batch
  metadata, semantic execution signatures, scoped context, and runner-owned
  GDN metadata construction.
- 2026-07-22: Authorized Stage 2 extraction of hybrid GPU state ownership and
  speculative commit/rollback from `ModelRunner`.
- 2026-07-22: Authorized Stage 3 extraction of MTP proposal and greedy
  acceptance behind replaceable speculative interfaces.
- 2026-07-22: Authorized Stage 4 semantic Full Graph keys for ordinary decode
  and fixed-k MTP verification, followed by Graph lifecycle extraction.
- 2026-07-22: Authorized typed rank-local step results to replace mutable
  `last_execution_mode` and speculative-stat side channels.
- 2026-07-23: Made runner orchestration and batch-builder delegation explicit
  structural requirements after later optimizations expanded the facade.
