# Project architecture

## Serving path

```text
OpenAI-compatible FastAPI
-> EngineClient
-> local ZMQ
-> EngineProc
-> LLMEngine
-> Scheduler
-> ModelRunner
```

The offline path invokes `LLMEngine` directly. Online and offline benchmarks
must remain framework-level clients so model, kernel, graph, quantization, and
scheduler changes can be compared without rewriting the workload.

## Qwen3.5/3.6 hybrid model

The model alternates full-attention and Gated DeltaNet layers. Full attention
uses paged KV cache; DeltaNet uses causal-convolution and recurrent state slabs
owned by `ModelRunner` and indexed by request state slots.

The durable capability contract is the
[Gated DeltaNet spec](../specs/gated-deltanet.md). Individual refactors,
research, validation, and benchmarks are separate task records linked to it.

## Rank-local runtime boundary

The target rank-local path keeps `ModelRunner` as the orchestration facade:

```text
SchedulerBatch -> BatchPlanner -> PreparedBatch
  -> StateTransaction -> Graph dispatch -> target model
  -> optional Speculator -> state commit -> StepMetrics
```

The minimum durable internal owners are:

- `BatchPlanner`: typed attention, GDN, sampling, speculative, and execution
  metadata;
- `CacheManager`: physical KV payload/scales and per-layer cache specs;
- `HybridStateManager`: GDN slabs, state slots, working/committed state;
- `Speculator`: proposal model/input construction and acceptance policy;
- `GraphManager`: capabilities, finite keys, stable buffers, capture/replay,
  memory, and fallback evidence;
- layer quantization methods: parameter creation, loading/finalization, apply,
  and kernel capability;
- `CapacityPlanner`: one plan across cache, state, draft, workspaces, and Graph.

The [runtime architecture spec](../specs/inference-runtime-architecture.md)
owns the detailed contracts and staged migration.

`ModelRunner` continues to own initialization and step order. This internal
decomposition does not require a new EngineCore/LocalExecutor process boundary.

## Validated internal contracts

- `PreparedBatch` and its typed attention/GDN/speculative metadata are the
  only model-step input contract; model layers do not reconstruct scheduler
  metadata.
- Forward context is scoped and exception-safe. Nested target/draft execution
  must restore the prior context.
- `HybridStateManager` owns state slots and committed/branch slabs;
  speculative acceptance selects slots rather than copying or replaying the
  target model.
- Runner results carry immutable typed execution/speculative metrics. Engines
  and benchmarks consume the returned envelope rather than mutable
  `last_*` side channels.
- `MTPProposer` owns draft input/cache construction and recursive proposal;
  acceptance and state commit remain separate policies/phases.

These contracts were exercised together by the RTX 4090D unfiltered suite and
Qwen3.6 GPTQ/MTP smoke before their extraction tasks were archived.

## Parallelism boundary

Data parallelism remains whole-request routing across independent EngineProc
replicas in the serving layer. Tensor parallelism belongs to layer sharding,
collectives, and per-rank resource/capability plans. Neither concern should
change the scheduler's semantic batch protocol.
