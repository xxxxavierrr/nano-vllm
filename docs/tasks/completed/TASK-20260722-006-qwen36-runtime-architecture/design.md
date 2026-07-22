# Design

## Minimal target components

```text
                 +-------------------+
SchedulerBatch ->| BatchPlanner      |-> PreparedBatch/ExecutionSignature
                 +-------------------+
                            |
                 +----------v---------+
                 | ModelRunner        | rank-local orchestration facade
                 +----------+---------+
                            |
       +--------------------+--------------------+
       |                    |                    |
StateTransaction      GraphManager          Speculator
KV + GDN state        mode/key/replay        proposal/acceptance
       |                    |                    |
       +--------------------+--------------------+
                            |
                   target/model layers
              quant methods + opaque custom ops
```

Supporting initialization data are `ResourceSpec`, backend capabilities, and a
single capacity/runtime plan.

## Why these boundaries are justified

- `BatchPlanner` owns transformation and reusable pinned/stable buffers.
- Cache/state managers own physical lifetime and mutation.
- `Speculator` owns a replaceable algorithm and model lifecycle.
- `GraphManager` owns policy, persistent buffers/pools, and capture lifecycle.
- quant methods own parameter storage/loading/application lifecycle.
- `ModelRunner` retains cross-component ordering, so no executor abstraction is
  created merely to forward calls.

## Compatibility strategy

Each stage initially adapts the current tensors and algorithms behind the new
contract. Numerical kernels, scheduler token decisions, ZMQ/API messages, and
user configuration remain unchanged. Old branches are removed only after
eager/Graph and state parity gates pass.

## Priority

1. GPU-baseline the active GDN refactor.
2. Typed batch/context and move metadata out of the model.
3. Cache/GDN state ownership plus speculative transaction and capacity plan.
4. Extract MTP proposer/greedy acceptance.
5. Replace graph dispatch/capture with semantic keys and capabilities.
6. Extract quantization methods and unify target/draft loading.
7. Tune kernels and add new proposers/formats only afterward.

## Main risks

- A flag-day runner split would combine metadata, state, Graph, and numerical
  changes and make regressions unlocalizable.
- Moving MTP before central state ownership would preserve the current hidden
  coupling in a different file.
- Graph refactoring before semantic batch metadata would invent another
  temporary descriptor.
- Quantization abstractions can become a registry framework larger than this
  engine; keep only used lifecycle hooks.
