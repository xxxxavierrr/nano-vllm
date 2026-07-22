# Design implications

## Adopted architecture boundaries

Quantization uses a small capability object per resolved checkpoint method and
a per-layer runtime method with create/post-load/apply lifecycle hooks. Loader
format, runtime kernel, and cache dtype remain separate. nano-vLLM will not copy
the full vLLM plugin matrix.

Speculative decoding keeps one scheduler path. Drafts live on request state and
are normal scheduled verification tokens. Model-specific proposal and
sampling/acceptance are separate components; cache and recurrent-state commit
is a protocol step, not an incidental effect of forward execution.

## Compatibility choices

- Preserve existing nano-vLLM GPTQ raw-`g_idx` behavior and strict fused-shard
  comparison. Optimized layouts are post-load transformations with reference
  parity requirements.
- Preserve the current greedy-only MTP restriction. It is a declared acceptance
  policy and carries no probabilistic losslessness claim.
- Preserve one `Scheduler` and one `ModelRunner` execution path. Future
  proposers plug into the path instead of creating worker pairs.
- Treat MRv1/MRv2 class names as upstream implementation detail; follow stable
  V1 data ownership and order.

## Risks and mitigations

- A generic abstraction can outgrow this small engine. Add only the lifecycle
  hooks needed by GPTQ/FP8 and one proposer interface; do not import vLLM's
  registry breadth.
- Token rollback without state rollback can silently corrupt hybrid models.
  Require tests for every accepted length plus preemption/abort and keep state
  commit centralized.
- Benchmark conclusions can confuse acceptance with speed. Require proposal,
  verification, replay, memory, concurrency, and end-to-end metrics together.
