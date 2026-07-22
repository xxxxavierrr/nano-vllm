---
subject: gated-deltanet
title: Qwen3.5/3.6 Gated DeltaNet
status: active
created: 2026-07-22
updated: 2026-07-22
owner: Codex
---

# Qwen3.5/3.6 Gated DeltaNet

## Motivation

nano-vLLM needs a small, correct, graph-compatible Gated DeltaNet path for the
hybrid Qwen3.5/3.6 model. Its boundaries should follow proven upstream vLLM
semantics while avoiding infrastructure that nano-vLLM does not need.

## Requirements

- `forward` is projection -> one opaque stateful GDN core custom op -> gated
  normalization and output projection.
- Remove the whole-layer `torch.compiler.disable` boundary.
- Fuse QKV+Z and B+A input projections while preserving BF16/FP8/GPTQ loading.
- Production, fallback, tests, and benchmarks use one formal packed DeltaNet
  API with scheduler-style metadata.
- Do not expose recurrent-only or chunk-only execution helpers.
- Retain only numerical kernels whose parallel responsibilities are genuinely
  different; do not multiply public wrappers to mirror kernels.
- Preserve convolution/recurrent state lifetime, MTP working state, padding,
  W4A16 compatibility, and CUDA Graph behavior.
- Speculative execution retains the state after every verified prefix in
  distinct candidate state slots. Acceptance commits by remapping the request
  to the selected prefix slot; rejection discards unselected suffix slots.
  Production execution must not reconstruct accepted recurrent/conv state by
  replaying the target model or by copying and then restoring a whole state
  slab.
- Source-level decisions must distinguish essential vLLM semantics from
  vLLM-specific registry, platform, and distributed infrastructure.
- Scheduler/runner code must build all request partition, sequence-offset,
  state-slot, chunk, speculative, actual-token, and graph-padding metadata.
  The model layer consumes this metadata and must not rebuild it.
- One public GDN core may privately dispatch to recurrent decode and chunked
  prefill kernels. Mixed batches use the same outer core and precomputed
  partition metadata; API unity does not require one numerical kernel.
- Production GDN execution must not loop over requests in Python. Packed
  convolution and DeltaNet kernels consume tensor metadata.
- Prefill kernel compilation/autotuning must be warmed before final cache
  capacity is calculated, independently of CUDA Graph mode.
- State-capacity planning includes speculative branch slots and exposes their
  memory cost. If branch capacity is unavailable, scheduling reduces
  speculation or concurrency explicitly; state slots must never alias.
- `delta_state_dtype` is independent of model weights and KV dtype. FP8 E4M3
  uses separate FP16 dynamic scales: convolution state is scaled per
  layer/slot/channel across kernel taps, and recurrent state per
  layer/slot/head/K-row across V. Committed and speculative branch slots share
  exactly the same payload/scale layout and lifecycle.
- FP8 state kernels dequantize on state load and requantize on state write in
  the recurrent/conv operator; production must not construct a full native
  state slab. Until SM89 numerical and Graph validation passes, the runtime
  fails explicit FP8-state startup rather than silently using native state.

## Scope

- Qwen3.5/3.6 GDN layer organization and projections.
- Packed recurrent/chunk execution and sequence metadata.
- Convolution and recurrent-state ownership.
- Compiler/custom-op boundaries and loader integration.
- Correctness, graph, accuracy, and performance validation.

## Non-goals

- Copying vLLM implementation verbatim.
- Adding TP support beyond the current Qwen3.5/3.6 restriction.
- Changing the recurrent/chunk crossover without benchmark evidence.

## Acceptance criteria

- One production packed API and one stateful layer-core custom-op boundary.
- Fused BF16/FP8/GPTQ shards load at correct offsets; fused GPTQ `g_idx` values
  are verified elementwise.
- Recurrent, chunk, mixed, continuation, state, greedy-output, CUDA Graph, and
  benchmark validation is recorded by the executing tasks.
- Upstream source findings and intentional nano-vLLM differences are captured
  in this spec and durable knowledge.
- Production, warmup, and tests obtain GDN metadata through the runner-side
  builder contract; the model layer has no request-derived fallback builder.
- FP8 recurrent/conv state remains opt-in until its lower state bytes increase
  maximum stable scheduler batch and SLO goodput after conversion cost and
  numerical error are included.

## Constraints

- Local CPU validation cannot establish Triton numerical correctness.
- GPU correctness and performance evidence must be produced on the server.

## Open questions

- Which current upstream GDN helper/custom-op boundaries should nano-vLLM
  preserve, and which are framework-specific?
- What recurrent/chunk crossover performs best on the target GPU and workload?

## Upstream vLLM source study

Inspected upstream revision
[`6e96891ba00d3d61a1eaa9c95bdd8d2663b183bd`](https://github.com/vllm-project/vllm/commit/6e96891ba00d3d61a1eaa9c95bdd8d2663b183bd),
dated 2026-07-21.

Primary files:

- [Qwen GDN layer and core](https://github.com/vllm-project/vllm/blob/6e96891ba00d3d61a1eaa9c95bdd8d2663b183bd/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py)
- [GDN metadata backend](https://github.com/vllm-project/vllm/blob/6e96891ba00d3d61a1eaa9c95bdd8d2663b183bd/vllm/v1/attention/backends/gdn_attn.py)
- [GDN base state contract](https://github.com/vllm-project/vllm/blob/6e96891ba00d3d61a1eaa9c95bdd8d2663b183bd/vllm/model_executor/layers/mamba/gdn/base.py)
- [Qwen3.5 model and weight mapping](https://github.com/vllm-project/vllm/blob/6e96891ba00d3d61a1eaa9c95bdd8d2663b183bd/vllm/model_executor/models/qwen3_5.py)
- [Vendored FLA chunk orchestration](https://github.com/vllm-project/vllm/blob/6e96891ba00d3d61a1eaa9c95bdd8d2663b183bd/vllm/third_party/flash_linear_attention/ops/chunk.py)

### Runtime flow

1. `Qwen3_5DecoderLayer` selects full attention or the shared
   `QwenGatedDeltaNetAttention`; Qwen3.5-specific code mainly supplies layout
   flags and declarative checkpoint-to-fused-weight mapping.
2. `GDNAttentionMetadataBuilder` translates scheduler metadata into a typed GDN
   record: actual counts, decode/prefill/spec partitions, sequence offsets,
   state indices, accepted-token counts, chunk indices, initial-state masks,
   and graph-stable buffers.
3. CUDA `forward` has three visible parts: fused QKVZ/BA projection, one
   mutating `qwen_gdn_attention_core` custom op, then gated RMSNorm and output
   projection. The core output uses `torch.zeros`, not uninitialized storage.
4. The custom op resolves the static layer from forward context and contains
   convolution/state mutation. Projections and output remain compilable.
5. Decode-only non-spec batches take a packed recurrent fast path. General
   execution applies convolution, then speculative recurrent, ordinary decode
   recurrent, and prefill chunk work according to metadata. Mixed outputs are
   restored to original token order.
6. Prefill chunk metadata is computed before layer execution, preferably on
   CPU and copied asynchronously, avoiding per-layer device-to-host sync. Full
   Graph buffers are request-padded while `num_actual_tokens` guards real work.
7. Prefill backend selection happens once during construction. FlashInfer,
   CuteDSL, and Triton/FLA share one semantic wrapper; kernels remain private.
8. Chunk autotuning is warmed during profiling before cache allocation because
   first-request autotuning can otherwise exhaust remaining GPU memory.

### Code-style rules adopted by nano-vLLM

- Keep model `forward` short and readable at the semantic level.
- Build scheduler-derived metadata outside the model layer.
- Make mutation explicit through output buffers, custom-op schemas, and fake
  implementations for compilation.
- Separate stable public boundaries from private optimized fast paths.
- Select platform/backend implementations once, not in the token hot loop.
- Precompute reusable metadata and avoid hidden host/device synchronization.
- Keep state shape, dtype, and lifetime ownership explicit and centralized.
- Map checkpoint layout to fused runtime layout declaratively at load time.
- Comment non-obvious correctness/performance constraints and assert invariants
  where ownership changes.
- Do not copy vLLM's registry, LoRA, PP/TP, or multi-backend machinery unless
  nano-vLLM supports and tests that capability.

### Current nano-vLLM comparison

Already aligned:

- Three-part forward, fused QKVZ/BA, one mutating core op, and output projection.
- Zero-initialized core output buffer.
- ModelRunner-owned state slabs and explicit state slots.
- ModelRunner-built production chunk/recurrent partitions.
- Stable decode Graph slots and actual-token trimming.
- Declarative logical-to-fused checkpoint mapping.

Required follow-up:

- Remove `_build_packed_metadata` and transient request orchestration from the
  model layer; tests and warmup must construct runner-style metadata.
- Replace the non-uniform convolution Python request loop with a packed kernel
  or packed convolution operator.
- Add explicit chunk-prefill warmup before cache sizing even when Piecewise
  CUDA Graph capture is disabled.
- Keep `gated_delta_packed` as one semantic API while recurrent and chunk
  kernels remain private execution strategies.

## Change log

- 2026-07-22: Created from the active GDN refactor requirements.
- 2026-07-22: Clarified that this is one durable capability spec shared by
  separate implementation and source-study tasks.
- 2026-07-22: Studied vLLM main at `6e96891`; recorded metadata ownership,
  three-part forward, private fast paths, graph buffers, warmup behavior, and
  concrete nano-vLLM alignment work.
- 2026-07-22: Required the Stage 1 typed prepared-batch/context migration and
  complete removal of model-layer request metadata construction.
- 2026-07-22: Defined independent FP8 DeltaNet state configuration, scale
  granularity, branch lifecycle, capacity accounting, and fail-closed GPU
  enablement pending SM89 validation.
- 2026-07-22: Replaced uniform/non-uniform model-layer convolution branches
  with one packed causal-convolution entry. Output and state-commit remain two
  private kernels to preserve the read-old-state-before-overwrite dependency.
- 2026-07-22: Required per-prefix speculative state branches and pointer/index
  commit, eliminating rejected-prefix whole-model replay from the production
  path.
- 2026-07-22: Classified FP8 DeltaNet state as a capacity optimization gated by
  effective scheduler batch and SLO-goodput evidence, not byte savings alone.
