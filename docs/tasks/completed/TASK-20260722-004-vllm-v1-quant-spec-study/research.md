# Research

## Baseline and source policy

- Upstream: vLLM commit
  [`6e96891ba00d3d61a1eaa9c95bdd8d2663b183bd`](https://github.com/vllm-project/vllm/commit/6e96891ba00d3d61a1eaa9c95bdd8d2663b183bd),
  dated 2026-07-21.
- Source code at that commit is authoritative. Latest developer-preview docs
  were used only to confirm current public configuration and terminology.
- The pinned tree contains Model Runner V1 and Model Runner V2. Stable protocol
  conclusions were checked against the shared V1 scheduler; MRv2 was used for
  the current proposer/sampler separation.

## Quantization trace

1. `vllm/config/quantization.py` resolves user-facing online quantization
   arguments separately from serialized checkpoint quantization.
2. The quantization registry maps a resolved name to a `QuantizationConfig`
   class. The registry is plugin infrastructure, not the numerical runtime.
3. `QuantizationConfig` validates activation dtype/hardware/config and selects a
   `QuantizeMethodBase` by layer instance and full module prefix.
4. `LinearBase` stores the selected method, calls `create_weights` during
   construction, and delegates forward to `apply`.
5. Model loading calls `process_weights_after_loading` once for each quantized
   module. This is the supported place for repack, transpose, replacement, and
   final validation.
6. GPTQ demonstrates that serialized GPTQ may dispatch to an optimized runtime
   method, and dynamic per-prefix rules may change or skip quantization.
7. KV-cache scale loading/post-processing uses a quant-method lifecycle on the
   attention layer, while cache dtype/allocation belongs to the cache config and
   V1 KV-cache specification. Per-token/per-head scale mode computes scale at
   cache-write time and intentionally ignores checkpoint scale values.

## Speculative-decoding trace

1. The V1 scheduler states that it has no prefill/decode phase split. A request
   owns prompt, output, and speculative tokens; scheduling closes the gap from
   `num_computed_tokens` to `num_tokens_with_spec` under one token budget.
2. Scheduled draft IDs are truncated to the available scheduled range and
   emitted in `SchedulerOutput`. Request draft IDs are cleared so stale
   proposals cannot be reused.
3. The scheduler optimistically advances computed tokens for in-flight work.
   After target output, it derives accepted/rejected counts and subtracts
   rejected draft positions from the committed computed position.
4. In MRv2, `BaseSpeculator.propose` is separate from `RejectionSampler`.
   MTP is one `BaseSpeculator` implementation; target logits and optional draft
   logits are consumed by the sampler/acceptance component.
5. The MRv2 model runner orders target forward, rejection sampling,
   post-processing of sampled/rejected state, then the next proposal. Async host
   output copy may overlap proposal work but does not change state ordering.
6. Metrics count draft rounds, drafted/accepted tokens, and per-position
   acceptance. Mean acceptance length includes the bonus target token.

## Obsolete material excluded

- V0 `SpecDecodeWorker` and separate draft-worker/target-worker orchestration.
- V0 `MultiStepWorker` as the mechanism for speculative steps.
- Legacy `docs/features/spec_decode.md` descriptions of those worker paths.
- MRv1-only helper structure as a new nano-vLLM interface. The pinned tree
  retains MRv1 for compatibility, but the shared V1 scheduler semantics and
  current MRv2 proposer/sampler contract are the design references.

## Local comparison

Quantization is numerically ahead of its abstraction: GPTQ direct packed
allocation, strict fused `g_idx`, W4A16, and FP8 KV cache are present, but
checkpoint resolution and FP8 post-load behavior remain centralized type
branches in `Config`/`ModelRunner`. A lightweight per-layer method lifecycle is
needed before adding more formats. Hybrid cache accounting should become an
explicit per-layer cache specification.

Speculative decoding already stores drafts on sequences, budgets verification,
supports arbitrary greedy accepted prefixes, maintains working/committed GDN
state, and reports useful metrics. The main gap is ownership: MTP loading,
proposal, acceptance orchestration, replay, and metrics remain concentrated in
`ModelRunner`. The next refactor should extract proposer and acceptance
interfaces without changing the scheduler protocol.
