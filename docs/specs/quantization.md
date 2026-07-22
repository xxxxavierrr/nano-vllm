---
subject: quantization
title: Model and KV-cache quantization
status: active
created: 2026-07-22
updated: 2026-07-23
owner: Codex
---

# Model and KV-cache quantization

## Motivation

nano-vLLM needs quantization as an explicit inference capability spanning
checkpoint recognition, packed parameter loading, kernels, cache allocation,
graph compatibility, accuracy, observability, and benchmark comparison.

## Requirements

- Keep checkpoint format, runtime quantization method, kernel capability, and
  cache dtype as separate concepts.
- Resolve weight quantization once before model construction:
  `checkpoint metadata + explicit override -> quantization config -> per-layer
  quant method`. A serialized format name must not be used as a kernel name.
- Detect serialized quantization metadata and reject incompatible explicit
  overrides rather than silently reinterpret weights.
- A layer quant method owns three distinct lifecycle hooks: production
  parameter creation, post-load validation/repacking, and forward application.
  Repacking and calibration must not occur in the request hot path.
- Quantized modules allocate their production storage directly; they must not
  materialize full BF16 weights on GPU.
- The production W4A16 fast path must combine packed-weight decode,
  dequantization, and matrix multiplication in one runtime operator without a
  global BF16 weight or dequantized-weight scratch tensor. A reference or
  fallback implementation may dequantize explicitly, but it is not the
  performance backend.
- GPTQ checkpoint layout and runtime kernel layout are distinct. Any weight
  permutation, interleave, zero-point normalization, or scale repack happens
  once after loading and is owned by the selected quantization method.
- Weight quantization and KV-cache quantization remain independently selectable.
- Quantization decisions are per layer/module and prefix. Unsupported layers or
  explicit skip rules may use another method only through a documented,
  validated policy; otherwise construction fails.
- Packed/tensor layout, dtype, group/scale/zero semantics, device capability,
  TP restrictions, and fused-shard invariants are validated at load time.
- Fused QKV and gate/up loading preserves each logical shard's packed metadata.
  In particular, GPTQ `g_idx` tensors must be compared element by element and
  fusion must fail if they differ.
- KV-cache allocation is derived from a cache specification that includes the
  per-layer dtype, shape, scales, and backend support. It must not be inferred
  from the model weight dtype.
- KV scale ownership is explicit: checkpoint/static calibration, warmup
  calibration, and per-token/per-head dynamic scaling are different modes and
  cannot be mixed silently. Layers excluded from KV quantization keep their
  native cache dtype.
- Quantized operators work under the configured eager/Full/Piecewise Graph
  modes without hidden host synchronization or first-request compilation.
- Accuracy and performance are measured against the same model/workload in a
  memory-safe sequential comparison.
- Calibration separates Hessian conditioning, scale derivation, blockwise
  error propagation, and packed serialization. The public quantizer expresses
  their order without implementing every numerical phase inline.
- FP8 DeltaNet layout/capacity arithmetic, CPU reference codecs, state-pool
  lifecycle, and experimental runtime kernels remain separate modules with a
  compatibility facade for existing imports.
- Kernel selection is split by numerical shape/hardware regime, not by request
  label. W4A16 must cover optimized small-M and large-M paths; fused W4A8 is a
  later large-M optimization after W4A16, state branching, and rejection
  correctness are established.
- FP8 KV cache and FP8 DeltaNet state are capacity optimizations. They are
  selected by maximum SLO goodput at each mode's best stable concurrency, not
  by equal-concurrency latency alone.
- FFN W3A16 is deferred until W4, Graph, speculative state, KV, and recurrent
  state measurements show that weight capacity still prevents the desired
  scheduler concurrency.

## Scope

- GPTQ W4A16 weights, FP8 post-load weights, and FP8 E4M3 KV cache.
- Configuration, model loading, linear/attention dispatch, allocation,
  observability, correctness, and benchmark methodology.
- Current vLLM V1 quantization conventions relevant to nano-vLLM.

## Non-goals

- Supporting every vLLM quantization plugin.
- Silent conversion between AWQ, GPTQ, GGUF, Marlin, or compressed-tensor
  checkpoint formats.
- Training-time quantization.

## Acceptance criteria

- Quantization selection and checkpoint compatibility have one documented
  resolution path.
- Weight/cache allocation accounts for payload, metadata, scales, workspace,
  and graph memory.
- Correctness, accuracy, memory, kernel, offline, and online evidence is
  recorded for each supported method.
- V1 source conventions and intentional nano-vLLM differences are documented.

## Architecture and ownership

The required weight path is:

```text
model/checkpoint config
        -> quantization resolver
        -> QuantizationConfig-like capability object
        -> get method for (layer type, module prefix)
        -> create packed production parameters
        -> checkpoint weight loaders fill logical/fused shards
        -> validate and optionally repack once
        -> forward dispatches the selected kernel
```

The cache path is independent:

```text
cache dtype / per-layer skip policy / scale mode
        -> attention-backend compatibility validation
        -> per-layer cache specification and byte accounting
        -> cache allocation
        -> cache-write quantization and attention-time dequantization/use
```

`Config` owns user/checkpoint resolution. Layers own their selected quant
method and parameters. The loader owns tensor routing and completeness. Kernels
own only supported numerical layouts. The cache manager owns cache lifetime and
capacity; attention owns scale consumption.

## Current vLLM V1 source conventions

The source study is pinned to vLLM commit
[`6e96891ba00d3d61a1eaa9c95bdd8d2663b183bd`](https://github.com/vllm-project/vllm/commit/6e96891ba00d3d61a1eaa9c95bdd8d2663b183bd)
(2026-07-21). Relevant current contracts are:

- [`QuantizationConfig` and `QuantizeMethodBase`](https://github.com/vllm-project/vllm/blob/6e96891ba00d3d61a1eaa9c95bdd8d2663b183bd/vllm/model_executor/layers/quantization/base_config.py)
  separate format/config resolution from `create_weights`, `apply`, and
  `process_weights_after_loading`.
- [`LinearBase`](https://github.com/vllm-project/vllm/blob/6e96891ba00d3d61a1eaa9c95bdd8d2663b183bd/vllm/model_executor/layers/linear.py)
  selects a method using layer type and full prefix, creates the method's
  parameters during construction, and delegates forward to `apply`.
- The [quantization registry](https://github.com/vllm-project/vllm/blob/6e96891ba00d3d61a1eaa9c95bdd8d2663b183bd/vllm/model_executor/layers/quantization/__init__.py)
  is extensibility infrastructure, not the runtime abstraction itself. nano-vLLM
  needs the lifecycle boundary, not vLLM's full plugin catalog.
- [`AutoGPTQConfig`](https://github.com/vllm-project/vllm/blob/6e96891ba00d3d61a1eaa9c95bdd8d2663b183bd/vllm/model_executor/layers/quantization/auto_gptq.py)
  shows that a GPTQ checkpoint may select an optimized runtime method and that
  per-prefix rules can differ. This is why checkpoint format and kernel choice
  remain separate.
- [`BaseKVCacheMethod`](https://github.com/vllm-project/vllm/blob/6e96891ba00d3d61a1eaa9c95bdd8d2663b183bd/vllm/model_executor/layers/quantization/kv_cache.py)
  owns scale loading/post-processing. Its per-token/per-head mode explicitly
  ignores checkpoint scales because the cache-write kernel computes them.
- Current vLLM documentation exposes KV-cache quantization and per-layer skip
  policy independently from weight quantization; this is consistent with V1's
  per-layer cache specifications rather than a model-dtype shortcut.

Quantization is not a separate V1 scheduler subsystem. V1 reuses model-layer
quant methods and cache specifications; request scheduling must remain unaware
of whether a linear kernel reads BF16, FP8, or packed INT4 weights.

## nano-vLLM alignment and required follow-ups

Already aligned:

- GPTQ allocates `qweight/scales/qzeros/g_idx` directly and keeps full BF16
  dequantization in test/reference code only.
- Fused GPTQ shards reject unequal `g_idx` rather than adopting vLLM's weaker
  assumption that the metadata matches.
- W4A16 forward is a traceable operator and FP8 KV cache is selected
  independently from weight quantization.
- FP8 KV uses explicit per-token/per-KV-head dynamic scales and includes scale
  bytes in capacity accounting.

Follow-up architecture requirements:

- Replace the growing `Config` condition tree with a small quantization resolver
  and per-layer method interface before adding another serialized format.
- Move FP8 post-load conversion behind the same layer lifecycle as GPTQ instead
  of model-wide type checks.
- Represent hybrid-model KV allocation as per-layer cache specifications,
  including native-dtype recurrent state and any layer skip policy.
- Report packed payload, metadata, repack workspace, graph memory, cache scales,
  and recurrent state separately; a single "quantized model bytes" number is
  insufficient.
- Keep nano-vLLM's direct raw-`g_idx` kernel as the correctness baseline. Weight
  shuffle/permutation is a measured kernel optimization, not a checkpoint
  semantic. The optimized backend may precompute `argsort(g_idx)`, repack
  weights, and fuse the matching activation permutation into its input loads;
  a standalone request-time gather is not acceptable.
- Select W4A16 runtime kernels by tensor shape and hardware capability rather
  than scheduler labels such as prefill/decode. The current raw-layout Triton
  operator remains a correctness/fallback path; the production target is an
  Ada SM89/RTX 4090D-oriented Marlin-style CUDA kernel whose dataflow does not
  construct a reusable BF16 weight tile in global memory.
- Native CUDA extension construction is explicit opt-in through
  `NANOVLLM_BUILD_CUDA_EXT=1`. Until RTX 4090D validation is recorded,
  `auto` remains Triton; `marlin` is an explicit experimental request that
  fails if the SM89 extension or normalized symmetric repacked layout is
  unavailable. Native W4A16 dispatch uses `M<=64` and `M>64` shape regimes;
  experimental W4A8 is large-M only and creates no global INT8 activation
  scratch.

## Constraints

GPU correctness/performance claims require the server environment and real
target checkpoint shapes.

## Open questions

- Which quantized formats should share kernels versus remain separate loaders?
- When a future backend cannot support mixed per-layer cache dtypes, should the
  configuration fail or require an explicit all-native fallback policy?

## Change log

- 2026-07-22: Created for the current GPTQ/FP8 weight and FP8 KV capabilities
  and a source study of vLLM V1 conventions.
- 2026-07-22: Added the pinned vLLM V1 resolution/layer/cache lifecycle,
  nano-vLLM alignment, stricter fused-`g_idx` invariant, and follow-up gaps.
- 2026-07-22: Required load-time GPTQ-to-runtime repacking and a fused W4A16
  production operator; clarified that the current one-launch Triton path is a
  correctness fallback rather than a two-kernel dequantization design.
- 2026-07-22: Corrected the target GPU to RTX 4090D 24 GB (Ada SM89); kernel
  selection, workspace, and state/KV capacity benchmarks must use that target.
- 2026-07-22: Ordered optimization around Marlin-style W4A16 small/large-M,
  then fused W4A8 large-M and measured FP8 cache/state capacity; deferred
  W3A16 until goodput evidence proves memory remains the limiting resource.
- 2026-07-22: Added the opt-in SM89 native-extension contract, safe Triton
  default, explicit native layout validation, and experimental large-M W4A8
  source boundary; CUDA validation remains required before enablement.
- 2026-07-23: Added calibration-phase and FP8 DeltaNet module ownership
  boundaries; public orchestration may not absorb their internal algorithms.
- 2026-07-23: Recorded SM89 native W4A16 numerical/fullgraph/direct-Graph
  success and raw latency failure versus repacked Triton. `auto` remains
  Triton; future native enablement requires Marlin layout plus an asynchronous
  multi-stage pipeline, not WMMA tiling alone.
