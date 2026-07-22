# Research

## Baseline and source policy

- Upstream: vLLM commit
  [`6e96891ba00d3d61a1eaa9c95bdd8d2663b183bd`](https://github.com/vllm-project/vllm/commit/6e96891ba00d3d61a1eaa9c95bdd8d2663b183bd),
  dated 2026-07-21.
- Current official CUDA Graph and `torch.compile` design pages supplied public
  terminology; pinned source code was authoritative for exact ownership and
  behavior.
- The pinned tree contains both Model Runner V1 and Model Runner V2. Stable V1
  semantics and the current MRv2 graph manager were promoted; legacy V0 graph
  runners were excluded.

## Upstream execution policy

1. `CUDAGraphMode` separates single concrete modes (`NONE`, `PIECEWISE`,
   `FULL`) from dual policies (`FULL_DECODE_ONLY`, `FULL_AND_PIECEWISE`).
2. V1 with Piecewise compilation defaults to `FULL_AND_PIECEWISE`.
3. A central dispatcher owns available keys and prioritizes Full over Piecewise
   over eager/no graph.
4. Compile sizes and capture sizes are configured separately.
5. The configured mode is checked against attention-backend capability and may
   be resolved to a supported policy before capture.

## Keys and speculative batches

The current MRv2 execution descriptor records graph mode, tokens, requests,
uniform token count, and active LoRA count. Full uniform candidates round token
and request dimensions together and retain uniform query length. Piecewise
usually needs only the token bucket.

Uniform decode is not synonymous with query length one. Target verification
for speculative decoding has a uniform query length such as `1+k`. The graph
manager creates candidates for every configured fixed/dynamic verification
length that can occur. Compatibility compares all relevant descriptor fields;
token capacity alone does not make a key reusable.

## Compile, capture, and replay

- The graph manager creates a finite candidate set at startup.
- Capture visits Piecewise candidates before Full candidates and visits larger
  descriptors first, enabling the shared memory pool to accommodate the larger
  Piecewise activations.
- Each capture descriptor receives an eager warmup. Attention metadata is
  rebuilt when capture can mutate or lazily initialize it.
- The runner owns persistent input/metadata buffers and copies runtime data
  before replay. The wrapper owns capture/replay and can verify input pointer
  identity.
- Dummy capture rows are explicitly marked padding.
- Runtime logging contains unpadded and padded dimensions and selected mode.

## Backend and hybrid-state compatibility

Attention backends expose ordered support: never, single-token uniform decode,
uniform batch, or all batches. A hybrid model uses the lowest capability across
its groups. At the pinned revision FlashAttention 2 and GDN both declare
uniform-batch support, so a hybrid model can use Full Graph for uniform
`1+k` verification when its stable state buffers and metadata are correct.

## Startup and memory

The worker profiles compile/activation requirements and estimates CUDA Graph
memory before committing cache capacity. Kernel warmup/autotune precedes actual
capture. Actual graph memory is later compared with the estimate. The stable
lesson is capacity reservation, not a requirement that actual capture occur
before cache tensors whose addresses are needed by the graph.

## Local comparison

Present nano-vLLM behavior:

- implements the four requested public policies and the concrete
  Full/Piecewise/Eager modes;
- uses finite Piecewise and Full token buckets and Full-first dispatch;
- executes explicit Full capture with stable buffers;
- uses a compiler-disabled attention boundary for Piecewise and pads compiled
  regions while attention sees real tokens;
- prepares Piecewise before KV/cache capacity and captures Full afterward;
- exposes per-step mode and aggregates mode steps/tokens/time in offline bench.

Identified gaps:

1. `uniform_decode` is true only when every sequence schedules one token. MTP
   verification with `1+k` therefore cannot select Full Graph.
2. Full keys contain only a token bucket and a Boolean; request count and
   uniform query length are missing.
3. Piecewise startup calls the compiled model twice per size and assumes
   Inductor CUDAGraph Trees captured it. There is no explicit captured registry
   or replay evidence.
4. GDN Full Graph is intentionally limited to one sequence, but this is an
   ad hoc capture-size restriction rather than a backend capability result.
5. Full graph pools are allocated after KV and DeltaNet capacity is maximized;
   their memory is not explicitly reserved in that capacity formula.
6. Metrics identify only the broad execution mode. They omit padded tokens,
   key, replay count, fallback reason, and graph memory.
7. Existing comparison tooling checks final tokens but does not compare KV/GDN
   state mutation or prove Piecewise capture/replay.

## Obsolete or non-applicable material excluded

- V0 CUDA Graph runners and V0 worker orchestration.
- Copying LoRA key dimensions before nano-vLLM supports LoRA.
- Treating current MRv1 capture-size rounding as the preferred speculative key
  model; MRv2 explicitly keys uniform verification length.
- Copying version-specific wrapper class layout instead of stable ownership.
