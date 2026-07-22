---
subject: cuda-graphs
title: CUDA Graph compilation, capture, and dispatch
status: active
created: 2026-07-22
updated: 2026-07-22
owner: Codex
---

# CUDA Graph compilation, capture, and dispatch

## Motivation

nano-vLLM needs CUDA Graph acceleration as an explicit execution policy that
works for decode, mixed batches, chunked prefill, quantized operators,
speculative verification, and stateful hybrid layers without changing model
semantics or silently falling back.

`torch.compile` is a compiler mechanism, not proof of CUDA Graph capture or
replay. The engine therefore owns graph compatibility, finite capture keys,
startup preparation, runtime dispatch, padding, stable buffers, memory
accounting, and observability.

## Runtime model

The public policy modes are:

| Policy | Uniform supported batches | Other supported batches |
| --- | --- | --- |
| `NONE` | eager | eager |
| `PIECEWISE` | Piecewise | Piecewise |
| `FULL_DECODE_ONLY` | Full | eager |
| `FULL_AND_PIECEWISE` | Full | Piecewise |

The concrete execution modes are only `FULL`, `PIECEWISE`, and `EAGER`.
`FULL_AND_PIECEWISE` is the default policy. A single dispatcher resolves each
runtime batch in priority order `FULL -> PIECEWISE -> EAGER`.

Full Graph captures the model forward, including graph-compatible attention or
stateful custom operators. Piecewise Graph captures compiled regions around
declared graph breaks; incompatible attention remains eager. Eager is either an
explicit policy or an observable fallback when no compatible captured key
exists.

## Capture key and batch descriptor

The dispatcher input must describe semantic shape rather than request phase.
The minimum descriptor is:

- real scheduled token count;
- real request count;
- uniform query length when every request has the same query length, otherwise
  `None`;
- operator/backend capability class;
- optional dimensions that change graph structure or stable buffers.

For the currently supported feature set, the stable graph key is conceptually:

```text
(execution_mode, padded_tokens, padded_requests?, uniform_query_len?,
 attention/state capability)
```

Dimensions not implemented by nano-vLLM, such as LoRA cardinality, are not
added pre-emptively. They must be added before that feature can replay an
existing graph.

`uniform_query_len` is deliberately not a Boolean decode flag. Ordinary decode
has value `1`; MTP verification may have `1 + k`; dynamic speculative decoding
may require a finite key set for every supported verification length. A
uniform `1 + k` batch can use Full Graph when every participating backend
declares uniform-batch support.

The selected descriptor records both real and padded dimensions. Padding is a
graph implementation detail and never changes scheduler accounting, sampling,
accepted-token accounting, cache positions, or user-visible output.

## Capture sizes and candidate selection

- Capture sizes are finite, deterministic, sorted, and known at startup.
- Compilation sizes and CUDA Graph capture sizes are separate concepts even
  when configured from the same default list.
- The current default token buckets remain `1, 2, 4`, then multiples of eight
  through the configured Piecewise limit, including the exact upper limit.
- The effective Piecewise limit is the minimum of the requested limit, the
  scheduled-token budget, and 512; request concurrency does not reduce it.
- Full candidates additionally encode request count and uniform query length;
  token count alone is insufficient for speculative verification.
- The dispatcher precomputes the smallest compatible captured candidate for a
  runtime descriptor. It does not create graphs on first traffic.
- If a Full candidate is unavailable, dispatch tries a compatible Piecewise
  candidate, then returns Eager with a reason.
- A bucket is compatible only when every key dimension is compatible; merely
  having enough token capacity is not sufficient.

## Backend compatibility

Graph compatibility is a declared operator capability, ordered from least to
most capable:

1. never graph-safe;
2. single-token uniform decode only;
3. any uniform batch, including speculative `1 + k` verification;
4. all supported batch shapes.

A hybrid model uses the minimum capability across all attention and stateful
backends. The configured policy is resolved before capture. An incompatible
Full policy may be downgraded to `FULL_AND_PIECEWISE`,
`FULL_DECODE_ONLY`, `PIECEWISE`, or `NONE` as appropriate, with an explicit
startup message. Runtime must not discover backend incompatibility by failing
inside a replay.

For Qwen3.6, both full attention and Gated DeltaNet must declare their
capability. Full Graph for a uniform batch is allowed only when stable KV,
convolution-state, recurrent-state, and metadata buffers are all safe for that
key.

## Compile, warmup, capture, and readiness

Startup owns the following ordered responsibilities:

1. resolve graph policy against backend capabilities;
2. build the complete finite candidate/key registry;
3. compile configured shapes and run kernel autotune/warmup in `EAGER` mode;
4. allocate or bind stable graph input, metadata, cache, and state buffers;
5. explicitly capture every configured Piecewise and Full key;
6. synchronize, verify the captured registry, publish graph memory, and only
   then declare the engine ready.

Piecewise capture is performed before Full capture when both share a graph
memory pool, because Piecewise activations are normally larger and Full can
then fit the established pool. Attention metadata that may be mutated or
initialized lazily is rebuilt for each warmup/capture case.

The runner, not the graph wrapper, owns warmup policy and stable inputs. A
wrapper owns capture and replay and may assert that input addresses match the
captured addresses. Runtime data is copied into stable buffers before replay.

Compilation or capture failure is fatal unless the operator explicitly
selected `NONE`. The error names the failed key and recommends eager mode; it
does not silently make the advertised Graph mode eager.

## Padding and state safety

- Dummy tokens use invalid cache slots and never write K/V payload or scales.
- Dummy rows never enter logits selection, sampling, speculative acceptance,
  token usage, or output.
- Dummy requests use dedicated inert state slots or a kernel mask. They must
  not mutate live, free, or prefix-shared recurrent/conv state.
- Full replay copies only real metadata and deterministically resets all padded
  portions of stable buffers on every replay.
- Piecewise eager attention receives real token metadata. Padding around the
  compiled regions must preserve row correspondence without exposing dummy
  rows to attention/cache writes.
- Abort, preemption, prefix sharing, and speculative rejection preserve the
  same cache/state ownership rules in all execution modes.

## Feature interactions

### Quantization

W4A16, FP8 weights, and FP8 KV custom operators declare compile/graph support
and stable workspace requirements. Packed weights and KV pages remain at fixed
addresses. A quantized path must not add an unexpected graph break or allocate
a full dequantized weight/cache during capture or replay.

### Speculative decoding

The target verification key includes the uniform query length. For fixed
`k`, uniform verification uses `uniform_query_len = 1 + k`; dynamic `k`
captures the supported finite lengths. Proposal-model execution has its own
compatibility and capture decisions. Accepted-prefix commit and rejected-state
rollback are semantic operations outside any assumption that a replayed batch
accepts every drafted token.

### Stateful hybrid layers

Gated DeltaNet state slabs and state-slot metadata are persistent graph inputs.
Full Graph requires numerical parity for all captured batch sizes and uniform
query lengths. Restricting Full to a proven subset is valid, but the restriction
must be represented by the candidate registry and reported as a fallback
reason, not hidden in ad hoc batch classification.

## Memory accounting

KV-cache and recurrent-state capacity planning includes estimated or measured
persistent compiler caches, CUDA Graph private pools, stable buffers, and
operator workspaces. Actual capture may occur after cache allocation when the
runtime requires cache addresses, but graph memory must already be reserved in
the capacity plan and actual-versus-estimated memory must be reported.

The engine must not compute a maximal KV cache and then allocate Full Graph
pools from untracked utilization headroom.

## Observability and benchmark contract

Startup logs and benchmark JSON record:

- requested and resolved policy;
- capture-size list and complete captured keys;
- compile, warmup, Piecewise capture, and Full capture time;
- estimated and actual persistent graph memory;
- per-mode steps, real tokens, padded tokens, time, and throughput;
- per-key capture and replay counts;
- fallback count and structured reason;
- speculative verification length and graph mode;
- state/attention backend capability that constrained the resolved policy.

`execution_mode` alone is not enough to prove Piecewise capture. GPU evidence
must show a captured key and subsequent replay count or equivalent profiler/
framework evidence.

## Current vLLM V1 reference conventions

This contract was checked against vLLM commit
[`6e96891ba00d3d61a1eaa9c95bdd8d2663b183bd`](https://github.com/vllm-project/vllm/commit/6e96891ba00d3d61a1eaa9c95bdd8d2663b183bd)
from 2026-07-21:

- [`CUDAGraphMode` and compilation configuration](https://github.com/vllm-project/vllm/blob/6e96891ba00d3d61a1eaa9c95bdd8d2663b183bd/vllm/config/compilation.py)
  distinguish public policy from concrete runtime modes and keep compile sizes
  separate from capture sizes.
- [CUDA Graph design](https://github.com/vllm-project/vllm/blob/6e96891ba00d3d61a1eaa9c95bdd8d2663b183bd/docs/design/cuda_graphs.md)
  defines central dispatch, Full/Piecewise nesting, backend capability, and
  uniform speculative batches.
- [MRv2 CUDA Graph manager](https://github.com/vllm-project/vllm/blob/6e96891ba00d3d61a1eaa9c95bdd8d2663b183bd/vllm/v1/worker/gpu/cudagraph_utils.py)
  precomputes compatible finite candidates and captures dynamic speculative
  query lengths.
- [MRv2 model runner](https://github.com/vllm-project/vllm/blob/6e96891ba00d3d61a1eaa9c95bdd8d2663b183bd/vllm/v1/worker/gpu/model_runner.py)
  owns stable buffers, dispatch, warmup, capture, and runtime input copies.
- [Attention backend contract](https://github.com/vllm-project/vllm/blob/6e96891ba00d3d61a1eaa9c95bdd8d2663b183bd/vllm/v1/attention/backend.py)
  exposes graph capability; hybrid models resolve against the least-capable
  backend.
- [GPU worker memory planning](https://github.com/vllm-project/vllm/blob/6e96891ba00d3d61a1eaa9c95bdd8d2663b183bd/vllm/v1/worker/gpu_worker.py)
  includes CUDA Graph memory estimates in available-cache calculation and later
  compares actual capture memory with the estimate.

V0 graph runners, V0 worker orchestration, and version-specific wrapper class
layout are not design references. The stable V1 ownership and semantics above
are the contract.

## Current nano-vLLM alignment and gaps

Already aligned:

- default `FULL_AND_PIECEWISE` policy and `FULL -> PIECEWISE -> EAGER`
  priority;
- finite Full/Piecewise buckets and explicit eager fallback beyond the range;
- explicit Full capture with stable buffers;
- Piecewise attention graph break and real-token-only attention metadata;
- early Piecewise compile/warmup before KV capacity calculation;
- per-step execution-mode metrics and offline mode aggregation;
- fatal startup error when configured compilation/capture raises.

Required follow-ups:

1. Replace `uniform_decode: bool` with a uniform query length. Current batches
   qualify for Full only when every request schedules exactly one token, so MTP
   verification with `1 + k` incorrectly falls to Piecewise.
2. Key Full candidates by token count, request count, and uniform query length;
   capture stable metadata for every supported `1 + k` verification length.
3. Replace the current two-call `torch.compile(mode="reduce-overhead")`
   Piecewise preparation with an explicit captured-key registry and evidence of
   capture/replay. Compiler invocation alone is not sufficient evidence.
4. Add an explicit backend capability contract for full attention, FP8 paged
   attention, and Gated DeltaNet. The current one-sequence GDN Full limit is an
   appropriate safety restriction but should be a declared candidate limit.
5. Reserve or estimate Full Graph pool/stable-buffer memory before final KV and
   DeltaNet capacity. Current code captures Full graphs after allocating the
   maximal cache from the then-current memory state.
6. Report padded tokens, capture keys, replay counts, fallback reasons, and
   graph memory. Current `last_execution_mode` cannot prove a Piecewise replay.
7. Add GPU state-mutation parity tests for padded GDN batches and MTP
   accept/reject paths; final token equality alone does not prove state safety.

## Acceptance criteria

- Each runtime batch maps deterministically to `FULL`, `PIECEWISE`, or `EAGER`
  with a documented capture key and fallback reason.
- Ordinary decode and supported uniform MTP verification use Full Graph; mixed
  and prefill use Piecewise; unsupported shapes visibly use Eager.
- Graph replay matches eager token output and KV/recurrent-state mutation for
  decode, mixed, prefill, quantized, speculative, and hybrid-state cases.
- Startup logs and benchmark JSON prove which keys were captured and replayed
  and how much persistent memory they consume.
- BF16/FP8 KV, W4A16, MTP, Gated DeltaNet, online serving, prefix cache,
  preemption, and abort regressions pass on the CUDA server.

## Constraints

Real capture/replay, persistent-memory, numerical-state, and latency evidence
requires the CUDA server environment. CPU tests validate only policy, keys,
candidate selection, padding metadata, and fallback reasons.

## Non-goals

- Copying all vLLM platform backends, LoRA graph cases, or compilation-pass
  infrastructure before nano-vLLM supports those features.
- Runtime creation of unbounded graphs for previously unseen shapes.
- Treating a compiler cache hit, matching final text, or `execution_mode` label
  as sufficient capture/replay proof.

## Open questions

- Whether nano-vLLM should use explicit Piecewise CUDA Graph wrappers or retain
  Inductor CUDAGraph Trees behind a verifiable captured-key adapter.
- The measured Full/Piecewise persistent-memory reserve on the 24 GiB target
  GPU for Qwen3.6-27B W4A16 with FP8 KV.

## Change log

- 2026-07-22: Created for a current vLLM V1 CUDA Graph source study and review
  of nano-vLLM's existing `FULL_AND_PIECEWISE` implementation.
- 2026-07-22: Added pinned V1 policy, keys, backend capability, startup/memory,
  quantization/speculative/stateful integration rules, local alignment, and
  implementation gaps.
- 2026-07-23: Fixed Piecewise capacity to follow the configured token budget up
  to 512 instead of being truncated by decode concurrency, and retained the
  requirement for captured-key/replay evidence beyond execution-mode labels.
