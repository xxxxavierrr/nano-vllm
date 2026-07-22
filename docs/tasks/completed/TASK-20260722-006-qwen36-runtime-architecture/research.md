# Research

## Source inventory

The assessment reconciled the current GDN, quantization, speculative-decoding,
and CUDA Graph specs and inspected the current runtime implementation.

Key size/ownership facts:

- `model_runner.py`: about 1,500 lines and owns initialization, loading,
  warmup, cache allocation, GDN state, batch metadata, MTP proposal/rollback,
  model execution, Graph capture/replay, and last-step metrics.
- `qwen3_5.py`: about 780 lines and currently combines model architecture,
  GDN metadata fallback/orchestration, fused checkpoint mapping, state shape,
  and model forward.
- `Context` contains more than twenty optional, weakly grouped fields and is
  manually set/reset by target, replay, MTP proposal, recursive MTP, warmup,
  and Graph paths.
- `LinearBase` contains BF16, FP8, and GPTQ branches; GPTQ config is threaded
  through Qwen constructors, while FP8 is applied by a later model traversal.
- target and MTP checkpoints use separate loaders; MTP explicitly constructs
  its layer with `quant_config=None`.
- physical KV tensors, FP8 scales, MTP KV, GDN committed/working slabs, graph
  pools, and capacity decisions are allocated in `ModelRunner`.

## Existing strengths

- Scheduler already has one token-budget path for prefill/decode/speculative
  verification and request-owned draft IDs.
- Logical KV block management remains scheduler-side; request objects do not
  contain GPU tensors.
- GDN refactor is converging on one custom-op boundary and one formal packed
  API with private recurrent/chunk kernels.
- GPTQ packed loading, strict fused `g_idx`, W4A16, FP8 KV, and online/offline
  benchmarks are functional foundations.
- DP is correctly implemented as routing across independent EngineProc
  replicas, outside the rank-local model runtime.

## Coupling that blocks extension

### GDN

The model layer still has `_build_packed_metadata` and transient fallback
orchestration. State allocation and working/commit/replay logic lives in the
runner. Adding another hybrid state-space layer would repeat both patterns.

### Quantization

Checkpoint format, model constructor parameters, packed parameter creation,
post-load conversion, runtime kernel choice, and cache dtype use separate
feature branches rather than one resolved lifecycle. A quantized draft model
or another format would multiply those branches.

### Speculative decoding

The scheduler protocol is reusable, but `_run_mtp_proposal` builds all tensors,
cache slots, contexts, recursive steps, logits, and draft chains inside the
runner. Acceptance, GDN working state, rejected-prefix replay, and metrics are
also runner concerns. DSpark/tree proposal would duplicate this orchestration.

### CUDA Graph

Graph dispatch consumes only token count and a decode Boolean. It cannot
represent uniform `1+k` MTP verification or backend capability. Piecewise
capture evidence and Full graph memory reservation are missing. Graph code also
constructs attention/GDN stable metadata itself.

## Architectural conclusion

The first abstraction should be a typed prepared batch and semantic execution
signature. It is the common dependency needed to move GDN metadata out of the
model, extract proposal input construction, form correct graph keys, and make
state/resource ownership testable.

The next correctness boundary is a shared physical cache/state manager with a
speculative state transaction. Graph and proposer extraction should build on
those two foundations. Quantization lifecycle extraction is important for
future formats/draft precision but can follow because the current numerical
path works and should remain stable during state/graph changes.
