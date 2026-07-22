# Gated DeltaNet engineering rules

The capability contract and pinned upstream references live in the
[Gated DeltaNet spec](../specs/gated-deltanet.md).

- `ModelRunner` owns request partitioning, offsets, state slots, actual/padded
  counts, and recurrent/chunk metadata.
- The model layer owns projections, reshaping, one opaque stateful core, gated
  normalization, and output projection.
- A unified public core may use separate private recurrent-decode and
  chunk-prefill kernels. Mixed scheduling does not erase different numerical
  parallelism.
- Stateful custom ops declare mutation and provide fake implementations.
- No production request loop or host synchronization belongs in the token hot
  path.
- GDN prefill kernels compile/autotune before cache sizing.
- Full Graph state-slot/metadata buffers need stable storage; real token counts
  remain distinct from padded capture sizes.
- Checkpoint fusion belongs in declarative loader mapping, with quantization
  invariants checked at the fusion boundary.

