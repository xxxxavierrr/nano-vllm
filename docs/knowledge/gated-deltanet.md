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
- Speculative GDN state is a set of indexed prefix branches. Acceptance commits
  the selected conv/recurrent slot mapping and releases the rest; it does not
  restore a tensor snapshot or replay rejected target tokens.
- Zero-replay evidence requires an explicit counter in addition to output
  correctness. The Qwen3.6 MTP k=2 smoke observed branch commits/discards and
  `rejected_prefix_target_replays=0`.
- GPU integration does not replace dedicated recurrent/chunk crossover and
  semantic Full/Piecewise dummy-slot tests; those remain goodput/Graph gates.
