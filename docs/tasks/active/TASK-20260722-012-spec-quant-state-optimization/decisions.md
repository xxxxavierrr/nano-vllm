# Decisions

## 2026-07-22

- Correct the W4 diagnosis precisely: the current path is one Triton launch,
  but still repeats unpack/dequant work and lacks a runtime-optimized layout.
- Implement branch-state selection before probabilistic sampling because
  probability-based rejection would otherwise make expensive replay common.
- Preserve greedy acceptance as a dedicated fast policy.
- Require exact lossless residual sampling; sampling directly from target after
  rejection is statistically incorrect.
- Prefer indexed state commit over tensor rollback or target-model replay.
- Keep raw-`g_idx` Triton as the correctness fallback; make repack and kernel
  choice post-load runtime concerns.
- Evaluate repacked Triton first, but treat an RTX 4090D/SM89 Marlin-style CUDA
  kernel as the likely production path if Triton cannot deliver packed W4
  tensor-core dataflow efficiently.
- Do not implement or commit locally; implementation and GPU evidence belong on
  the GPU server after design approval.
- The target device is RTX 4090D 24 GB. Earlier SM86/RTX 3090 assumptions are
  invalid; branch-state and draft-logit designs must fit beside the W4 model,
  KV cache, and CUDA Graph pools within this fixed budget.
- Optimization priority is W4A16 small/large-M, indexed DeltaNet branches,
  lossless rejection, W4A8 large-M, FP8 KV capacity, then FP8 DeltaNet-state
  capacity. W3A16 is last and requires proof that memory still limits useful
  concurrency.
- Default-setting decisions use maximum goodput under a declared latency SLO.
  Kernel latency, compression ratio, acceptance rate, and equal-concurrency
  throughput are diagnostics rather than independent success criteria.
- Use an open-loop offered-load sweep for maximum serving throughput and a
  closed-loop concurrency sweep for saturation/capacity diagnosis.
- The paired AWQ target/BF16 DSpark result is an external acceptance reference.
  On the 24 GB device the BF16 draft is offline-only, and the first runnable
  online baseline uses the paired AWQ target plus an INT4 draft.
- Mark branch-state and probability-difference sampling as implemented pending
  GPU validation, not as future implementations. Mark FP8 KV as implemented
  pending capacity/goodput measurement.

## 2026-07-23

- Keep `auto` on repacked Triton. Native W4A16 is correct and Graph-safe but
  remains explicit opt-in because every measured K=N=5120 point is slower.
- Do not call the WMMA prototype Marlin-complete. True Marlin work requires a
  load-time Marlin weight layout and a multi-stage asynchronous
  global-to-shared pipeline/warp-specialized dataflow, not only WMMA tiles.
- Retain the best measured small/large template specialization and document
  rejected tiles; do not continue blind constant tuning without hardware
  profiler or a faithful Marlin port.
- Archive typed-batch, speculator, and typed-metrics extraction tasks after GPU
  integration evidence. Archive GDN/state/semantic-Graph tasks as superseded
  only after their unresolved Graph/performance gates are copied into this
  optimization task.
