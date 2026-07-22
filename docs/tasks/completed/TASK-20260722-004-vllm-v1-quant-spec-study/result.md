# Result

Completed a documentation-only study of current vLLM V1 quantization and
speculative decoding at commit `6e96891...`.

Delivered:

- Quantization spec now defines checkpoint resolution, per-layer method
  lifecycle, packed loading/repacking, KV-cache independence, scale ownership,
  local alignment, and required follow-ups.
- Speculative-decoding spec now defines the unified V1 scheduler protocol,
  proposer/sampler separation, verification rollback, state ordering, metrics,
  local alignment, and explicit V0 exclusions.
- Durable knowledge pages summarize stable rules without task chronology.
- Existing runtime/source changes were not modified and no runtime claim was
  made from this documentation task.

Follow-ups are implementation tasks, not blockers for this study: extract a
small quant-method lifecycle before adding another format; extract MTP behind a
proposer interface; centralize hybrid cache/state commit; and retain real GPU
accuracy/performance validation for those changes.
