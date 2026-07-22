# Result

## Delivered locally

- Four pre-existing feature groups were separated into independently reviewed
  local commits; nothing was pushed.
- Benchmark schema v3 now measures SLO-good output-token throughput,
  accepted-token throughput when provided, planned-arrival and service
  latency, time-weighted client/engine occupancy, and actual versus padded
  scheduled work.
- `--mode online-sweep` performs offered-load growth and boundary refinement;
  optional `nvidia-smi` telemetry records sampling provenance or an explicit
  missing reason.

GPU validation, kernel implementation, real DSpark calibration, and FP8
DeltaNet state work remain active and are not claimed complete.

Status: locally implemented; GPU validation pending.

Delivered so far:

- corrected local implementation assessment;
- pinned vLLM V1 source comparison;
- integrated design for indexed GDN branch state, lossless probabilistic
  rejection sampling, and W4A16 runtime repacking/kernel optimization;
- staged implementation and validation plan.
- indexed speculative GDN prefix-state slots with commit-by-remap and no target
  replay;
- temperature-aware MTP draft sampling, retained draft logits, probability
  ratio acceptance, and `(p-q)+` recovery sampling;
- one-time GPTQ `argsort(g_idx)` qweight repack plus fused activation
  permutation/direct-group runtime access;
- CPU correctness tests and deferred CUDA tests for all three paths.

No GPU correctness or performance claim has been validated. The PyTorch
rejection sampler is currently the correctness backend; a blockwise Triton
sampler and optional SM89 Marlin-style CUDA W4 backend remain benchmark-driven
follow-ups. The task stays active until RTX 4090D validation is complete.

The roadmap now uses SLO goodput as its primary objective and explicitly orders
W4A16 small/large-M, state-branch/rejection correctness, W4A8 large-M, FP8 KV,
FP8 DeltaNet state, and only then optional W3A16. Benchmark instrumentation can
be completed locally, but all GPU and capacity conclusions remain pending.

Status is now explicit: GDN state branches, the probability-difference
correctness sampler, GPTQ repack/Triton fallback, and FP8 KV functionality
already exist locally. They require GPU validation or capacity measurement;
they are not reimplementation milestones. The 8.8 GB BF16 DSpark draft is
offline-only on 24 GB, so the first online DSpark baseline requires INT4 draft
calibration and packing.
