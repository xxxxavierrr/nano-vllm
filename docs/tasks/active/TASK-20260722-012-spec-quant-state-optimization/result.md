# Result

Status: active.

Delivered and validated:

- framework-neutral schema-v3 goodput/SLO aggregation and offered-load sweep;
- indexed speculative DeltaNet branch state with zero rejected-prefix target
  replay in the Qwen3.6 MTP k=2 GPU smoke;
- lossless probability-ratio rejection and `(p-q)+` recovery correctness path;
- GPTQ load-time repack plus fused activation permutation and strict fused
  shard `g_idx` validation;
- FP8 KV runtime/capacity reporting and Qwen3 GPU smoke;
- local DSpark calibration/cache/GPTQ checkpoint tooling with synthetic
  production-loader round trip;
- FP8 DeltaNet state references/capacity/plumbing, still fail-closed pending
  fused GPU validation;
- native SM89 W4A16 build, numerical correctness, fullgraph compatibility,
  direct CUDA Graph replay, and raw kernel measurements.

Current W4 conclusion:

- loading each packed INT32 once and expanding eight INT4 values removed the
  largest scalar-prototype waste;
- the best retained WMMA specializations are small `16x64x128` and large
  `32x128x32` with two accumulators per warp;
- for K=N=5120 they still trail repacked Triton by about 1.18x at M=1 and
  1.20x at M=512, with larger mid-M gaps;
- `auto` therefore remains Triton, and native remains opt-in;
- a future native attempt must implement Marlin layout plus asynchronous
  multi-stage dataflow instead of further unguided tile tuning.

TASK-002 and TASK-007 through TASK-011 have been consolidated and archived.
Their remaining GDN/Graph/MTP goodput gates are now explicit plan items here;
no missing evidence was silently marked complete.

Still required before this task can archive: real INT4 DSpark calibration and
paired target baseline, combined Graph/state/sampler validation, W4A8 decision,
FP8 KV and DeltaNet-state offered-load goodput sweeps, complete 24 GB memory
ledger, and the final SLO-based benchmark report.
