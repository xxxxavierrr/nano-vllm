# Plan

## Completed local foundations

- [x] Review and approve the cross-capability design and memory/correctness
  contracts.
- [x] Implement `StateBranchTable`, speculative slot capacity accounting, and
  prefix-state writes for convolution and recurrent GDN state.
- [x] Remove rejected-prefix target-model replay and committed/working slab
  copies from the production speculative path.
- [x] Add typed draft proposals and return MTP draft logits.
- [x] Implement deterministic greedy and lossless probabilistic rejection
  policies, then integrate temperature sampling.
- [x] Add GPTQ post-load runtime repack and a repacked-layout Triton path;
  retain the raw-layout implementation as fallback/reference.
- [x] Implement the FP8 KV runtime path, capacity accounting, comparison tool,
  and attention microbenchmark entry points.

## Remaining milestones in execution order

- [x] Extend the framework-neutral benchmark schema and aggregation with
  accepted tokens/s, SLO-good output-token throughput, time-weighted running
  requests, scheduled actual/padded tokens, and GPU telemetry provenance; add
  an open-loop offered-load sweep for maximum SLO throughput.
- [x] Add an opt-in native SM89 W4 extension, small/large-M W4A16 dispatcher,
  experimental large-M W4A8 source, layout validation, and safe Triton
  fallback. CUDA compilation/correctness/performance remain a later GPU gate.
- [ ] Validate and optimize the native W4A16/W4A8 implementation on SM89,
  including tensor-core dataflow, Graph behavior, and target shapes.
- [>] Restore the RTX 4090D server gate, then begin native W4 compilation and
  correctness validation. No GPU result is inferred from local source/tests.
- [x] Implement the local DSpark calibration shell, resumable sharded cache,
  strict streaming weight mapping, FP32-Hessian GPTQ quantizer, dry-run memory
  projection, and synthetic production-loader checkpoint round-trip.
- [ ] Build the real INT4 DSpark checkpoint by persisting target-produced draft
  inputs, then loading the BF16 draft alone for calibration/reference. Never
  plan an online BF16-draft-plus-target cell on the 24 GB GPU.
- [ ] Record the first runnable paired-AWQ-target + INT4-draft baseline,
  including kernel shapes, speculative acceptance, state/KV memory, TTFT/TPOT
  p50/p99, goodput, scheduler occupancy, utilization, and stable concurrency.
- [x] Strengthen local zero-replay/state-prefix, probabilistic sampler property,
  and unified FP8 KV capacity-report coverage before adding FP8 DeltaNet state.
- [ ] Validate the already implemented DeltaNet branch-state path and prove
  zero rejected-prefix target replay under eager, Full, and Piecewise Graph.
- [ ] Validate the already implemented probability-difference sampler on GPU;
  add a blockwise fused implementation only after the correctness path passes.
- [ ] Validate combined W4A16 + MTP/DSpark + branch state + probabilistic
  sampler + Full/Piecewise Graph behavior on the RTX 4090D.
- [ ] Implement and evaluate fused W4A8 for large-M only; retain W4A16 for
  small-M unless end-to-end evidence selects otherwise.
- [ ] Measure native versus the existing FP8 KV path across context length,
  concurrency, and offered load; decide from maximum SLO goodput rather than
  equal-concurrency latency.
- [x] Implement the local FP8 DeltaNet committed/branch state configuration,
  quantization references, capacity model, lifecycle integration, and disabled
  Triton load/store source. GPU enablement and goodput sweep remain pending.
- [ ] Sweep FP8 DeltaNet committed/branch state and decide from
  effective scheduler batch, numerical correctness, and SLO goodput.
- [ ] Recompute the complete 24 GB memory budget. Consider FFN W3A16 only if
  memory still prevents the best useful concurrency after prior phases.
- [ ] Benchmark, document results, and archive this task only after GPU evidence.

## Commit boundaries

1. framework-neutral goodput instrumentation and CPU aggregation tests;
2. Marlin-style W4A16 small-M/large-M backend;
3. INT4 DSpark calibration/checkpoint and first runnable baseline;
4. GPU validation fixes for existing GDN branch-state and rejection sampling;
5. fused W4A8 large-M backend;
6. measurement of the existing FP8 KV runtime;
7. optional FP8 DeltaNet state after its capacity model is established;
8. optional FFN W3A16 only after the capacity gate;
9. combined benchmark/report updates.

While the server is unavailable, the user explicitly permits local commits but
not push. When the server returns, local commits stop and GPU development again
becomes the only commit/push source.
