# Quantization engineering knowledge

Originating spec: [Model and KV-cache quantization](../specs/quantization.md)

## Stable distinctions

- Checkpoint format describes serialized tensors. A quantization config
  validates that format and chooses a per-layer runtime method. The runtime
  method creates parameters, performs one-time post-load transformation, and
  dispatches a kernel. These names may differ.
- Weight quantization and KV-cache quantization are independent. Weight dtype
  must never be used as a shortcut for cache dtype or cache capacity.
- Packed production parameters are allocated directly. Full dequantization is a
  test/reference operation, not a loading or inference step.
- Per-layer prefix dispatch is needed for fused projections, excluded modules,
  hybrid models, and future mixed quantization. Fallback must be explicit.
- Repacking, permutation, calibration, and hardware/layout validation belong to
  construction or post-load processing, never the request hot path.
- A custom CUDA symbol exposed only through pybind is not a compiler-safe
  runtime operator. Register schema/dispatch with `TORCH_LIBRARY`, provide a
  fake implementation, and call through `torch.ops` when Full/Piecewise Graph
  compatibility is required.

## nano-vLLM invariants

- GPTQ fused QKV/gate-up shards must have elementwise-identical `g_idx` before
  fusion. This intentionally tightens vLLM's assumption-based behavior.
- Raw `g_idx` is the W4A16 correctness baseline. A shuffled layout is accepted
  only after numerical and end-to-end benchmark evidence.
- FP8 KV per-token/per-head scales are produced at cache write and live exactly
  as long as their matching cache slots.
- Memory reporting separates weight payload/metadata, kernel workspace, graph
  memory, KV payload/scales, MTP cache, and recurrent/conv state.

## SM89 W4 evidence and optimization rule

- The opt-in native W4A16 path is numerically correct on RTX 4090D, supports
  non-monotonic `g_idx` through a fused activation permutation, and is safe
  under `torch.compile(fullgraph=True)` plus direct CUDA Graph replay.
- Load each INT32 packed word once and expand its eight INT4 values in
  registers. Re-reading the same packed word per K lane was the largest defect
  in the scalar prototype.
- WMMA tiling alone is not Marlin. On K=N=5120 the best tested native
  small/large prototypes still trailed repacked Triton; production `auto`
  remains Triton.
- A future Marlin implementation needs a load-time Marlin layout and
  asynchronous multi-stage global-to-shared/warp-specialized dataflow.
  Matching Triton block sizes without its pipeline regressed performance.
- Kernel choices remain shape-based. Small/large M describe GEMM geometry, not
  scheduler prefill/decode labels.

## Source baseline

The V1 study used vLLM commit
[`6e96891ba00d3d61a1eaa9c95bdd8d2663b183bd`](https://github.com/vllm-project/vllm/commit/6e96891ba00d3d61a1eaa9c95bdd8d2663b183bd).
The durable pattern is the config-to-layer-method lifecycle, not the breadth of
vLLM's quantization registry.
