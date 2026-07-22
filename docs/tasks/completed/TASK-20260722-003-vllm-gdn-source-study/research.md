# Research

## Upstream revision and primary sources

- Revision: [`6e96891ba00d3d61a1eaa9c95bdd8d2663b183bd`](https://github.com/vllm-project/vllm/commit/6e96891ba00d3d61a1eaa9c95bdd8d2663b183bd)
- [Qwen GDN layer/core](https://github.com/vllm-project/vllm/blob/6e96891ba00d3d61a1eaa9c95bdd8d2663b183bd/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py)
- [GDN metadata builder](https://github.com/vllm-project/vllm/blob/6e96891ba00d3d61a1eaa9c95bdd8d2663b183bd/vllm/v1/attention/backends/gdn_attn.py)
- [GDN base](https://github.com/vllm-project/vllm/blob/6e96891ba00d3d61a1eaa9c95bdd8d2663b183bd/vllm/model_executor/layers/mamba/gdn/base.py)
- [Qwen3.5 integration](https://github.com/vllm-project/vllm/blob/6e96891ba00d3d61a1eaa9c95bdd8d2663b183bd/vllm/model_executor/models/qwen3_5.py)
- [FLA chunk orchestration](https://github.com/vllm-project/vllm/blob/6e96891ba00d3d61a1eaa9c95bdd8d2663b183bd/vllm/third_party/flash_linear_attention/ops/chunk.py)

## Source trace

- Qwen3.5 chooses the shared GDN layer and maps separately stored checkpoint
  projections into fused runtime parameters.
- The attention backend, not the layer, builds decode/prefill/spec partitions,
  sequence offsets, state indices, chunk metadata, accepted counts, initial
  state masks, and graph-stable request buffers.
- Layer forward is projection -> mutating core custom op -> norm/output.
- The core owns convolution and recurrent-state mutation. It consumes metadata
  from forward context and trims padded input to `num_actual_tokens`.
- Decode-only uses a private packed recurrent path. General mixed execution can
  run speculative recurrent, ordinary recurrent decode, and chunk prefill,
  then restore token ordering.
- Backend selection is constructor-time. FLA/FlashInfer/CuteDSL share a
  semantic chunk interface while their numerical implementations differ.
- Chunk metadata is precomputed to avoid per-layer host sync. Autotuned prefill
  kernels are warmed before cache allocation.

## Local comparison

The local three-part forward, zero output buffer, state slabs, production
ModelRunner partitioning, Graph slots, and fused loader mapping align well.

Local gaps are layer-local `_build_packed_metadata`, transient request
orchestration, a non-uniform convolution Python loop, and the lack of explicit
chunk warmup when Piecewise capture is disabled.

