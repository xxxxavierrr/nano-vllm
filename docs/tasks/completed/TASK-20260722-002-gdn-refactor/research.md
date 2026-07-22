# Research

## References

- [vLLM Qwen GDN linear attention](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py)
- [vLLM FLA chunk operator](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fla/ops/chunk.py)
- [Pinned upstream study](../../../specs/gated-deltanet.md#upstream-vllm-source-study)

## Findings

- The old surface mixed production, recurrent-only, chunk-only, and packed
  variants, allowing tests to bypass scheduler semantics.
- Persistent convolution and recurrent state make the GDN core the correct
  compiler boundary; the whole layer and individual kernel launches are not.
- Decode/short and long-prefill need different numerical kernels but can share
  one packed API and metadata contract.
- Fused GPTQ shards need explicit offsets and elementwise `g_idx` validation.
- vLLM separates projection, one custom attention-core op, and output
  projection, keeping ordinary linear/norm work visible to Dynamo.
- vLLM constructs GDN request/chunk/state metadata outside the layer. The local
  layer fallback builder belongs in runner/test infrastructure.
- Recurrent decode and chunk prefill are private fast paths behind one semantic
  core; mixed scheduling does not imply one numerical kernel.
- Chunk autotuning is explicitly warmed before cache allocation.
