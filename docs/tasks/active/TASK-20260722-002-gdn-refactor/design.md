# Design

```text
hidden states
  -> fused QKV+Z and B+A projections
  -> torch.ops.nanovllm.qwen_gdn_core
       -> scheduler metadata + convolution state
       -> gated_delta_packed
       -> recurrent state slab update
  -> gated RMS norm -> output projection
```

`gated_delta_packed` is the sole formal numerical entry. The custom op is the
single compiler boundary and owns mutation/backend dispatch.

All partitions, chunk indices, state slots, and actual/padded counts come from
`ModelRunner`. Production layer code does not rebuild metadata or loop over
requests. Tests and warmup construct the same context externally.

Three internal Triton kernels remain because they perform distinct parallel
work: indexed recurrent execution, chunk-summary preparation, and chunk-state
application. They are implementation details, not alternate public paths.

Runtime parameters fuse QKV+Z and B+A. Loader offsets handle BF16/FP8/GPTQ
logical shards; fused GPTQ `g_idx` values must match exactly.

## Risks

- Stateful mutation needs real CUDA Graph replay validation.
- The chunk/recurrent crossover is workload and hardware dependent.
- Local CPU checks cannot establish Triton numerical correctness.
- Piecewise capture currently doubles as chunk warmup; explicit warmup is
  needed so eager and Full-only configurations are also safe before sizing
  cache memory.
