# CUDA Graph engineering rules

See the owning [CUDA Graph specification](../specs/cuda-graphs.md).

## Stable rules

- CUDA Graph is an engine execution policy, not an incidental side effect of
  calling `torch.compile`.
- Public dual modes resolve to the concrete runtime modes `FULL`, `PIECEWISE`,
  and `EAGER`; one dispatcher is the only source of truth.
- Dispatch prioritizes the most complete compatible capture:
  `FULL -> PIECEWISE -> EAGER`.
- A Full key needs token count, request count, and uniform query length.
  `uniform_query_len=1+k` is the important speculative-decoding case.
- Capture keys are finite and prepared before readiness. Runtime pads to the
  smallest compatible key and never captures a new key on traffic.
- Compile sizes and capture sizes are separate even if their defaults overlap.
- Hybrid models use the minimum declared graph capability across attention and
  stateful backends.
- Warmup/autotune uses eager execution; capture is a separate explicit step.
- Stable input and metadata buffers belong to the runner. Capture/replay
  wrappers do not own scheduler semantics.
- Dummy rows never write cache/state or enter logits, sampling, acceptance, or
  metrics.
- Persistent graph pools, stable buffers, and workspaces are part of cache
  capacity planning.
- A mode label does not prove replay. Evidence needs captured keys and replay
  counters or equivalent profiler/framework output.

## nano-vLLM watch points

- Do not model uniformity as `num_scheduled_tokens == 1`; fixed-`k` MTP target
  verification is a uniform batch with query length `1+k`.
- Do not use only padded token count as a Full key; the same token count may
  represent different request and query-length layouts.
- `torch.compile(mode="reduce-overhead")` plus two startup calls does not by
  itself establish that every advertised Piecewise key was captured.
- Piecewise compiler/graph memory is currently present before KV allocation,
  but Full graphs are captured afterward. Reserve Full graph memory explicitly
  before final KV/DeltaNet capacity.
- Gated DeltaNet padding requires inert state slots or masking and numerical
  state comparison, not only final-token comparison.

## Source basis

These rules were checked against current vLLM V1 at commit
[`6e96891ba00d3d61a1eaa9c95bdd8d2663b183bd`](https://github.com/vllm-project/vllm/commit/6e96891ba00d3d61a1eaa9c95bdd8d2663b183bd),
especially its [CUDA Graph design](https://github.com/vllm-project/vllm/blob/6e96891ba00d3d61a1eaa9c95bdd8d2663b183bd/docs/design/cuda_graphs.md)
and [MRv2 graph manager](https://github.com/vllm-project/vllm/blob/6e96891ba00d3d61a1eaa9c95bdd8d2663b183bd/vllm/v1/worker/gpu/cudagraph_utils.py).
