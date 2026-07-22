# TASK-20260722-012 speculative, quantization, and state optimization

Status: active

Owning specs:

- `docs/specs/quantization.md`
- `docs/specs/speculative-decoding.md`
- `docs/specs/gated-deltanet.md`
- `docs/specs/benchmarking.md`
- `docs/specs/inference-runtime-architecture.md`
- `docs/specs/cuda-graphs.md`

Goal: maximize Qwen3.6 serving goodput under latency SLOs by removing W4,
speculative-state, and rejection-sampling waste before adding lower-precision
large-M, KV-cache, recurrent-state, or FFN formats.

Current gate: native W4A16 compiles on SM89 and passes numerical,
`torch.compile(fullgraph=True)`, and direct CUDA Graph replay tests. Packed-word
loading and specialized WMMA tiles reduced the prototype substantially, but
the best measured native result is still about 1.18x slower at M=1 and 1.20x
slower at M=512 than repacked Triton for K=N=5120, with larger gaps in the
middle. Production `auto` therefore remains Triton. A real Marlin follow-up
needs load-time Marlin layout plus asynchronous multi-stage global-to-shared
dataflow; W4A8 and end-to-end Piecewise evidence remain open.

Records: research, design, plan, commands, tests, decisions, and result in this
directory.
