# TASK-20260722-012 speculative, quantization, and state optimization

Status: active

Owning specs:

- `docs/specs/quantization.md`
- `docs/specs/speculative-decoding.md`
- `docs/specs/gated-deltanet.md`
- `docs/specs/benchmarking.md`

Goal: maximize Qwen3.6 serving goodput under latency SLOs by removing W4,
speculative-state, and rejection-sampling waste before adding lower-precision
large-M, KV-cache, recurrent-state, or FFN formats.

Current gate: complete goodput/SLO benchmark instrumentation and static/CPU
validation locally. RTX 4090D kernel correctness, CUDA Graph, capacity, and
performance validation remain explicitly pending because the server is
unavailable.

Records: research, design, plan, commands, tests, decisions, and result in this
directory.
