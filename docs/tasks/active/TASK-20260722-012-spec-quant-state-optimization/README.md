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

Current gate: all authorized no-GPU implementation and static/CPU validation
is complete. RTX 4090D native-kernel compilation/correctness, CUDA Graph,
real DSpark conversion, capacity, and goodput validation remain explicitly
pending because the server is unavailable.

Records: research, design, plan, commands, tests, decisions, and result in this
directory.
