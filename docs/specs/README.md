# Specifications

Specs are long-lived capability and module contracts. They are not divided by
task: multiple implementation, research, benchmark, and validation tasks may
all execute against and amend the same spec.

Every request must create or revise its owning spec before an execution task is
opened. Material changes go in the spec change log. Task status, commands, and
test evidence belong only under `docs/tasks/`.

The schema and lifecycle are defined in
[the repository workflow](../../.workflow/README.md).

## Index

- [Gated DeltaNet](gated-deltanet.md) - active capability work
- [Quantization](quantization.md) - weights and KV-cache quantization
- [Speculative decoding](speculative-decoding.md) - proposal and verification
- [CUDA Graphs](cuda-graphs.md) - compilation, capture, and dispatch
- [Inference runtime architecture](inference-runtime-architecture.md) -
  Qwen3.6 cross-capability ownership and extension boundaries
- [Benchmarking](benchmarking.md) - goodput, SLO, capacity, and telemetry
- [Engineering workflow](engineering-workflow.md) - repository contract
