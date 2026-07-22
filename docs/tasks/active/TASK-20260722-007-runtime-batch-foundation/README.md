# Runtime typed-batch foundation

- Task: `TASK-20260722-007`
- Specs: [Runtime architecture](../../../specs/inference-runtime-architecture.md),
  [Gated DeltaNet](../../../specs/gated-deltanet.md)
- Status: `active`
- Current gate: CUDA attention/GDN/MTP/Graph validation on the server

## Goal

Implement Stage 1 of the Qwen3.6 runtime architecture: typed prepared-batch
metadata, semantic execution signatures, an exception-safe scoped forward
context, and runner-owned GDN metadata construction without changing numerical
kernels, scheduler results, serving APIs, or process topology.

## Records

[Research](research.md) | [Design](design.md) | [Plan](plan.md) |
[Commands](commands.md) | [Tests](tests.md) | [Decisions](decisions.md) |
[Result](result.md)
