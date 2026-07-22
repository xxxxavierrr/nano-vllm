# TASK-20260723-013 structural harness and refactor

Status: active

Owning specs:

- [Engineering workflow](../../../specs/engineering-workflow.md)
- [Inference runtime architecture](../../../specs/inference-runtime-architecture.md)
- [Benchmarking](../../../specs/benchmarking.md)
- [Quantization](../../../specs/quantization.md)

## Goal

Turn the repository's vLLM-inspired ownership rules into an executable review
gate, then refactor recently added or expanded Python orchestration so feature
work cannot continue accumulating inside `ModelRunner`, benchmark aggregation,
calibration orchestration, or mixed reference/runtime modules.

## Current gate

The strict structure hook and local CPU/Mock suite pass. GPU numerical,
Full/Piecewise Graph, online, and performance equivalence remain pending until
the RTX 4090D server returns; no local result is promoted to GPU evidence.

## Records

- [Research](research.md)
- [Design](design.md)
- [Plan](plan.md)
- [Commands](commands.md)
- [Tests](tests.md)
- [Decisions](decisions.md)
- [Result](result.md)
