# TASK-20260723-013 structural harness and refactor

Status: completed

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

The strict structure hook, full `198 passed` CPU/CUDA regression,
Qwen3/Qwen3.6 model smokes, MTP branch-state,
FP8 KV, offline benchmark, and online streaming gates pass on the RTX 4090D.

## Records

- [Research](research.md)
- [Design](design.md)
- [Plan](plan.md)
- [Commands](commands.md)
- [Tests](tests.md)
- [Decisions](decisions.md)
- [Result](result.md)
