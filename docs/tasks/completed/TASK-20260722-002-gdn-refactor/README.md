# vLLM-style Qwen Gated DeltaNet refactor

- Task: `TASK-20260722-002`
- Spec: [Gated DeltaNet](../../../specs/gated-deltanet.md)
- Status: `superseded`
- Owner: Codex
- Last updated: 2026-07-23
- Current gate: archived into TASK-20260722-012 for remaining end-to-end Graph
  and goodput optimization

## Goal

Reshape Qwen3.5/3.6 Gated DeltaNet into fused input projections, one opaque
stateful GDN core custom op, then gated normalization and output projection.
Production, tests, and benchmarks must use one scheduler-oriented packed API.

## Current state

The refactor is implemented and exercised by the RTX 4090D full suite,
Qwen3.6-27B GPTQ/MTP smoke, and online smoke. Dedicated GDN crossover,
per-region Graph proof, and SLO goodput remain optimization work and are owned
by TASK-20260722-012.

## Acceptance gates

- [x] Remove public recurrent-only and chunk-only helper APIs.
- [x] Keep one formal `gated_delta_packed` entry.
- [x] Use one `torch.ops.nanovllm.qwen_gdn_core` compiler boundary.
- [x] Fuse QKV+Z and B+A projections with shard mapping.
- [x] Retain only three required numerical Triton kernels.
- [x] Apply source-study alignment before GPU validation.
- [x] Run GPU correctness, loader, and state integration tests.
- [!] Dedicated Graph replay and recurrent/chunk performance evidence moved to
  TASK-20260722-012.
- [x] Record the result and archive as superseded.

## Records

[Research](research.md) | [Design](design.md) | [Plan](plan.md) |
[Commands](commands.md) | [Tests](tests.md) | [Decisions](decisions.md) |
[Result](result.md)
