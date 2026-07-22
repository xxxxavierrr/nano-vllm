# vLLM-style Qwen Gated DeltaNet refactor

- Task: `TASK-20260722-002`
- Spec: [Gated DeltaNet](../../../specs/gated-deltanet.md)
- Status: `active`
- Owner: Codex
- Last updated: 2026-07-22
- Current gate: apply source-study alignment before GPU validation

## Goal

Reshape Qwen3.5/3.6 Gated DeltaNet into fused input projections, one opaque
stateful GDN core custom op, then gated normalization and output projection.
Production, tests, and benchmarks must use one scheduler-oriented packed API.

## Current state

The local refactor passes static and CPU structural checks. CUDA numerical
correctness, real checkpoint loading, graph replay, online smoke, and
performance validation remain pending, so this task stays active.

## Acceptance gates

- [x] Remove public recurrent-only and chunk-only helper APIs.
- [x] Keep one formal `gated_delta_packed` entry.
- [x] Use one `torch.ops.nanovllm.qwen_gdn_core` compiler boundary.
- [x] Fuse QKV+Z and B+A projections with shard mapping.
- [x] Retain only three required numerical Triton kernels.
- [>] Apply source-study alignment before GPU validation.
- [ ] Run GPU correctness, loader, CUDA Graph, and state tests.
- [ ] Benchmark recurrent, chunk, mixed, and online workloads.
- [ ] Record the final result and archive.

## Records

[Research](research.md) | [Design](design.md) | [Plan](plan.md) |
[Commands](commands.md) | [Tests](tests.md) | [Decisions](decisions.md) |
[Result](result.md)
