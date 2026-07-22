# Speculator extraction

- Task: `TASK-20260722-009`
- Specs: [Runtime architecture](../../../specs/inference-runtime-architecture.md),
  [Speculative decoding](../../../specs/speculative-decoding.md)
- Status: `completed`
- Current gate: archived after Qwen3.6 MTP integration validation

## Goal

Extract MTP proposal input construction, recursive draft steps, cache metadata,
and greedy acceptance behind a reusable proposer/policy boundary without
changing scheduler or token semantics.

## Records

[Research](research.md) | [Design](design.md) | [Plan](plan.md) |
[Commands](commands.md) | [Tests](tests.md) | [Decisions](decisions.md) |
[Result](result.md)
