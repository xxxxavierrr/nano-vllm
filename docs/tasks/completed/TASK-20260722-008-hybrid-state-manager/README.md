# Hybrid state manager extraction

- Task: `TASK-20260722-008`
- Spec: [Runtime architecture](../../../specs/inference-runtime-architecture.md)
- Status: `superseded`
- Current gate: archived into TASK-20260722-012 for combined Graph/goodput work

## Goal

Move Qwen3.6 convolution/recurrent state slabs, sequence slots, working-state
copies, release/reset, and speculative commit into a dedicated rank-local
manager while preserving current capacity, MTP replay, and Graph behavior.

## Records

[Research](research.md) | [Design](design.md) | [Plan](plan.md) |
[Commands](commands.md) | [Tests](tests.md) | [Decisions](decisions.md) |
[Result](result.md)
