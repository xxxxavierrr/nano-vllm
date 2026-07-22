# vLLM V1 CUDA Graph source study

- Task: `TASK-20260722-005`
- Spec: [CUDA Graphs](../../../specs/cuda-graphs.md)
- Status: `completed`
- Current gate: none; documentation-only study archived

## Goal

Extract current vLLM V1 CUDA Graph semantics, compare them with nano-vLLM's
FULL_AND_PIECEWISE implementation, and amend the durable spec with verified
rules, obsolete-path exclusions, and concrete follow-up gaps.

The study is pinned to vLLM commit `6e96891ba00d3d61a1eaa9c95bdd8d2663b183bd`
from 2026-07-21. Stable V1 semantics and the current MRv2 graph manager are the
reference; legacy V0 runners were excluded.

## Records

[Research](research.md) | [Design](design.md) | [Plan](plan.md) |
[Commands](commands.md) | [Tests](tests.md) | [Decisions](decisions.md) |
[Result](result.md)
