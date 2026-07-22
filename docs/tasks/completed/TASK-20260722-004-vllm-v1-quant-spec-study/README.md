# vLLM V1 quantization and speculative-decoding study

- Task: `TASK-20260722-004`
- Specs: [Quantization](../../../specs/quantization.md),
  [Speculative decoding](../../../specs/speculative-decoding.md)
- Status: `completed`
- Current gate: none; documentation-only study archived

## Goal

Extract current vLLM V1 architecture and code conventions for quantization and
speculative decoding, compare them with nano-vLLM, and amend the two durable
specs with evidence-backed rules and follow-up gaps.

The study is pinned to vLLM commit `6e96891ba00d3d61a1eaa9c95bdd8d2663b183bd`
from 2026-07-21. Both MRv1 and MRv2 exist at that revision; only their shared
V1 semantics and current MRv2 boundaries were promoted to project specs.

## Records

[Research](research.md) | [Design](design.md) | [Plan](plan.md) |
[Commands](commands.md) | [Tests](tests.md) | [Decisions](decisions.md) |
[Result](result.md)
