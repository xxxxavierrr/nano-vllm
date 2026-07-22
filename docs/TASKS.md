# Task dashboard

## Active

| Spec | Task | Status | Current gate |
| --- | --- | --- | --- |
| [Gated DeltaNet](specs/gated-deltanet.md) | [TASK-20260722-002 GDN refactor](tasks/active/TASK-20260722-002-gdn-refactor/README.md) | active | GPU correctness, loader, graph, and performance validation |
| [Runtime architecture](specs/inference-runtime-architecture.md) / [Gated DeltaNet](specs/gated-deltanet.md) | [TASK-20260722-007 typed-batch foundation](tasks/active/TASK-20260722-007-runtime-batch-foundation/README.md) | active | Dirty-diff inventory and compatibility-safe metadata migration |
| [Runtime architecture](specs/inference-runtime-architecture.md) | [TASK-20260722-008 hybrid state manager](tasks/active/TASK-20260722-008-hybrid-state-manager/README.md) | active | Extract GDN state lifetime and speculative transaction |
| [Runtime architecture](specs/inference-runtime-architecture.md) / [Speculative decoding](specs/speculative-decoding.md) | [TASK-20260722-009 speculator extraction](tasks/active/TASK-20260722-009-speculator-extraction/README.md) | active | Extract MTP proposal and acceptance interfaces |
| [Runtime architecture](specs/inference-runtime-architecture.md) / [CUDA Graphs](specs/cuda-graphs.md) | [TASK-20260722-010 semantic Full Graph](tasks/active/TASK-20260722-010-semantic-full-graph/README.md) | active | Implement semantic q=1/q=1+k capture keys and replay |
| [Runtime architecture](specs/inference-runtime-architecture.md) | [TASK-20260722-011 typed step metrics](tasks/active/TASK-20260722-011-typed-step-metrics/README.md) | active | Return metrics with rank-local model results |
| [Quantization](specs/quantization.md) / [Speculative decoding](specs/speculative-decoding.md) / [Gated DeltaNet](specs/gated-deltanet.md) / [Benchmarking](specs/benchmarking.md) | [TASK-20260722-012 quant/spec/state optimization](tasks/active/TASK-20260722-012-spec-quant-state-optimization/README.md) | active | Implement benchmark instrumentation locally; RTX 4090D correctness and SLO-goodput validation pending |

## Completed

| Spec | Task | Completed | Result |
| --- | --- | --- | --- |
| [Engineering workflow](specs/engineering-workflow.md) | [TASK-20260722-001 documentation harness](tasks/completed/TASK-20260722-001-documentation-harness/README.md) | 2026-07-22 | Repository workflow, indexes, task records, and knowledge structure added |
| [Gated DeltaNet](specs/gated-deltanet.md) | [TASK-20260722-003 vLLM source study](tasks/completed/TASK-20260722-003-vllm-gdn-source-study/README.md) | 2026-07-22 | Pinned upstream flow, code-style rules, and local alignment gaps documented |
| [Quantization](specs/quantization.md) / [Speculative decoding](specs/speculative-decoding.md) | [TASK-20260722-004 vLLM V1 study](tasks/completed/TASK-20260722-004-vllm-v1-quant-spec-study/README.md) | 2026-07-22 | Pinned V1 quant lifecycle and speculative scheduler/proposer protocol; obsolete V0 paths excluded |
| [CUDA Graphs](specs/cuda-graphs.md) | [TASK-20260722-005 vLLM V1 study](tasks/completed/TASK-20260722-005-vllm-v1-cudagraph-study/README.md) | 2026-07-22 | Pinned V1 policy, keys, capture ownership, backend capability, memory rules, and local gaps documented |
| [Inference runtime architecture](specs/inference-runtime-architecture.md) | [TASK-20260722-006 Qwen3.6 runtime architecture](tasks/completed/TASK-20260722-006-qwen36-runtime-architecture/README.md) | 2026-07-22 | Cross-capability ownership, target components, non-goals, and staged refactoring order documented |

## Rules

- Every active row must point to an existing active task directory and spec.
- Completed rows are append-only except for correcting factual errors.
- Moving a task between states requires updating this dashboard and
  `docs/README.md` in the same change.
