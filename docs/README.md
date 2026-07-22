# nano-vLLM engineering documentation

This directory is the persistent engineering memory for nano-vLLM. It keeps
requirements, execution state, design rationale, evidence, and reusable
knowledge outside transient chat history.

## Start here

- [Workflow](../.workflow/README.md): mandatory request lifecycle.
- [Task dashboard](TASKS.md): active and completed work.
- [Specifications](specs/README.md): canonical requirements and acceptance
  criteria.
- [Benchmarking](specs/benchmarking.md): goodput-first performance contract.
- [Knowledge](knowledge/README.md): durable project facts and practices.

## Current state

- Active: [TASK-20260722-002 GDN refactor](tasks/active/TASK-20260722-002-gdn-refactor/README.md)
- Active implementation: [TASK-20260722-007 typed-batch foundation](tasks/active/TASK-20260722-007-runtime-batch-foundation/README.md)
- Active implementation: [TASK-20260722-008 hybrid state manager](tasks/active/TASK-20260722-008-hybrid-state-manager/README.md)
- Active implementation: [TASK-20260722-009 speculator extraction](tasks/active/TASK-20260722-009-speculator-extraction/README.md)
- Active implementation: [TASK-20260722-010 semantic Full Graph](tasks/active/TASK-20260722-010-semantic-full-graph/README.md)
- Active implementation: [TASK-20260722-011 typed step metrics](tasks/active/TASK-20260722-011-typed-step-metrics/README.md)
- Active optimization: [TASK-20260722-012 quant/spec/state optimization](tasks/active/TASK-20260722-012-spec-quant-state-optimization/README.md)
- Latest completed: [TASK-20260722-006 Qwen3.6 runtime architecture](tasks/completed/TASK-20260722-006-qwen36-runtime-architecture/README.md)

This index must be updated whenever a task is archived or the project-level
architecture, interfaces, test strategy, or operational workflow changes.
