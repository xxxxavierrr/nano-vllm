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

- Active optimization: [TASK-20260722-012 quant/spec/state optimization](tasks/active/TASK-20260722-012-spec-quant-state-optimization/README.md)
- Latest archive consolidation: TASK-002 and TASK-007 through TASK-011 moved
  under [completed tasks](tasks/completed/); unresolved Graph/GDN/goodput gates
  are explicitly retained by TASK-012.
- Latest completed implementation: [TASK-20260723-013 structural harness and refactor](tasks/completed/TASK-20260723-013-structural-harness-refactor/README.md)

This index must be updated whenever a task is archived or the project-level
architecture, interfaces, test strategy, or operational workflow changes.
