---
subject: engineering-workflow
title: Repository documentation and task harness
status: active
created: 2026-07-22
updated: 2026-07-22
owner: Codex
---

# Repository documentation and task harness

## Motivation

Long-running inference work needs persistent requirements, design rationale,
command history, evidence, task state, and reusable knowledge outside chat.

## Requirements

- Maintain `docs/` with subject specs, active/completed tasks, knowledge, and
  project indexes.
- Specs are not decomposed by task. Every request creates or revises an owning
  subject spec, while each execution unit receives an independent `TASK-*`.
- Each task records goal, research, design, process, commands, tests, decisions,
  and result.
- `.workflow/README.md` is the single schema and lifecycle authority.
- Archiving updates overall docs and promotes durable knowledge.
- `AGENTS.md` makes the workflow discoverable.

## Scope

Repository engineering documentation and agent operating procedure.

## Non-goals

- Runtime or model behavior.
- Per-task template files that duplicate the workflow contract.

## Acceptance criteria

- Required directories, indexes, records, and workflow rules exist.
- Multiple tasks can link to one subject spec without duplicating it.
- Active and completed tasks contain complete execution evidence.

## Constraints and open questions

The structure must remain lightweight enough that records are updated during
real implementation rather than becoming a parallel bureaucracy.

## References

- GitHub Spec Kit: Spec -> Plan -> Tasks -> Implement lifecycle.
- Spec Kit persistence guidance for repository-owned artifacts.

## Change log

- 2026-07-22: Added the initial documentation harness.
- 2026-07-22: Removed templates; workflow became the single authority.
- 2026-07-22: Separated long-lived subject specs from independently identified
  execution tasks.

