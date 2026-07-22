# Design

```text
.workflow/README.md             mandatory lifecycle
AGENTS.md                       discovery bridge
docs/README.md + TASKS.md       project indexes
docs/specs/                     canonical what and why
docs/tasks/active/<TASK-ID>/    live execution and evidence
docs/tasks/completed/<TASK-ID>/ archived history
docs/knowledge/                 durable cross-task knowledge
```

Each task has README, research, design, plan, commands, tests, decisions, and
result files. Their schema lives only in `.workflow/README.md`. `AGENTS.md`
directs agents to the workflow; the workflow requires
a spec/task before implementation and project-doc updates during archival.
