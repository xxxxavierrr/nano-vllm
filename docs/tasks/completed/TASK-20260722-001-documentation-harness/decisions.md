# Decisions

- **Separate specs from task decomposition:** subject specs own durable
  contracts; independently identified tasks own execution and evidence. Many
  tasks may share one spec.
- **Physical active/completed directories:** lifecycle is visible in navigation
  and Git history; `docs/TASKS.md` is the dashboard.
- **Knowledge stays durable:** task-local debugging is not promoted unless it
  becomes reusable project truth.
- **Workflow directory plus discovery bridge:** `.workflow/README.md` is the
  requested contract; `AGENTS.md` ensures coding agents find it.
