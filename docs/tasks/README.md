# Task records

Task records are resumable execution packages linked to canonical specs.

- `active/`: work still being implemented, validated, or externally blocked.
- `completed/`: immutable execution history for delivered, cancelled, or
  superseded work.

Each task directory must contain:

```text
README.md      goal, scope, status, current state, links
research.md    external references and repository findings
design.md      architecture, interfaces, alternatives, risks
plan.md        ordered work, dependencies, live checkboxes
commands.md    meaningful command and environment log
tests.md       planned/actual validation and evidence
decisions.md   material decisions and rationale
result.md      delivered outcome, limitations, follow-ups
```

Specs are stored by subject under `docs/specs/` and linked from task READMEs.
Many tasks may implement or investigate the same spec; task decomposition never
creates duplicate specs.
