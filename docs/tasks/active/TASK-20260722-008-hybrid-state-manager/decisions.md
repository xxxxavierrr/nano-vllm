# Decisions

- Extract state ownership before physical KV Cache ownership.
- Preserve runner compatibility properties used by existing benchmarks.
- Keep rejected-prefix model replay in `ModelRunner`; the manager owns only
  state preparation/commit and lifecycle.
