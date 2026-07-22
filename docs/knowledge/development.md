# Development practices

- Preserve unrelated dirty-worktree changes.
- Record the intended execution host in each task. CUDA correctness and
  performance claims require the GPU environment; local syntax/collection is
  not a substitute.
- Never store access tokens or credentials in specs, task logs, shell history
  excerpts, or committed files.
- Keep public behavior, architecture decisions, and benchmark methodology in
  docs; do not rely on commit messages or chat history as the only record.
- New work follows [the repository workflow](../../.workflow/README.md).
- When the GPU server is available, it is the sole implementation/commit/push
  source for this project; the local Windows checkout only pulls. Temporary
  build and profiler artifacts stay outside the repository and are not
  committed.
