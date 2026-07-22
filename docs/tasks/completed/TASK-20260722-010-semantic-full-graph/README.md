# TASK-20260722-010 semantic Full Graph

Status: superseded

Owning spec: `docs/specs/inference-runtime-architecture.md`

Goal: make Full CUDA Graph selection and capture use semantic uniform query
length plus request buckets, so fixed-k MTP verification does not silently
fall back to Piecewise. Keep eager fallback explicit and preserve prefill.

Current gate: archived into TASK-20260722-012; per-region replay proof remains
open there.

Records: `research.md`, `design.md`, `plan.md`, `commands.md`, `tests.md`,
`decisions.md`, and `result.md`.
