# Plan and process

- [x] Inspect dirty diffs, context call sites, model metadata fallbacks, and
  current tests before editing.
- [x] Add typed metadata, prepared batch, semantic execution signature, and
  scoped forward context with compatibility tests.
- [x] Move GDN metadata construction to runner-side preparation and remove the
  model fallback builder.
- [x] Migrate warmup, target, MTP, replay, Piecewise, and Full paths without
  changing numerical kernels.
- [x] Run static and CPU tests; record unavailable GPU gates accurately.
- [>] Validate attention, GDN numerics/state, MTP, and Full/Piecewise Graph on
  the CUDA server before archival.
