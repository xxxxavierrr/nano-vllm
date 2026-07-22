# Decisions

- Preserve all existing dirty runtime work and patch incrementally.
- Keep custom-op numerical signatures stable in Stage 1.
- Keep a scoped process-local forward context because model/custom ops need
  ambient per-call metadata; replace manual global reset, not the access model.
- Do not extract CacheManager, StateManager, Speculator, or GraphManager in this
  task; prepare their data contracts only.
