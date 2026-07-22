# Plan and process

- [x] Extract state/slab/slot lifecycle and compatibility properties.
- [x] Route prepared GDN metadata, allocation, abort/preemption, and cleanup
  through the manager.
- [x] Add speculative state transaction and migrate commit/replay ordering.
- [x] Add CPU lifecycle tests with a fake model and run regressions.
- [>] Validate persistent mutation, rejected-prefix replay, and Graph dummy
  state slots on the CUDA server.
