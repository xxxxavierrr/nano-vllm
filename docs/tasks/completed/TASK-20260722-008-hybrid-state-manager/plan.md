# Plan and process

- [x] Extract state/slab/slot lifecycle and compatibility properties.
- [x] Route prepared GDN metadata, allocation, abort/preemption, and cleanup
  through the manager.
- [x] Add speculative state transaction and migrate commit/replay ordering.
- [x] Add CPU lifecycle tests with a fake model and run regressions.
- [x] Validate persistent mutation and zero rejected-prefix replay on the CUDA
  server.
- [!] Transfer semantic Graph dummy-slot proof and capacity/goodput sweeps to
  TASK-20260722-012.
