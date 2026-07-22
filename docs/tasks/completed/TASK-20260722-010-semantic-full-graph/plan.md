# Plan

- [x] Add semantic Full keys and request bucketing to the dispatcher.
- [x] Capture and replay finite q=1 / q=1+k graphs with correct metadata sizes.
- [x] Add CPU dispatcher/key tests and static regression.
- [x] Record GPU FULL/PIECEWISE dispatch and parity smoke evidence.
- [!] Transfer per-region captured-key/replay proof and performance evidence to
  TASK-20260722-012 because Inductor still reports CPU-argument graph skips.
