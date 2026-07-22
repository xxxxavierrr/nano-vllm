# Command log

Local implementation only; no environment or Git mutation is planned.

- `python -m compileall -q nanovllm tests tools bench.py`: passed.
- Metrics/Graph/MTP/state/scheduler CPU regression: 53 passed.

## 2026-07-23 server validation

- Offline smoke JSON recorded EAGER/PIECEWISE/FULL modes and speculative
  branch/replay metrics; online EngineProc/SSE smoke shut down cleanly.
- Unfiltered GPU suite: `198 passed`.
