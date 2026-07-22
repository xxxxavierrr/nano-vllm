# Decisions

- Full buckets count requests; Piecewise buckets count scheduled tokens.
- Graph identity includes uniform query length.
- Only finite configured query lengths are captured at startup.
- Stateful Qwen3.6 remains limited to one Full request until server evidence
  proves a larger packed-state capture safe.
