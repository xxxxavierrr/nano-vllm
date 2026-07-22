# Command log

Working directory: repository root. Local work will not commit, push, install,
or modify the Python environment.

- `python -m compileall -q nanovllm tests tools`: passed.
- Graph/MTP/context/state CPU regression excluding attention import: 31 passed.
- Including `test_piecewise_padding.py`: collection failed because the local
  environment has no `flash_attn`; this is recorded, not treated as a pass.

## 2026-07-23 server evidence and consolidation

- Qwen3-0.6B graph smoke reported one PIECEWISE prefill and seven FULL decode
  steps with zero EAGER fallback.
- Inductor still logged CPU-argument capture skips. Per-region replay proof and
  performance moved to TASK-20260722-012; this task is archived superseded.
