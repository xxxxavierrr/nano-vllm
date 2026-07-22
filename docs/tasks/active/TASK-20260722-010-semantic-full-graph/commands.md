# Command log

Working directory: repository root. Local work will not commit, push, install,
or modify the Python environment.

- `python -m compileall -q nanovllm tests tools`: passed.
- Graph/MTP/context/state CPU regression excluding attention import: 31 passed.
- Including `test_piecewise_padding.py`: collection failed because the local
  environment has no `flash_attn`; this is recorded, not treated as a pass.
