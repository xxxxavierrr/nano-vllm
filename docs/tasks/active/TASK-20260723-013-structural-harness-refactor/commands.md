# Commands

## 2026-07-23 local Windows CPU environment

- Read `.workflow/README.md`, task/spec indexes, the active optimization task,
  and runtime/benchmark/workflow specs: completed.
- Ran a read-only Python AST inventory across `bench.py`, `benchmarks/`,
  `nanovllm/`, and `tools/`: identified the large orchestration boundaries
  recorded in `research.md`.
- Inspected `git diff origin/main...HEAD`: working tree was clean before this
  task; nine local commits remain unpushed while the GPU server is unavailable.
- User refined the harness location: moved executable checks under
  `docs/hooks/` and made the baseline/strict commands mandatory in workflow.
- `.venv/Scripts/python.exe -m pytest tests/test_structure_harness.py -q`:
  `5 passed`.
- `.venv/Scripts/python.exe docs/hooks/check_structure.py --report-only`:
  completed and recorded required targets plus capped legacy debt; expected
  structural errors remain before refactoring.
- Focused benchmark regression: `10 passed` before the final SLO-denominator
  correction; final affected regression: `18 passed`.
- GPTQ/DSpark/loader regression: `26 passed, 1 CUDA skipped`.
- FP8 DeltaNet/capacity regression: `22 passed`.
- Speculative coordinator/MTP/state regression: `21 passed`.
- Combined local feature regression: `87 passed, 1 CUDA skipped`.
- `.venv/Scripts/python.exe docs/hooks/run_required_checks.py`: PASS for
  `git diff --check`, Python compilation, and strict structure policy.
- Full `pytest tests -q`: collection blocked by missing `flash_attn` in three
  files and a Windows GBK/PyTorch Inductor template decode error in sampler.
- Retried the remaining collection with UTF-8 and four blocked files excluded:
  timed out after 184 seconds with no final pytest result; not counted passed.
