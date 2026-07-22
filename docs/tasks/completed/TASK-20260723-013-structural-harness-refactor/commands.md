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

## 2026-07-23 RTX 4090D GPU environment

- `git fetch origin && git pull --ff-only`: advanced server from `439182c` to user-pushed `303cb51`.
- Existing stack retained: Python 3.12, PyTorch `2.8.0+cu128`, Triton `3.4.0`, flash-attn `2.8.3.post1`; pytest installed only under `/tmp/nanovllm-pytest-303cb51`.
- `python docs/hooks/run_required_checks.py`: PASS before and after GPU fixes.
- Focused CPU/Mock suite: `98 passed`; initial focused CUDA kernel/state suite: `55 passed, 2 failed` at one GPTQ BF16-reference tolerance point.
- Full suite before GPU fixes: `194 passed, 2 failed`; diagnostic showed Triton versus FP32 max error `7.6e-06` while BF16 `F.linear` versus FP32 reached `0.0625`.
- GPTQ tests now require a strict FP32-accumulation match and retain a separate BF16 `F.linear` compatibility bound: `16 passed`.
- Final unfiltered full suite: `198 passed, 1 deprecation warning`.
- Qwen3-0.6B eager offline smoke: 2 requests, 16 output tokens, 8 EAGER steps; JSON `/tmp/nanovllm-eager-smoke.json`.
- Qwen3-0.6B graph smoke after capture-limit fix: 1 PIECEWISE prefill step and 7 FULL decode steps, zero EAGER; JSON `/tmp/nanovllm-full-piecewise-smoke-fixed.json`.
- Qwen3.6-27B GPTQ eager smoke: model loaded at 17033.67 MiB and completed; JSON `/tmp/qwen36-gptq-eager-smoke.json`.
- Qwen3.6-27B GPTQ MTP `k=2`: 3 verification rounds, 3 branch commits, zero rejected-prefix target replays; JSON `/tmp/qwen36-gptq-mtp-k2-smoke.json`.
- Sequential Qwen3-0.6B BF16/FP8 KV smokes: JSON `/tmp/qwen3-bf16-kv-accuracy.json` and `/tmp/qwen3-fp8-kv-accuracy.json`.
- Isolated online server on ports 18080/19080: `/health` ready and OpenAI-compatible SSE stream returned chunks; trap shut down the server and EngineProc.
