# Command log

Run from the repository root on 2026-07-22. Secrets and routine inspections are
omitted.

```powershell
.venv\Scripts\python.exe -m compileall -q nanovllm\layers\deltanet.py nanovllm\layers\deltanet_chunk.py nanovllm\models\qwen3_5.py tests\test_deltanet.py tests\test_deltanet_chunk.py tests\test_deltanet_packed.py tools\bench_deltanet.py
```

Passed.

```powershell
.venv\Scripts\python.exe -m pytest -q tests\test_deltanet.py tests\test_deltanet_packed.py tests\test_deltanet_chunk.py
```

Collected but all 13 tests skipped because local CUDA is unavailable. This is
not a numerical pass.

Two local Python smoke scripts also checked the full-graph custom-op boundary
(output shape `(3, 64)`) and synthetic BF16 fused-loader offsets. Reusable GPU
versions still need to run on the server.

After the source-alignment pass:

```powershell
.venv\Scripts\python.exe -m compileall -q nanovllm tests tools
.venv\Scripts\python.exe -m pytest -q tests\test_deltanet.py tests\test_mtp_proposer.py tests\test_speculative_scheduler.py tests\test_forward_context.py tests\test_hybrid_state_manager.py tests\test_cudagraph_dispatcher.py
```

Compilation passed; the combined suite reported 30 passed and 5 CUDA skips.

## 2026-07-23 server validation and consolidation

- Reused TASK-20260723-013 GPU evidence: unfiltered suite `198 passed`;
  Qwen3.6-27B GPTQ eager and MTP k=2 completed; online SSE completed.
- Remaining GDN-specific Graph/performance gates were transferred to
  TASK-20260722-012 before archival. No dependency was changed.
