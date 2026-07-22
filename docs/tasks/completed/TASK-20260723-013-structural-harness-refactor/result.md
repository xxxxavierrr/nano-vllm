# Result

Status: completed on the RTX 4090D GPU server.

Delivered:

- mandatory pre-change and strict pre-commit hooks under `docs/hooks/`;
- single-source benchmark timing/SLO facts and a separated in-process runner,
  result builder, and reporter;
- phased GPTQ calibration helpers;
- separate FP8 DeltaNet layout, reference codec, pool, and experimental kernel
  modules with compatibility imports;
- a focused `BatchPlanner` and speculative-step coordinator;
- a short `ModelRunner` facade and phase-oriented MTP proposer;
- validation/launch separation for packed DeltaNet convolution and FP8 paged
  attention wrappers.

GPU evidence is an unfiltered `198 passed`, plus a passing strict hook and
regression tests for the runtime defects found only by model-level smoke.

GPU acceptance includes:

- Qwen3-0.6B eager and FULL/PIECEWISE offline execution;
- Qwen3.6-27B GPTQ eager and native MTP `k=2` branch-state execution;
- BF16 and FP8 KV engine execution and capacity reporting;
- online health and OpenAI-compatible streaming output with clean shutdown.

GPTQ CUDA correctness now uses strict FP32 accumulation plus a separate BF16 compatibility baseline.
The semantic CUDA Graph task still owns per-region capture/replay proof because Piecewise dispatch does
not prove that Inductor captured every region when CPU arguments trigger skip messages.
