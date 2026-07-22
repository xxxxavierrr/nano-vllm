# Plan

- [x] Add the workflow/spec rules, task records, checked-in structure policy,
  AST checker, and checker tests.
- [x] Refactor benchmark timing/SLO facts and aggregation; reduce `summarize`
  and the online/CLI orchestration it touches.
- [x] Refactor GPTQ quantization into focused Hessian, scale, propagation, and
  packing phases; review DSpark checkpoint functions against the serialization
  boundary without inventing another wrapper layer.
- [x] Split FP8 DeltaNet layout/reference/pool/runtime responsibilities while
  preserving its public imports.
- [x] Extract batch metadata builders and speculative-step coordination from
  `ModelRunner`; reduce the named runner hot paths.
- [x] Run structure, compile, CPU/Mock, quantization, state, scheduler, API, and
  benchmark regressions; record CUDA skips as pending.
- [x] Update local results and owning specs.
- [x] Validate runner metadata, MTP proposal/acceptance, DeltaNet state writes,
  FP8 attention dispatch, eager/Full/Piecewise Graph, offline JSON parity, and
  online smoke behavior on the RTX 4090D when the server returns.

## Structural targets

- `benchmarks.metrics.summarize <= 50` lines;
- `quantize_linear_gptq <= 40` lines;
- `ModelRunner.run <= 60` lines;
- `ModelRunner.prepare_inputs <= 60` lines after builder extraction;
- no new Python orchestration function over 80 lines;
- numerical-kernel exceptions remain explicit and do not cover wrappers.
