# Test evidence

- Local OS: Windows
- Local PyTorch: `2.13.0+cpu`
- CUDA: unavailable locally

| Area | Validation | Status |
| --- | --- | --- |
| Syntax | compile changed Python files | passed |
| Packed convolution structure | model request loop removed; one packed API | passed locally |
| Packed convolution numerics | mixed lengths `1,7,19` vs `F.conv1d` reference | pending GPU |
| API | alternate public helpers removed | passed |
| Custom op | registration/fake behavior | passed locally |
| Dynamo | one boundary, `fullgraph=True`, eager backend | passed locally |
| BF16 loader | synthetic fused offsets | passed locally |
| Triton | recurrent/chunk/mixed reference comparison | pending GPU |
| State | continuation, reuse, padding, abort/preemption | pending GPU |
| GPTQ | real packed checkpoint and `g_idx` validation | pending GPU |
| Graph | capture/replay mutation correctness | pending GPU |
| Accuracy | greedy token agreement | pending GPU |
| Performance | kernel and online benchmark | pending GPU |

The later server suite and Qwen3.6 GPTQ/MTP integration replace the original
local skips for production-path correctness. Dedicated GDN kernel crossover
and per-region Graph proof remain unclaimed and moved to TASK-20260722-012.
