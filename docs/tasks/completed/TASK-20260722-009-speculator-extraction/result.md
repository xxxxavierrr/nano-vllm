# Result

Local implementation is complete. `MTPProposer` now owns proposal input
construction, paged-cache slot mapping, the first draft pass, and recursive
draft steps. `GreedyAcceptance` owns exact-prefix verification. `ModelRunner`
retains orchestration and state commit/replay but no longer contains a second
copy of the proposal implementation.

Status: completed on 2026-07-23.

Qwen3.6-27B GPTQ MTP k=2 ran on RTX 4090D for three verification rounds with
typed proposal/acceptance metrics and zero rejected-prefix target replay. The
extraction boundary is complete. Broader k/acceptance/goodput and DSpark work
is optimization scope retained in TASK-20260722-012.
