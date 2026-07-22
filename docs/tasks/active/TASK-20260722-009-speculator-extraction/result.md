# Result

Implementation is active.
# Result

Local implementation is complete. `MTPProposer` now owns proposal input
construction, paged-cache slot mapping, the first draft pass, and recursive
draft steps. `GreedyAcceptance` owns exact-prefix verification. `ModelRunner`
retains orchestration and state commit/replay but no longer contains a second
copy of the proposal implementation.

The task remains active because the local Windows environment cannot execute
the required Qwen3.6 CUDA, Graph, and performance gates.
