# Design

`HybridStateManager` owns transient warmup state, persistent committed/working
slabs, request-to-slot mapping, free slots, reset/release, copy/commit, and peak
active count. `StateTransaction` scopes speculative working-state preparation
and accepted-sequence commit. Capacity calculation remains in the runner in
this task but calls one manager allocation method.
