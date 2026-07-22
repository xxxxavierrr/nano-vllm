# Result

Status: superseded by TASK-20260722-012 on 2026-07-23.

Implemented: one packed API, one GDN custom-op boundary, fused projections,
runner-owned metadata, packed causal convolution without a Python request
loop, three DeltaNet recurrent/chunk kernels, explicit pre-cache chunk warmup,
and updated tests/benchmark call sites. Packed convolution uses two additional
private kernels because state must only be overwritten after every output has
read the previous state.

Server evidence includes the unfiltered GPU suite, Qwen3.6-27B GPTQ load and
generation, MTP k=2 branch-state execution with zero rejected-prefix target
replays, and online SSE smoke. These validate the refactored production
integration without creating a second GDN API.

Dedicated recurrent/chunk crossover measurements, semantic Full/Piecewise
per-region replay proof, and SLO goodput were not completed in this task. They
are explicitly retained in TASK-20260722-012; this archive does not claim them.
