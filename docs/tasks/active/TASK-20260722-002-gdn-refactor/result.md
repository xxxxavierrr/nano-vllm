# Result

Status: active. The code refactor is locally implemented, but mandatory GPU
validation is incomplete.

Implemented: one packed API, one GDN custom-op boundary, fused projections,
runner-owned metadata, packed causal convolution without a Python request
loop, three DeltaNet recurrent/chunk kernels, explicit pre-cache chunk warmup,
and updated tests/benchmark call sites. Packed convolution uses two additional
private kernels because state must only be overwritten after every output has
read the previous state.

Before archive: CUDA numerics/state tests, real BF16/GPTQ loading, Full and
Piecewise Graph validation, greedy output comparison, and performance results.
