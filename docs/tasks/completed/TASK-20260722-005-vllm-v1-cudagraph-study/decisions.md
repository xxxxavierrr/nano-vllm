# Decisions

- Current V1 call sites and the current CUDA Graph design are authoritative;
  V0 graph runners are not nano-vLLM design references.
- Use stable V1 semantics rather than copy MRv1/MRv2 class structure.
- Keep `FULL_AND_PIECEWISE` as the default policy.
- Model uniformity with query length, not a decode Boolean, so `1+k` MTP
  verification can use Full Graph.
- Keep finite startup capture; never capture unseen traffic shapes lazily.
- Permit Inductor CUDAGraph Trees only with explicit captured-key/replay
  evidence.
- Represent GDN's current single-sequence Full limit as a declared capability
  restriction until GPU state-parity evidence supports expansion.
- Include Full Graph memory in capacity planning even if actual capture must
  follow cache allocation.
- This task changes documentation only. Runtime fixes and GPU evidence belong
  to a separate implementation task.
