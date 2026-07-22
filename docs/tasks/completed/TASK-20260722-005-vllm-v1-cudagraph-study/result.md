# Result

Completed a documentation-only study of current vLLM V1 CUDA Graph behavior at
commit `6e96891...`.

Delivered:

- Graph spec now defines policy/concrete modes, semantic keys, finite capture
  candidates, backend capability, startup/capture ownership, padding/state
  safety, memory accounting, observability, and feature interactions.
- Durable knowledge records the rules most likely to matter across future
  scheduler, MTP, GDN, quantization, and benchmark work.
- Local implementation was classified precisely: Full decode and Piecewise
  scaffolding exist, but MTP `1+k` misses Full, Piecewise replay is not proven,
  Full graph memory is not explicitly reserved, and state/graph evidence is
  incomplete.
- Existing runtime files were not changed and no GPU claim was made.

Follow-up implementation should first evolve the descriptor/key model and
observability, then add Full Graph MTP shapes and graph-memory reservation,
followed by CUDA state/capture/performance validation.
