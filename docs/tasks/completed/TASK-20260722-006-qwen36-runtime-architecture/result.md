# Result

Completed a documentation-only cross-capability architecture assessment for
the Qwen3.6 runtime.

The target keeps `ModelRunner` as the rank-local facade and introduces only
ownership-bearing components: typed batch planning, physical cache/GDN state,
speculative state transaction, replaceable speculator/acceptance, capability-
aware Graph management, quantization-method lifecycle, and joint capacity
planning.

The recommended first implementation is not a broad runner split. First finish
the current GDN GPU baseline; then introduce typed batch metadata/context,
followed by cache/state ownership. MTP, Graph, and quantization extraction build
on those foundations in that order.

No runtime file was changed. The active GDN validation task remains the current
implementation gate.
