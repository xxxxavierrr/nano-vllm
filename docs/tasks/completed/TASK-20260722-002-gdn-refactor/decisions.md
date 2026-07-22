# Decisions

## One custom-op boundary

Contain mutable state/backend dispatch in the GDN core while leaving projection
and output visible to `torch.compile`. Per-kernel public `triton_op` wrappers
are unnecessary.

## One formal packed API

Production, tests, and benchmarks use scheduler-style metadata. Removed helper
paths must not return as testing shortcuts.

## Three numerical kernels

API unification does not make recurrent scan and parallel chunk construction
the same computation. Indexed recurrent, chunk prepare, and chunk apply remain
internal kernels.

## Fused projections

Explicit loader offsets populate QKV+Z and B+A. GPTQ fusion requires strict
`g_idx` equality rather than silent overwrite.

