# Design implications

## Adopted boundary

Keep a small engine-owned Graph subsystem:

```text
scheduler output
  -> semantic batch descriptor
  -> capability-aware candidate dispatcher
  -> stable-input preparation
  -> FULL / PIECEWISE / EAGER runner
  -> capture/replay and fallback metrics
```

The dispatcher is the only owner of available keys. Model and operator code
declare capabilities and stable-buffer needs; they do not independently choose
execution mode.

## Descriptor evolution

Replace `uniform_decode: bool` with `uniform_query_len: int | None`. A Full key
must distinguish request count and uniform query length from total tokens. This
directly enables fixed-`k` MTP verification without creating request-type
branches.

## Piecewise choice

nano-vLLM may retain Inductor CUDAGraph Trees rather than copy vLLM wrappers,
but only behind an adapter that records which finite sizes were successfully
compiled/captured and emits replay evidence. If PyTorch cannot provide reliable
capture/replay evidence for this path, use explicit Piecewise wrappers.

## Capacity choice

Piecewise preparation already occurs before KV allocation. Add a measured or
conservative Full Graph reserve to the same capacity plan, then report actual
capture memory. Avoid solving this by reducing the user-configured utilization
silently.

## Compatibility choices

- Keep GDN Full restricted to the numerically proven key set until batched
  state parity is established.
- Treat W4A16, FP8 KV, MTP, and GDN as independent capability inputs to the
  dispatcher/capture plan.
- Do not add LoRA or distributed key dimensions until the corresponding
  feature is implemented.
- Preserve explicit startup failure for requested modes whose preparation
  fails; make capability-driven policy resolution visible but deterministic.

## Risks

- Token-only graph keys can replay structurally different speculative batches.
- Dummy GDN state slots can corrupt reusable state even when final tokens match
  for a short sample.
- Relying on compiler heuristics can report Piecewise mode without actual graph
  replay.
- Allocating Full graphs after a maximal cache can cause avoidable startup OOM
  or consume undocumented memory headroom.
