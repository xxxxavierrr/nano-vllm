# Design implications

## Adopt

- Short semantic model forward.
- One explicit stateful compiler boundary with declared mutation.
- Runner-owned typed metadata and stable Graph buffers.
- Private recurrent/chunk fast paths behind one semantic core.
- Constructor-time backend choice and pre-cache kernel warmup.
- Declarative checkpoint-to-runtime fusion.

## Keep simpler than vLLM

nano-vLLM currently needs one CUDA backend, TP=1 GDN, and no LoRA/PP/platform
registry. It should not reproduce vLLM's pluggable-layer, multi-platform, or
multiple chunk-backend wrappers until those features are in scope.

## Follow-up design changes

Move fallback metadata construction out of the model, replace production
per-request convolution looping with packed execution, and make GDN prefill
warmup independent of Graph mode.

