# Decisions

- Treat `forward` and the stateful custom-op boundary as semantic architecture;
  do not equate public API count with numerical kernel count.
- Put all scheduler-derived metadata in ModelRunner/backend code.
- Preserve private recurrent decode and chunk prefill strategies; do not force
  one kernel for different parallel computations.
- Keep nano-vLLM to one outer GDN custom op while it has one CUDA backend.
- Do not copy vLLM platform/LoRA/PP/registry complexity without supported use
  cases and tests.
- Require explicit prefill warmup and packed production convolution before the
  active GDN refactor proceeds to GPU validation.

