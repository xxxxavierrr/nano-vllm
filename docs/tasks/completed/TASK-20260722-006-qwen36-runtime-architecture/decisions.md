# Decisions

- Keep `LLMEngine -> ModelRunner`; do not split EngineCore/LocalExecutor now.
- Make typed prepared-batch metadata the first shared foundation.
- Centralize physical cache/GDN state and speculative commit/rollback before
  extracting the proposer or Graph manager.
- Keep the current generic scheduler speculative protocol.
- Keep one GDN semantic custom op and private recurrent/chunk kernels.
- Treat proposer precision as explicit; do not implicitly leave MTP BF16.
- Keep DP in the serving replica router and defer a rank executor abstraction
  until Qwen3.6 TP is implemented.
- Delay quantization lifecycle extraction until state/Graph boundaries stabilize
  because current W4A16 is functional and numerically sensitive.
- Use a small method map, not a vLLM-sized plugin registry.
- This assessment changes documentation only and does not claim GPU validation.
