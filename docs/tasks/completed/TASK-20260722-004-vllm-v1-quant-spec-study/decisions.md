# Decisions

- Current vLLM V1 call sites are authoritative.
- V0 `SpecDecodeWorker`, `MultiStepWorker`, and legacy worker orchestration are
  not design references for nano-vLLM.
- Quantization plugin breadth is separated from core configuration/loading
  contracts.
- This task changes documentation only; implementation follows separate tasks.
- Pin upstream evidence to a commit. Latest developer-preview docs can describe
  code newer than a stable release and are not sufficient as the only source.
- Treat the coexistence of MRv1 and MRv2 as a migration detail. Adopt the shared
  V1 scheduler semantics and current MRv2 proposer/sampler boundary.
- Keep nano-vLLM's stricter fused GPTQ `g_idx` validation even though upstream
  historically assumes fused shard metadata matches.
- Keep MTP greedy-only until a probabilistic rejection sampler and
  distributional tests are implemented; do not generalize the accuracy claim.
