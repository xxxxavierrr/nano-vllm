# Validation evidence

| Check | Status |
| --- | --- |
| Upstream revision and V1 entrypoints recorded | passed: pinned commit and source links in both specs |
| Quantization runtime flow traced | passed: config, layer method, loader post-process, GPTQ, and KV scale/cache boundaries |
| Speculative-decoding runtime flow traced | passed: scheduler, verification, sampler, post-process, proposal, and metrics |
| Obsolete V0 paths explicitly identified | passed: V0 workers/docs excluded in spec and knowledge |
| Local comparison grounded in current source | passed: `Config`, layers, scheduler, runner, tests, and benchmarks inspected |
| Specs and knowledge updated consistently | passed: subject specs and knowledge index updated |
| Runtime/GPU tests | not applicable: documentation-only source study |
| Links, task structure, and `git diff --check` | passed: validated before archival |
