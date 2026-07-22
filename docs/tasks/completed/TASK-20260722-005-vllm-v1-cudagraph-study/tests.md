# Validation evidence

| Check | Status |
| --- | --- |
| Upstream revision and current V1 entrypoints recorded | passed: pinned commit and direct source links in spec/research |
| Modes, keys, buckets, and dispatch traced | passed: policy/concrete modes, finite candidates, and priority documented |
| Warmup/capture/replay/memory ordering traced | passed: runner/wrapper ownership and capacity reservation documented |
| Quantization/speculative/stateful interactions traced | passed: W4/FP8, `1+k` verification, and GDN capability/state rules documented |
| Obsolete paths explicitly excluded | passed: V0, MRv1-only rounding, and unsupported LoRA dimensions excluded |
| Local comparison grounded in current source | passed: dispatcher, runner, cache allocation, tests, and benchmark metrics inspected |
| Runtime/GPU tests | not applicable: documentation-only source study |
| Spec, knowledge, links, and task structure | passed: validated before archival |
