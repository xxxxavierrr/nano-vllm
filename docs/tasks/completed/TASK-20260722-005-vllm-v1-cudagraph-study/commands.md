# Command log

Working directory: repository root. No credentials or private URLs were
recorded.

| Command/action | Environment | Outcome |
| --- | --- | --- |
| Read workflow, indexes, current Graph spec, and active task | local Windows | Confirmed mandatory documentation lifecycle and preserved unrelated dirty runtime files |
| `git status --short` | local Windows | Recorded existing GDN/runtime changes and untracked `uv.lock`; none were modified by this study |
| Open current official vLLM CUDA Graph, compilation, and MRv2 docs | web, 2026-07-22 | Confirmed current V1 modes, dispatcher, backend capability, and ownership terminology |
| Download selected raw files at vLLM commit `6e96891...` | OS temp directory | Pinned compilation config, wrappers, forward context, MRv1/MRv2 runners, graph manager, attention backends, speculator, and worker memory paths |
| `rg` and line inspection over pinned upstream and local source | local Windows | Traced modes, keys, capture order, stable buffers, speculative shapes, state compatibility, memory, metrics, and local gaps |
| Documentation patches | repository docs only | Expanded Graph spec, added durable knowledge, and completed task evidence |

The temporary pinned-source directory was removed after validation.
