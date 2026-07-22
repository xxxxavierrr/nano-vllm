# Command log

Working directory: repository root. Existing dirty GDN/runtime edits are part
of the current worktree and must not be discarded or overwritten blindly.

| Command/action | Outcome |
| --- | --- |
| Dirty diff and all context/model fallback call sites inspected | Preserved existing GDN numerical refactor and mapped every migration path |
| `compileall` over `nanovllm`, tests, and MTP comparison tool | passed |
| Targeted context/dispatcher tests | 11 passed |
| CPU regression including scheduler/speculation/capacity/GPTQ | 53 passed, 1 skipped |
| Attention/Qwen CPU collection | unavailable because local `flash_attn` is absent; no dependency changes made |

## 2026-07-23 server validation

- Unfiltered GPU suite: `198 passed`.
- Qwen3-0.6B eager and FULL/PIECEWISE smokes plus Qwen3.6 GPTQ/MTP smoke
  exercised the typed batch path. Archived without dependency changes.
