# Commands

## 2026-07-22 local planning environment

- Read `.workflow/README.md`, owning specs, task indexes, current GPTQ kernel,
  proposer/acceptance code, and hybrid-state/replay paths.
- Inspected pinned vLLM V1 GPTQ-Marlin, rejection sampler, rejection sampler
  utilities, GDN attention metadata, and Qwen GDN layer sources.
- No build, server, benchmark, commit, push, package installation, or GPU
  command was run for this planning-only task.
- Corrected the documented target from SM86 to RTX 4090D 24 GB / Ada SM89.

## 2026-07-22 local implementation

- `python -m compileall -q nanovllm tests`: passed after correcting import
  placement; final rerun passed.
- Targeted pytest for quantization, state, speculative sampling, scheduler, and
  sampler: passed (`53 passed, 2 skipped`) before the final focused rerun.
- Final focused GPTQ repack, probability sampler, sequence RNG transport, and
  branch-state tests: passed (`27 passed, 1 skipped`).
- Full test collection discovered 141 tests but failed to collect three modules
  because local Windows does not have `flash_attn`.
- Two broader regression invocations reached the 124-second command timeout;
  neither is recorded as passed.
- `uv run pytest` could not resolve the Linux-only Triton wheel on Windows; the
  existing project virtual environment was used without installing anything.
- No GPU, server, benchmark, commit, push, or dependency installation command
  was run.

## 2026-07-22 optimization-roadmap revision

- Re-read the repository workflow, active task/specs, benchmark aggregation,
  online result model, MTP sweep, and testing knowledge.
- Documented the revised optimization order and goodput/SLO metric contract.
- No implementation, GPU command, benchmark execution, dependency change,
  commit, or push was performed for this planning revision.
- Audited current source/tests for FP8 KV, indexed speculative state, lossless
  rejection sampling, GPTQ repack, W4A8, and benchmark metrics; corrected the
  roadmap to distinguish implemented-pending-validation from not implemented.

## 2026-07-22 atomic baseline and goodput implementation

- Checked tracked/untracked changes for common credential patterns;
  `git diff --check` and `compileall` passed before splitting.
- Created four local-only commits: indexed DeltaNet branches, lossless
  rejection sampling, GPTQ runtime repack, and the goodput-first roadmap.
- Ran focused pre-commit suites: state `19 passed, 16 skipped`; sampler
  `5 passed`; GPTQ `14 passed, 17 skipped`. CUDA skips remain pending.
- Implemented schema v3, planned-arrival SLO timing, service/client latency,
  SLO-good output tokens, accepted-token reporting, time-weighted occupancy,
  scheduled actual/padded tokens, adaptive offered-load sweep, and optional
  `nvidia-smi` telemetry with explicit missing provenance.
- Ran the full CPU/Mock goodput suite: `12 passed`.
- `compileall` and `git diff --check` passed after the goodput changes.
- No package installation, model download, GPU command, server command, or
  push was performed.
