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

## 2026-07-22 native W4 local scaffold

- Added an opt-in setuptools CUDA extension (`NANOVLLM_BUILD_CUDA_EXT=1`) with
  SM89 compile target; default installation does not import the PyTorch C++
  extension toolchain.
- Added explicit `auto|triton|marlin` backend selection. `auto` remains Triton;
  native selection requires extension availability, SM89, symmetric group-128
  repacked weights, and fused activation permutation metadata.
- Added small-M/large-M W4A16 CUDA entry points and an experimental large-M
  W4A8 entry point, plus CPU activation-quantization reference and dispatcher
  tests.
- Focused W4/config/API suite: `32 passed, 17 skipped`; all 17 skips require
  CUDA and are not counted as native-kernel validation.
- Python source compilation and `git diff --check` passed. CUDA compilation was
  not attempted because the local machine has no CUDA toolkit/GPU.
- A direct `python setup.py --name` metadata check was unavailable because the
  external local interpreter does not contain `setuptools`; no dependency was
  installed. Normal PEP 517 builds provision setuptools from `pyproject.toml`;
  opt-in CUDA builds must use an environment that already has matching PyTorch
  and CUDA (for example a no-build-isolation server build).

## 2026-07-22 DSpark calibration tooling

- Added a draft-only DFlash/Markov/confidence calibration shell, strict
  streaming safetensors loader, resumable/hash-verified calibration cache, and
  dry-run size projection. No target model is constructed by this path.
- Added an in-repository symmetric GPTQ INT4 quantizer: group size 128,
  128-column blocks, FP32 Hessian, 1% diagonal damping, sequential error
  propagation, GPTQ zero-point packing, and current-loader tensor names.
- Synthetic cache/model/checkpoint and production-loader compatibility suite:
  `20 passed, 1 skipped`; the skip requires CUDA and is not DSpark evidence.
- `compileall` and `git diff --check` passed. No real BF16 draft was loaded, no
  model was downloaded, and no GPU/server/push command was run.

## 2026-07-22 local state/sampler/KV evidence strengthening

- Made multi-request prefix commits prevalidated and atomic; added committed
  branch/discard counters and an explicit rejected-prefix target-replay counter.
- Extracted a pure KV layout/capacity report used by runtime allocation and
  reporting, including FP8 scale overhead and native MTP cache bytes.
- Added randomized rejection-prefix invariants, multi-request prefix tests, and
  Qwen3.6-style BF16/FP8 capacity comparisons.
- Focused suite: `31 passed, 9 skipped`; all skips are CUDA attention tests and
  remain pending. No GPU/server/push command was run.
