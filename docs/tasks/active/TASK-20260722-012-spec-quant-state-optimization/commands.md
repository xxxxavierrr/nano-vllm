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

## 2026-07-22 FP8 DeltaNet state local implementation

- Added independent `delta_state_dtype={auto,fp8_e4m3}` plumbing to config,
  offline benchmark, API server, smoke/quant tools, and server script.
- Added CPU references for conv per-channel and recurrent per-head/K-row FP8
  E4M3 with FP16 scales, a shared slot pool, native/FP8 capacity reports, and a
  GPU-independent capacity CLI.
- Added experimental Triton row quantize/dequantize source behind an explicit
  environment gate. Production engine startup is fail-closed for FP8 state
  until fused GDN numerical/Graph validation exists; it never silently falls
  back to native state.
- Focused FP8 state/config/API/lifecycle suite: `37 passed`. `compileall` and
  `git diff --check` passed. No CUDA compilation, GPU/server command, package
  install, model download, or push was performed.
- Combined CPU/Mock regression across goodput, GPTQ/native dispatch, DSpark,
  speculative state/sampling, FP8 KV/capacity, FP8 Delta state, and API:
  `83 passed, 26 skipped`. The skipped cases are CUDA-only and remain pending.

## 2026-07-23 native W4 SM89 validation and first optimization pass

- Verified the restored server stack: RTX 4090D / SM89, PyTorch 2.8.0+cu128,
  CUDA toolkit 12.8, and the existing repository environment. No dependency
  was installed and build products stayed under `/tmp/nanovllm-w4-build`.
- Built the opt-in extension with
  `NANOVLLM_BUILD_CUDA_EXT=1 TORCH_CUDA_ARCH_LIST=8.9 python setup.py build_ext`
  using the existing toolkit. The first build used the scalar scaffold; later
  builds compiled the WMMA prototype. Distutils was used because ninja is not
  installed.
- The first native numerical run failed because the load-time repacked weight
  path discarded `input_perm` (`191/192` elements wrong). Propagated the
  permutation through Python, C++, W4A16, and W4A8 and added strict CUDA layout
  validation. The corrected scalar kernel met `rtol=atol=3e-2` for
  `M=1,8,32,64,65,128,512`, with relative L2 about `0.22%-0.24%` versus the
  strict FP32 reference.
- Direct pybind calls failed `torch.compile(fullgraph=True)` with an unsupported
  PyCapsule. Registered `nanovllm_native` schemas and CUDA implementations via
  `TORCH_LIBRARY`, added fake implementations, and routed runtime through
  `torch.ops`. Fullgraph output then matched eager exactly and direct CUDA
  Graph capture/replay matched eager exactly.
- Replaced the scalar W4A16 core with a shared-memory WMMA prototype. CUDA 12.8
  required `__nv_bfloat16` fragment types rather than
  `wmma::precision::bfloat16`; the failed compile and correction are retained
  as evidence. Both public small/large entry points reuse one templated core.
- Changed the K tile from 16 to the GPTQ group size 128 and used 16x64 small-M
  versus 32x64 large-M tiles. The dominant fix was loading each INT32 packed
  word once and expanding its eight INT4 values in registers instead of eight
  redundant global loads.
- Representative `K=N=5120` latency after the packed-word fix, native versus
  repacked Triton in milliseconds: `M=1 0.139/0.118`, `M=8 0.174/0.118`,
  `M=64 0.269/0.145`, `M=65 0.314/0.168`, `M=128 0.357/0.252`, and
  `M=512 1.313/0.747`. Outputs matched within the existing BF16 tolerance;
  this prototype remains opt-in because it is still `1.18x-1.87x` slower.
- Formal native/Triton GPU regression:
  `tests/test_gptq_native.py tests/test_gptq_kernel.py` passed (`29 passed`).
  `git diff --check` passed before the test run.
- Final server regression with the temporary native extension on the package
  path: `205 passed, 1 warning` in 36.24 seconds. `compileall` and
  `git diff --check` passed immediately before it.

## 2026-07-23 W4 tile investigation and task consolidation

- Triton autotune on `K=N=5120` selected `16x32x32` for M=1/8,
  `16x64x32` for M=64/128, and `32x128x32` with eight warps for M=512.
- A native K=32-only variant improved M=512 to about 1.00 ms but regressed
  small/mid M. The retained source specializes small to `16x64x128` and large
  to `32x128x32`, with two accumulators per warp for the large tile.
- The specialized large tile measured about 0.899 ms at M=512 versus Triton
  0.748 ms (1.20x slower). The retained small tile measured about 0.139 ms at
  M=1 versus Triton 0.118 ms (1.18x slower). M=8/19/32/64 remained farther
  behind. An attempted `16x32x32` native small tile was rejected after
  measuring 1.54x-2.66x slower.
- `cuobjdump --dump-resource-usage` reported 40 registers and 24/32 KiB shared
  memory for the prior small/large variants; no register spill was observed.
  Nsight Compute is not installed, and no profiler package was added.
- Queried the official vLLM main tree and located its current stable-ABI Marlin
  implementation under `csrc/libtorch_stable/quantization/marlin/`. The
  source design uses a load-time Marlin layout and multi-stage async
  global-to-shared pipeline; detailed porting is deferred rather than copied
  incompletely.
- Consolidated TASK-002 and TASK-007 through TASK-011. Completed extraction
  tasks were archived completed; unresolved Graph/GDN/goodput gates were moved
  here before the older tasks were archived as superseded.
- Restored the retained small/large source after rejecting the 16x32x32
  candidate and rebuilt the extension successfully with CUDA 12.8 / SM89.
- Final required hook passed: `git diff --check`, `compileall`, and structure
  policy. Final native/GPTQ focused GPU regression passed (`29 passed`).
- A read-only documentation check verified 126 Markdown files with no missing
  relative links. Exactly one active task and one `[>]` plan item remain.
- Redacted secret scan across all changed files reported clean.
