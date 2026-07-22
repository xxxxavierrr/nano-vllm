# Research

## Repository findings

The existing architecture spec already defines `BatchPlanner`, state/cache
owners, proposer and acceptance interfaces, and a small runner facade. The
optimization task nevertheless enlarged existing entry points because its
functional plan did not translate those boundaries into structural acceptance
criteria.

The 2026-07-23 AST inventory found the following review-trigger functions:

- `bench.py:main`: 479 lines;
- `ModelRunner.prepare_inputs`: 231 lines;
- `ModelRunner.run`: 210 lines;
- `ModelRunner.capture_cudagraph`: 205 lines;
- `ModelRunner.allocate_kv_cache`: 188 lines;
- `MTPProposer.propose`: 185 lines;
- `benchmarks.metrics.summarize`: 156 lines;
- `quantize_linear_gptq`: 65 lines with five algorithmic responsibilities.

Triton/CUDA numerical kernels are not judged by the orchestration line budget.
Their wrappers still require one semantic responsibility and explicit runtime
validation.

## Root cause

Functional tests covered numerical and protocol behavior, but no pre/post
structure scan existed. The task design said that `ModelRunner` owned phase
ordering without saying it must delegate each phase, allowing ownership of
order to drift into ownership of implementation.
