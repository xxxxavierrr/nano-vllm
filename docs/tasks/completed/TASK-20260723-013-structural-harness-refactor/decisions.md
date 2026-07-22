# Decisions

## 2026-07-23: line budgets are review triggers

Line count is used because it is executable and catches silent growth. It does
not replace ownership review and is not applied mechanically to Triton/CUDA
kernel bodies. Exceptions live in checked-in policy with reasons so they are
visible and removable.

## 2026-07-23: harness precedes refactoring

The checker and ownership contract land before production restructuring. This
prevents each refactor slice from inventing a different standard and makes the
before/after evidence reproducible.

## 2026-07-23: no flag-day runtime rewrite

Existing public contracts and numerical kernels remain stable. Components are
extracted behind current inputs/outputs and validated after every slice.

## 2026-07-23: legacy debt is capped, not blessed

The strict hook reports existing `Config`, cache/Graph manager, scheduler,
linear construction, transport, and serving functions as `debt`. Each ceiling
equals its pre-task size and growth fails the hook. These entries are not line
budget exceptions and must disappear when their owning component is touched.

## 2026-07-23: keep the task active for GPU equivalence

The refactor changes CUDA metadata assembly and launch orchestration even
though numerical kernels and public tensor layouts are preserved. CPU/Mock and
compile evidence are insufficient to archive it before RTX 4090D validation.

## 2026-07-23: model smoke is an import and ownership gate

Unit collection did not import the lazy public `LLM` path and did not exercise benchmark
warmup mutation. Public-entrypoint regression now covers moved types, while runtime metric
reset belongs to `HybridStateManager` and is exposed through `ModelRunner.reset_metrics()`;
benchmark code no longer assigns a read-only runner property.

## 2026-07-23: Piecewise capacity follows the token budget

The Piecewise capture limit is `min(piecewise_max_tokens, max_num_batched_tokens, 512)`.
Request concurrency and speculative decode width do not cap prefill/mixed capture buckets.
