# Design

## Harness

`docs/hooks/check_structure.py` parses Python ASTs and applies a checked-in policy.
It reports every function over the default review threshold and fails for
named architectural boundaries over their stricter budget. Numerical kernel
exceptions are explicit path/name patterns with reasons; they are not inferred
from decorators or module names.

The first version uses focused required limits rather than grandfathering the
whole repository. A baseline report remains visible so legacy debt cannot be
mistaken for compliance.

`docs/hooks/run_required_checks.py` is the cross-platform strict entry point.
It runs whitespace validation, Python compilation, and the structure policy;
focused pytest remains task-specific and is recorded separately.

## Ownership boundaries

### Benchmark aggregation

Owns conversion of request events to immutable timing/SLO facts and aggregation
of those facts. It delegates engine snapshot aggregation. It owns no transport,
load generation, CLI, or GPU sampling. `summarize` must not recompute an SLO
predicate in multiple branches.

### GPTQ calibration

The quantizer orchestrator owns algorithm order. Hessian conditioning, scale
construction, block error propagation, and packed serialization are focused
private operations. Checkpoint loading/writing remains outside the numerical
quantizer.

### FP8 DeltaNet state

Layout/capacity arithmetic, CPU/reference codecs, slot-pool lifecycle, and
experimental Triton runtime live behind separate module boundaries. None may
import engine request objects or select scheduler policy.

### Batch planner

Owns CPU translation from sequences and an execution descriptor to one
`PreparedBatch`. It delegates attention/KV, GDN/state, sampling/logits, and
stable-graph-buffer preparation. It owns no model forward, acceptance, proposal,
or metrics.

### Speculative step coordinator

Owns verification result parsing, acceptance-policy invocation, prefix commit,
next-proposal bookkeeping, and speculative metrics. It delegates numerical
target/draft forward to the runner/proposer and state lifetime to the state
transaction. `ModelRunner.run` owns only phase order and the ordinary sampling
fast path.

## Compatibility

Public CLI, JSON schema, model outputs, cache/state tensor layouts, custom-op
signatures, and checkpoint formats remain unchanged. Refactoring is performed
in behavior-preserving slices with focused tests after each slice.
