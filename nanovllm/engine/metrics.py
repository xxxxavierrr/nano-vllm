"""Typed rank-local execution results and metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class SpeculativeStepMetrics:
    drafted: int = 0
    proposed: int = 0
    accepted: int = 0
    rejected: int = 0
    bonus: int = 0
    verification_rounds: int = 0
    accepted_position_1: int = 0
    accepted_position_2: int = 0
    accepted_position_3: int = 0


@dataclass(frozen=True, slots=True)
class RunnerStepMetrics:
    execution_mode: str
    real_tokens: int
    padded_tokens: int
    num_requests: int
    speculative: SpeculativeStepMetrics = field(
        default_factory=SpeculativeStepMetrics
    )


@dataclass(slots=True)
class RunnerStepOutput:
    result: Any
    metrics: RunnerStepMetrics
