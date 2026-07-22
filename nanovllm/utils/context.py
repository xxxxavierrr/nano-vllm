"""Exception-safe ambient metadata for one model forward.

Custom operators need access to rank-local per-call metadata without threading
it through every model layer.  The prepared batch is installed only for the
duration of a model invocation and nested scopes restore their parent.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

from nanovllm.engine.batch import PreparedBatch


_CONTEXT: ContextVar[PreparedBatch | None] = ContextVar(
    "nanovllm_forward_context",
    default=None,
)


def get_context() -> PreparedBatch:
    context = _CONTEXT.get()
    if context is None:
        raise RuntimeError("model execution requires a prepared forward context")
    return context


@contextmanager
def forward_context(batch: PreparedBatch) -> Iterator[PreparedBatch]:
    token = _CONTEXT.set(batch)
    try:
        yield batch
    finally:
        _CONTEXT.reset(token)
