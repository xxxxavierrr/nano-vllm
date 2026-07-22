import pytest
import torch

from nanovllm.engine.batch import (
    AttentionMetadata,
    ExecutionSignature,
    PreparedBatch,
    SamplingMetadata,
)
from nanovllm.utils.context import forward_context, get_context


def make_batch(token: int) -> PreparedBatch:
    return PreparedBatch(
        input_ids=torch.tensor([token]),
        positions=torch.tensor([0]),
        signature=ExecutionSignature(1, 1, 1, 1),
        attention=AttentionMetadata(),
        sampling=SamplingMetadata(),
    )


def test_forward_context_is_required_and_nested_scopes_restore_parent():
    outer = make_batch(1)
    inner = make_batch(2)

    with pytest.raises(RuntimeError, match="prepared forward context"):
        get_context()
    with forward_context(outer):
        assert get_context() is outer
        with forward_context(inner):
            assert get_context() is inner
        assert get_context() is outer
    with pytest.raises(RuntimeError, match="prepared forward context"):
        get_context()


def test_execution_signature_rejects_inconsistent_uniform_shape():
    with pytest.raises(ValueError, match="uniform query length"):
        ExecutionSignature(7, 2, 8, 4)
