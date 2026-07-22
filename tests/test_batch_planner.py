import torch

import nanovllm.engine.batch_planner as planner_module
from nanovllm.engine.batch_planner import BatchPlanner
from nanovllm.engine.cudagraph import BatchDescriptor, ExecutionMode
from nanovllm.engine.sequence import Sequence


class NoHybridState:
    enabled = False


def _cpu_tensor(values, dtype):
    return torch.tensor(values, dtype=dtype)


def test_batch_planner_builds_typed_uncached_prefill(monkeypatch):
    monkeypatch.setattr(planner_module, "_cuda_tensor", _cpu_tensor)
    planner = BatchPlanner(
        block_size=4, use_fp8_kv=False, hybrid_state=NoHybridState()
    )
    seq = Sequence([7, 8, 9])
    seq.num_scheduled_tokens = 3
    descriptor = BatchDescriptor(
        num_tokens=3,
        num_padded_tokens=3,
        num_seqs=1,
        uniform_query_len=None,
        execution_mode=ExecutionMode.EAGER,
    )

    batch = planner.prepare([seq], descriptor)

    assert batch.input_ids.tolist() == [7, 8, 9]
    assert batch.positions.tolist() == [0, 1, 2]
    assert batch.attention.cu_seqlens_q.tolist() == [0, 3]
    assert batch.attention.cu_seqlens_k.tolist() == [0, 3]
    assert batch.sampling.logits_indices.tolist() == [2]
    assert batch.gdn is None


def test_batch_planner_piecewise_padding_does_not_enter_attention(monkeypatch):
    monkeypatch.setattr(planner_module, "_cuda_tensor", _cpu_tensor)
    planner = BatchPlanner(
        block_size=4, use_fp8_kv=False, hybrid_state=NoHybridState()
    )
    seq = Sequence([4, 5, 6])
    seq.num_scheduled_tokens = 3
    descriptor = BatchDescriptor(
        num_tokens=3,
        num_padded_tokens=8,
        num_seqs=1,
        uniform_query_len=None,
        execution_mode=ExecutionMode.PIECEWISE,
    )

    batch = planner.prepare([seq], descriptor)

    assert batch.input_ids.tolist() == [4, 5, 6, 0, 0, 0, 0, 0]
    assert batch.signature.num_tokens == 3
    assert batch.attention.cu_seqlens_q.tolist() == [0, 3]
