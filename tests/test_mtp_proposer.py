import pytest
import torch

from nanovllm.engine.mtp_proposer import MTPProposer
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.speculative import GreedyAcceptance


def test_proposer_requires_at_least_one_draft_step():
    with pytest.raises(ValueError, match="steps must be positive"):
        MTPProposer(None, None, block_size=4, num_steps=0)


def test_proposer_maps_tokens_across_paged_cache_blocks():
    proposer = MTPProposer(None, None, block_size=4, num_steps=2)
    sequence = Sequence([1, 2, 3, 4, 5])
    sequence.block_table = [7, 11, 13]

    assert proposer._slot_mapping(sequence, 2, 9) == [
        30,
        31,
        44,
        45,
        46,
        47,
        52,
    ]


def test_acceptance_policy_is_independent_of_proposal_generation():
    policy = GreedyAcceptance()

    assert policy.accept([4, 9, 6], [4, 5]) == ([4, 9], 1)


def test_proposer_initial_host_builder_owns_shifted_inputs():
    proposer = MTPProposer(None, None, block_size=4, num_steps=2)
    sequence = Sequence([10])
    sequence.temperature = 0
    sequence.draft_token_ids = [20, 30]
    sequence.num_scheduled_tokens = 3
    hidden = torch.arange(12, dtype=torch.float32).reshape(3, 4)

    host = proposer._collect_initial_host(
        [sequence], hidden, [[20, 99]], [1], use_kv_cache=False
    )

    assert host.token_ids == [20, 99]
    assert host.positions == [0, 1]
    assert host.cu_q == [0, 2]
    assert host.cu_k == [0, 2]
    assert host.logits_indices == [1]
    assert host.next_positions == [2]
    assert torch.equal(host.hidden[0], hidden[:2])
