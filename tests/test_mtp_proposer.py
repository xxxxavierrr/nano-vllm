import pytest

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
