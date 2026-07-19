from types import SimpleNamespace

from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.speculative import greedy_accept_k1
from nanovllm.sampling_params import SamplingParams


def make_scheduler(block_size: int = 4) -> Scheduler:
    Sequence.block_size = block_size
    return Scheduler(
        SimpleNamespace(
            max_num_seqs=4,
            max_num_batched_tokens=8,
            eos=99,
            enable_prefix_cache=False,
            kvcache_block_size=block_size,
            num_kvcache_blocks=16,
        )
    )


def test_greedy_accept_k1_accepts_and_returns_bonus():
    assert greedy_accept_k1([7, 8], 7) == ([7, 8], 1)


def test_greedy_accept_k1_rejects_and_returns_replacement():
    assert greedy_accept_k1([7, 8], 6) == ([7], 0)


def test_draft_is_not_committed_before_acceptance():
    scheduler = make_scheduler()
    seq = Sequence([1, 2], SamplingParams(temperature=0, max_tokens=8))
    scheduler.add(seq)
    first = scheduler.schedule()
    scheduler.postprocess(
        first,
        [[3]],
        accepted_counts=[0],
        next_draft_token_ids=[4],
    )
    assert seq.token_ids == [1, 2, 3]
    assert seq.draft_token_ids == [4]

    verify = scheduler.schedule()
    assert seq.scheduled_token_ids() == [3, 4]
    assert seq.token_ids == [1, 2, 3]


def test_accept_and_reject_advance_only_committed_target_inputs():
    scheduler = make_scheduler()
    seq = Sequence([1, 2], SamplingParams(temperature=0, max_tokens=8))
    scheduler.add(seq)
    scheduler.postprocess(
        scheduler.schedule(),
        [[3]],
        accepted_counts=[0],
        next_draft_token_ids=[4],
    )

    scheduler.postprocess(
        scheduler.schedule(),
        [[4, 5]],
        accepted_counts=[1],
        next_draft_token_ids=[6],
    )
    assert seq.token_ids == [1, 2, 3, 4, 5]
    assert seq.num_cached_tokens == 4
    assert seq.draft_token_ids == [6]

    scheduler.postprocess(
        scheduler.schedule(),
        [[7]],
        accepted_counts=[0],
        next_draft_token_ids=[8],
    )
    assert seq.token_ids == [1, 2, 3, 4, 5, 7]
    assert seq.num_cached_tokens == 5
    assert seq.draft_token_ids == [8]


def test_speculative_lookahead_reserves_boundary_block():
    scheduler = make_scheduler(block_size=4)
    seq = Sequence([1, 2, 3, 4], SamplingParams(temperature=0, max_tokens=8))
    scheduler.add(seq)
    scheduler.postprocess(
        scheduler.schedule(),
        [[5]],
        accepted_counts=[0],
        next_draft_token_ids=[6],
    )
    assert len(seq.block_table) == 1
    verify = scheduler.schedule()
    assert verify.total_tokens == 2
    assert len(seq.block_table) == 2


def test_eos_in_accepted_draft_discards_bonus_and_next_draft():
    scheduler = make_scheduler()
    seq = Sequence([1], SamplingParams(temperature=0, max_tokens=8))
    scheduler.add(seq)
    scheduler.postprocess(
        scheduler.schedule(),
        [[2]],
        accepted_counts=[0],
        next_draft_token_ids=[99],
    )
    scheduler.postprocess(
        scheduler.schedule(),
        [[99, 7]],
        accepted_counts=[1],
        next_draft_token_ids=[8],
    )
    assert seq.is_finished
    assert seq.finish_reason == "stop"
    assert seq.completion_token_ids == [2, 99]
    assert seq.draft_token_ids == []
