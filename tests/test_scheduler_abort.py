from types import SimpleNamespace

import pytest

from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


def make_scheduler(max_num_batched_tokens: int = 16) -> Scheduler:
    config = SimpleNamespace(
        max_num_seqs=8,
        max_num_batched_tokens=max_num_batched_tokens,
        eos=2,
        kvcache_block_size=256,
        num_kvcache_blocks=8,
    )
    Sequence.block_size = config.kvcache_block_size
    return Scheduler(config)


def make_sequence(length: int = 4) -> Sequence:
    return Sequence(
        [1] * length,
        SamplingParams(temperature=1.0, max_tokens=4),
    )


def test_abort_waiting_request_without_allocated_blocks():
    scheduler = make_scheduler()
    seq = make_sequence()
    scheduler.add(seq)

    assert scheduler.abort(seq.seq_id)
    assert scheduler.is_finished()
    assert len(scheduler.block_manager.free_block_ids) == 8


def test_abort_running_request_releases_blocks():
    scheduler = make_scheduler()
    seq = make_sequence()
    scheduler.add(seq)
    batch = scheduler.schedule()

    assert batch.sequences == [seq]
    assert batch.prefill_tokens == len(seq)
    assert batch.decode_tokens == 0
    assert seq.status == SequenceStatus.RUNNING
    assert seq.block_table
    assert scheduler.abort(seq.seq_id)
    assert scheduler.is_finished()
    assert not seq.block_table
    assert len(scheduler.block_manager.free_block_ids) == 8


def test_abort_partial_prefill_releases_blocks():
    scheduler = make_scheduler(max_num_batched_tokens=1)
    seq = make_sequence(length=4)
    scheduler.add(seq)
    batch = scheduler.schedule()

    assert batch.sequences == [seq]
    assert batch.prefill_tokens == 1
    assert seq in scheduler.running
    assert seq.block_table
    assert scheduler.abort(seq.seq_id)
    assert scheduler.is_finished()
    assert seq.num_scheduled_tokens == 0
    assert len(scheduler.block_manager.free_block_ids) == 8


def test_abort_preempted_request_is_safe():
    scheduler = make_scheduler()
    seq = make_sequence()
    scheduler.add(seq)
    scheduler.schedule()
    scheduler.preempt(seq)

    assert not seq.block_table
    assert scheduler.abort(seq.seq_id)
    assert scheduler.is_finished()
    assert len(scheduler.block_manager.free_block_ids) == 8


def test_prefix_cached_token_count_survives_block_release():
    Sequence.block_size = 4
    manager = BlockManager(num_blocks=4, block_size=4)
    first = Sequence([1, 2, 3, 4, 5, 6, 7, 8])
    manager.allocate(first, num_cached_blocks=0)
    first.num_scheduled_tokens = first.num_tokens
    manager.hash_blocks(first)
    manager.deallocate(first)

    second = Sequence([1, 2, 3, 4, 9, 10, 11, 12])
    cached_blocks = manager.can_allocate(second)
    manager.allocate(second, cached_blocks)

    assert cached_blocks == 1
    assert second.num_prefix_cached_tokens == 4


def test_decode_and_chunked_prefill_share_one_batch():
    scheduler = make_scheduler(max_num_batched_tokens=4)
    decode = make_sequence(length=1)
    scheduler.add(decode)
    first_batch = scheduler.schedule()
    scheduler.postprocess(first_batch, [7])

    prefill = make_sequence(length=6)
    scheduler.add(prefill)
    mixed_batch = scheduler.schedule()

    assert mixed_batch.sequences == [decode, prefill]
    assert mixed_batch.decode_tokens == 1
    assert mixed_batch.prefill_tokens == 3
    assert decode.num_scheduled_tokens == 1
    assert prefill.num_scheduled_tokens == 3
    assert mixed_batch.sampled_sequences == [decode]

    scheduler.postprocess(mixed_batch, [8])
    assert decode.num_completion_tokens == 2
    assert prefill.num_cached_tokens == 3


def test_partial_prefill_only_samples_when_it_reaches_frontier():
    scheduler = make_scheduler(max_num_batched_tokens=2)
    seq = make_sequence(length=5)
    scheduler.add(seq)

    first_batch = scheduler.schedule()
    assert first_batch.sampled_sequences == []
    scheduler.postprocess(first_batch, [])
    assert seq.num_cached_tokens == 2
    assert seq.num_completion_tokens == 0

    second_batch = scheduler.schedule()
    assert second_batch.sampled_sequences == []
    scheduler.postprocess(second_batch, [])
    assert seq.num_cached_tokens == 4

    final_batch = scheduler.schedule()
    assert final_batch.sampled_sequences == [seq]
    scheduler.postprocess(final_batch, [9])
    assert seq.num_cached_tokens == 5
    assert seq.completion_token_ids == [9]


def test_postprocess_validates_sample_count_before_mutating_state():
    scheduler = make_scheduler()
    seq = make_sequence()
    scheduler.add(seq)
    batch = scheduler.schedule()

    with pytest.raises(ValueError, match="1 sampled sequences"):
        scheduler.postprocess(batch, [])

    assert seq.num_cached_tokens == 0
    assert seq.num_scheduled_tokens == len(seq)
    assert seq.num_completion_tokens == 0


def test_postprocess_distinguishes_stop_and_length():
    eos_scheduler = make_scheduler()
    eos_seq = make_sequence()
    eos_scheduler.add(eos_seq)
    eos_scheduler.postprocess(eos_scheduler.schedule(), [eos_scheduler.eos])
    assert eos_seq.is_finished
    assert eos_seq.finish_reason == "stop"
    assert not eos_seq.block_table

    length_scheduler = make_scheduler()
    length_seq = make_sequence()
    length_seq.max_tokens = 1
    length_scheduler.add(length_seq)
    length_scheduler.postprocess(length_scheduler.schedule(), [7])
    assert length_seq.is_finished
    assert length_seq.finish_reason == "length"
    assert not length_seq.block_table
