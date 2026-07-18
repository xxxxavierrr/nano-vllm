from types import SimpleNamespace

from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus


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
    scheduled, is_prefill = scheduler.schedule()

    assert is_prefill and scheduled == [seq]
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
    scheduled, is_prefill = scheduler.schedule()

    assert is_prefill and scheduled == [seq]
    assert seq in scheduler.waiting
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
    scheduler.running.remove(seq)
    scheduler.preempt(seq)

    assert not seq.block_table
    assert scheduler.abort(seq.seq_id)
    assert scheduler.is_finished()
    assert len(scheduler.block_manager.free_block_ids) == 8
