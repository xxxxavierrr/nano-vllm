from collections import deque
from dataclasses import dataclass

from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


@dataclass(slots=True)
class SchedulerBatch:
    sequences: list[Sequence]
    prefill_tokens: int
    decode_tokens: int
    reset_sequence_ids: tuple[int, ...] = ()

    @property
    def total_tokens(self) -> int:
        return self.prefill_tokens + self.decode_tokens

    @property
    def sampled_sequences(self) -> list[Sequence]:
        return [seq for seq in self.sequences if seq.will_sample]


class Scheduler:

    def __init__(self, config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.enable_prefix_cache = getattr(config, "enable_prefix_cache", True)
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(
            config.num_kvcache_blocks,
            config.kvcache_block_size,
            enable_prefix_cache=self.enable_prefix_cache,
        )
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self._reset_sequence_ids: list[int] = []
        self.num_preemptions = 0

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def abort(self, seq_id: int) -> bool:
        for queue in (self.waiting, self.running):
            for seq in queue:
                if seq.seq_id != seq_id:
                    continue
                queue.remove(seq)
                if seq.block_table:
                    self.block_manager.deallocate(seq)
                seq.num_scheduled_tokens = 0
                return True
        return False

    def schedule(self) -> SchedulerBatch:
        """Fill one token budget with running work, then waiting prefills.

        There is intentionally no global prefill/decode phase. Each sequence
        advances from ``num_cached_tokens`` toward its current token frontier.
        Existing decode work is ordered before partial prefills; remaining
        budget admits and chunks new prompts.
        """
        scheduled_seqs: list[Sequence] = []
        protected: set[int] = set()
        token_budget = self.max_num_batched_tokens
        prefill_tokens = 0
        decode_tokens = 0

        # Stable sorting preserves request order within decode and prefill.
        running_snapshot = sorted(self.running, key=lambda seq: seq.is_prefill)
        for seq in running_snapshot:
            if token_budget == 0 or len(scheduled_seqs) >= self.max_num_seqs:
                break
            if seq not in self.running:
                continue
            outstanding = seq.num_tokens - seq.num_cached_tokens
            assert outstanding > 0
            was_prefill = seq.is_prefill
            if not was_prefill:
                if not self._reserve_decode_slot(seq, protected):
                    continue
                self.block_manager.may_append(seq)
            num_tokens = min(outstanding, token_budget)
            seq.num_scheduled_tokens = num_tokens
            scheduled_seqs.append(seq)
            protected.add(seq.seq_id)
            token_budget -= num_tokens
            if was_prefill:
                prefill_tokens += num_tokens
            else:
                decode_tokens += num_tokens

        # New prompts consume only the budget left after running requests.
        while (
            self.waiting
            and token_budget > 0
            and len(self.running) < self.max_num_seqs
            and len(scheduled_seqs) < self.max_num_seqs
        ):
            seq = self.waiting[0]
            num_cached_blocks = self.block_manager.can_allocate(seq)
            if num_cached_blocks == -1:
                break
            self.block_manager.allocate(seq, num_cached_blocks)
            self.waiting.popleft()
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)

            outstanding = seq.num_tokens - seq.num_cached_tokens
            assert outstanding > 0
            num_tokens = min(outstanding, token_budget)
            seq.num_scheduled_tokens = num_tokens
            scheduled_seqs.append(seq)
            protected.add(seq.seq_id)
            prefill_tokens += num_tokens
            token_budget -= num_tokens

        assert scheduled_seqs
        reset_sequence_ids = tuple(self._reset_sequence_ids)
        self._reset_sequence_ids.clear()
        return SchedulerBatch(
            scheduled_seqs, prefill_tokens, decode_tokens, reset_sequence_ids
        )

    def _reserve_decode_slot(self, seq: Sequence, protected: set[int]) -> bool:
        while not self.block_manager.can_append(seq):
            candidates = [
                candidate
                for candidate in reversed(self.running)
                if candidate is not seq and candidate.seq_id not in protected
            ]
            victim = next((candidate for candidate in candidates if candidate.is_prefill), None)
            if victim is None:
                victim = next(iter(candidates), None)
            if victim is None:
                self.preempt(seq)
                return False
            self.preempt(victim)
        return True

    def preempt(self, seq: Sequence):
        self.num_preemptions += 1
        if seq in self.running:
            self.running.remove(seq)
        seq.status = SequenceStatus.WAITING
        seq.num_scheduled_tokens = 0
        self.block_manager.deallocate(seq)
        self._reset_sequence_ids.append(seq.seq_id)
        self.waiting.appendleft(seq)

    def postprocess(self, batch: SchedulerBatch, token_ids: list[int]):
        expected_tokens = len(batch.sampled_sequences)
        if len(token_ids) != expected_tokens:
            raise ValueError(
                f"model returned {len(token_ids)} tokens for {expected_tokens} sampled sequences"
            )
        token_iter = iter(token_ids)
        for seq in batch.sequences:
            will_sample = seq.will_sample
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if not will_sample:
                continue
            token_id = next(token_iter)
            seq.append_token(token_id)
            reached_eos = not seq.ignore_eos and token_id == self.eos
            reached_limit = seq.num_completion_tokens == seq.max_tokens
            if reached_eos or reached_limit:
                seq.finish_reason = "stop" if reached_eos else "length"
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
