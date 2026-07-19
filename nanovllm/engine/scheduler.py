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
                seq.draft_token_ids.clear()
                return True
        return False

    def schedule(self) -> SchedulerBatch:
        """Fill one token budget with running work, then waiting prefills."""
        scheduled_seqs: list[Sequence] = []
        protected: set[int] = set()
        token_budget = self.max_num_batched_tokens
        prefill_tokens = 0
        decode_tokens = 0

        running_snapshot = sorted(self.running, key=lambda seq: seq.is_prefill)
        for seq in running_snapshot:
            if token_budget == 0 or len(scheduled_seqs) >= self.max_num_seqs:
                break
            if seq not in self.running:
                continue
            outstanding = seq.num_target_inputs
            assert outstanding > 0
            was_prefill = seq.is_prefill
            if seq.is_speculative and token_budget < outstanding:
                continue
            num_tokens = min(outstanding, token_budget)
            if not was_prefill:
                if not self._reserve_decode_slots(
                    seq, protected, num_tokens
                ):
                    continue
                self.block_manager.may_append_tokens(seq, num_tokens)
            seq.num_scheduled_tokens = num_tokens
            scheduled_seqs.append(seq)
            protected.add(seq.seq_id)
            token_budget -= num_tokens
            if was_prefill:
                prefill_tokens += num_tokens
            else:
                decode_tokens += num_tokens

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

        if not scheduled_seqs:
            raise RuntimeError(
                "scheduler could not reserve a speculative verification batch; "
                "increase max_num_batched_tokens"
            )
        reset_sequence_ids = tuple(self._reset_sequence_ids)
        self._reset_sequence_ids.clear()
        return SchedulerBatch(
            scheduled_seqs, prefill_tokens, decode_tokens, reset_sequence_ids
        )

    def _reserve_decode_slots(
        self,
        seq: Sequence,
        protected: set[int],
        num_input_tokens: int,
    ) -> bool:
        while not self.block_manager.can_append_tokens(seq, num_input_tokens):
            candidates = [
                candidate
                for candidate in reversed(self.running)
                if candidate is not seq and candidate.seq_id not in protected
            ]
            victim = next(
                (candidate for candidate in candidates if candidate.is_prefill),
                None,
            )
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
        seq.draft_token_ids.clear()
        self.block_manager.deallocate(seq)
        self._reset_sequence_ids.append(seq.seq_id)
        self.waiting.appendleft(seq)

    def _append_output_token(self, seq: Sequence, token_id: int) -> bool:
        seq.append_token(token_id)
        reached_eos = not seq.ignore_eos and token_id == self.eos
        reached_limit = seq.num_completion_tokens >= seq.max_tokens
        if reached_eos or reached_limit:
            seq.finish_reason = "stop" if reached_eos else "length"
            seq.status = SequenceStatus.FINISHED
            return True
        return False

    def postprocess(
        self,
        batch: SchedulerBatch,
        token_ids: list[int] | list[list[int]],
        *,
        accepted_counts: list[int] | None = None,
        next_draft_token_ids: list[int | None] | None = None,
    ):
        sampled = batch.sampled_sequences
        if len(token_ids) != len(sampled):
            raise ValueError(
                f"model returned {len(token_ids)} token groups for "
                f"{len(sampled)} sampled sequences"
            )
        token_groups = [
            group if isinstance(group, list) else [group]
            for group in token_ids
        ]
        if accepted_counts is None:
            accepted_counts = [0] * len(sampled)
        if next_draft_token_ids is None:
            next_draft_token_ids = [None] * len(sampled)
        if not (
            len(accepted_counts)
            == len(next_draft_token_ids)
            == len(sampled)
        ):
            raise ValueError("speculative result metadata has the wrong length")

        result_iter = iter(
            zip(token_groups, accepted_counts, next_draft_token_ids)
        )
        for seq in batch.sequences:
            will_sample = seq.will_sample
            if not will_sample:
                self.block_manager.hash_blocks(seq)
                seq.num_cached_tokens += seq.num_scheduled_tokens
                seq.num_scheduled_tokens = 0
                continue

            outputs, accepted, next_draft = next(result_iter)
            previous_drafts = list(seq.draft_token_ids)
            is_speculative = bool(previous_drafts)
            if is_speculative:
                if not 0 <= accepted <= len(previous_drafts):
                    raise ValueError(
                        f"invalid accepted draft count {accepted} for "
                        f"sequence {seq.seq_id}"
                    )
                if outputs[:accepted] != previous_drafts[:accepted]:
                    raise ValueError(
                        "accepted outputs do not match the proposed draft prefix"
                    )
                committed_inputs = 1 + accepted
            else:
                if accepted:
                    raise ValueError(
                        "non-speculative sequence cannot accept draft tokens"
                    )
                committed_inputs = seq.num_scheduled_tokens

            original_scheduled = seq.num_scheduled_tokens
            seq.num_scheduled_tokens = committed_inputs
            finished = False
            for token_id in outputs[:accepted]:
                if self._append_output_token(seq, token_id):
                    finished = True
                    break

            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += committed_inputs
            seq.num_scheduled_tokens = 0
            seq.draft_token_ids.clear()

            if not finished:
                for token_id in outputs[accepted:]:
                    if self._append_output_token(seq, token_id):
                        finished = True
                        break

            if finished:
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
            elif next_draft is not None:
                seq.draft_token_ids.append(next_draft)

            if original_scheduled < committed_inputs:
                raise AssertionError("committed more target inputs than scheduled")
