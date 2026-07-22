"""Qwen3.6 native MTP proposal implementation."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from nanovllm.engine.batch import (
    AttentionMetadata,
    ExecutionSignature,
    PreparedBatch,
    SamplingMetadata,
)
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.speculative import sample_from_logits
from nanovllm.utils.context import forward_context


@dataclass(slots=True)
class DraftProposal:
    token_ids: list[int]
    logits: torch.Tensor


@dataclass(slots=True)
class _InitialHostBatch:
    hidden: list[torch.Tensor] = field(default_factory=list)
    token_ids: list[int] = field(default_factory=list)
    positions: list[int] = field(default_factory=list)
    cu_q: list[int] = field(default_factory=lambda: [0])
    cu_k: list[int] = field(default_factory=lambda: [0])
    slots: list[int] = field(default_factory=list)
    context_lens: list[int] = field(default_factory=list)
    logits_indices: list[int] = field(default_factory=list)
    next_positions: list[int] = field(default_factory=list)


def _cuda_tensor(values, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(values, dtype=dtype, pin_memory=True).cuda(non_blocking=True)


class MTPProposer:
    """Build and execute native MTP draft chains for a target batch."""

    def __init__(
        self,
        target_model,
        draft_model,
        *,
        block_size: int,
        num_steps: int,
    ):
        if num_steps <= 0:
            raise ValueError("MTP proposal steps must be positive")
        self.target_model = target_model
        self.draft_model = draft_model
        self.block_size = block_size
        self.num_steps = num_steps

    def _slot_mapping(self, seq: Sequence, start: int, end: int) -> list[int]:
        slots = []
        start_block = start // self.block_size
        end_block = (end + self.block_size - 1) // self.block_size
        for block_index in range(start_block, end_block):
            block_start = seq.block_table[block_index] * self.block_size
            slot_start = block_start + (
                start % self.block_size if block_index == start_block else 0
            )
            slot_end = block_start + self.block_size
            if block_index == end_block - 1:
                slot_end = block_start + end - block_index * self.block_size
            slots.extend(range(slot_start, slot_end))
        return slots

    @staticmethod
    def _block_tables(seqs: list[Sequence]) -> torch.Tensor:
        width = max(len(seq.block_table) for seq in seqs)
        rows = [
            seq.block_table + [-1] * (width - len(seq.block_table))
            for seq in seqs
        ]
        return _cuda_tensor(rows, torch.int32)

    @staticmethod
    def _cache_mode(seqs: list[Sequence]) -> bool:
        modes = [bool(seq.block_table) for seq in seqs]
        if any(modes) and not all(modes):
            raise ValueError("MTP batch mixes cached and uncached sequences")
        return all(modes)

    @staticmethod
    def _shifted_inputs(
        seq: Sequence,
        token_groups: list[list[int]],
        accepted_counts: list[int],
        group: int,
    ) -> tuple[int, list[int], int]:
        scheduled = seq.num_scheduled_tokens
        if seq.will_sample:
            outputs = token_groups[group]
            accepted = accepted_counts[group]
            valid = 1 + accepted if seq.draft_token_ids else scheduled
            target_inputs = seq.scheduled_token_ids()[:valid]
            return valid, target_inputs[1:] + [outputs[-1]], group + 1
        start = seq.num_cached_tokens
        shifted = seq.token_ids[start + 1 : start + scheduled + 1]
        return scheduled, shifted, group

    def _collect_initial_host(
        self,
        seqs: list[Sequence],
        target_hidden: torch.Tensor,
        token_groups: list[list[int]],
        accepted_counts: list[int],
        use_kv_cache: bool,
    ) -> _InitialHostBatch:
        host = _InitialHostBatch()
        offset = group = 0
        for seq in seqs:
            scheduled = seq.num_scheduled_tokens
            sequence_hidden = target_hidden[offset : offset + scheduled]
            offset += scheduled
            valid, shifted, group = self._shifted_inputs(
                seq, token_groups, accepted_counts, group
            )
            if len(shifted) != valid:
                raise ValueError(
                    f"cannot build shifted MTP inputs for sequence {seq.seq_id}"
                )
            start, end = seq.num_cached_tokens, seq.num_cached_tokens + valid
            host.hidden.append(sequence_hidden[:valid])
            host.token_ids.extend(shifted)
            host.positions.extend(range(start, end))
            host.cu_q.append(host.cu_q[-1] + valid)
            host.cu_k.append(host.cu_k[-1] + end)
            host.context_lens.append(end)
            if seq.will_sample:
                host.logits_indices.append(host.cu_q[-1] - 1)
                host.next_positions.append(end)
            if use_kv_cache:
                host.slots.extend(self._slot_mapping(seq, start, end))
        return host

    def _prepared_batch(
        self,
        *,
        token_ids: torch.Tensor,
        positions: torch.Tensor,
        cu_q: torch.Tensor,
        cu_k: torch.Tensor,
        slots: torch.Tensor,
        context_lens: torch.Tensor,
        logits_indices: torch.Tensor,
        block_tables: torch.Tensor | None,
        max_q: int,
        max_k: int,
        uniform_q: int | None,
    ) -> PreparedBatch:
        return PreparedBatch(
            input_ids=token_ids,
            positions=positions,
            signature=ExecutionSignature(
                token_ids.numel(), context_lens.numel(), token_ids.numel(), uniform_q
            ),
            attention=AttentionMetadata(
                cu_seqlens_q=cu_q,
                cu_seqlens_k=cu_k,
                max_seqlen_q=max_q,
                max_seqlen_k=max_k,
                slot_mapping=slots,
                block_tables=block_tables,
                context_lens=context_lens,
                use_kv_cache=block_tables is not None,
            ),
            sampling=SamplingMetadata(logits_indices=logits_indices),
        )

    def _initial_batch(
        self,
        host: _InitialHostBatch,
        seqs: list[Sequence],
        use_kv_cache: bool,
    ) -> tuple[PreparedBatch, torch.Tensor]:
        query_lengths = [end - start for start, end in zip(host.cu_q, host.cu_q[1:])]
        uniform = (
            query_lengths[0]
            if use_kv_cache and all(item == query_lengths[0] for item in query_lengths)
            else None
        )
        batch = self._prepared_batch(
            token_ids=_cuda_tensor(host.token_ids, torch.long),
            positions=_cuda_tensor(host.positions, torch.long),
            cu_q=_cuda_tensor(host.cu_q, torch.int32),
            cu_k=_cuda_tensor(host.cu_k, torch.int32),
            slots=_cuda_tensor(host.slots, torch.int32),
            context_lens=_cuda_tensor(host.context_lens, torch.int32),
            logits_indices=_cuda_tensor(host.logits_indices, torch.long),
            block_tables=self._block_tables(seqs) if use_kv_cache else None,
            max_q=max(query_lengths),
            max_k=max(host.context_lens),
            uniform_q=uniform,
        )
        return batch, torch.cat(host.hidden, dim=0)

    @staticmethod
    def _sample_logits(
        logits: torch.Tensor,
        temperatures: list[float],
        generators: list[torch.Generator | None],
    ) -> torch.Tensor:
        values = [
            sample_from_logits(row, temperature, generator=generator)
            for row, temperature, generator in zip(logits, temperatures, generators)
        ]
        return torch.tensor(values, dtype=torch.long, device=logits.device)

    def _run_draft(
        self,
        batch: PreparedBatch,
        hidden_states: torch.Tensor,
        temperatures: list[float],
        generators: list[torch.Generator | None],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        with forward_context(batch):
            embeddings = self.target_model.model.language_model.embed_tokens(
                batch.input_ids
            )
            output = self.draft_model(batch.positions, hidden_states, embeddings)
            indices = batch.sampling.logits_indices
            if indices is None or indices.numel() == 0:
                return None
            sampled_hidden = output.index_select(0, indices)
            logits = self.target_model.compute_logits(output)
            proposal_ids = self._sample_logits(logits, temperatures, generators)
        return proposal_ids, sampled_hidden, logits

    def _recursive_batch(
        self,
        sampled_seqs: list[Sequence],
        proposal_ids: torch.Tensor,
        next_positions: list[int],
        use_kv_cache: bool,
    ) -> PreparedBatch:
        size = len(sampled_seqs)
        positions = _cuda_tensor(next_positions, torch.long)
        cu_q = torch.arange(size + 1, dtype=torch.int32, device="cuda")
        if not use_kv_cache:
            return self._prepared_batch(
                token_ids=proposal_ids.to(torch.long), positions=positions,
                cu_q=cu_q, cu_k=cu_q,
                slots=torch.empty(0, dtype=torch.int32, device="cuda"),
                context_lens=torch.ones(size, dtype=torch.int32, device="cuda"),
                logits_indices=torch.arange(size, dtype=torch.long, device="cuda"),
                block_tables=None, max_q=1, max_k=1, uniform_q=None,
            )
        cu_k_host = [0]
        slots, contexts = [], []
        for seq, position in zip(sampled_seqs, next_positions):
            cu_k_host.append(cu_k_host[-1] + position + 1)
            contexts.append(position + 1)
            slots.extend(self._slot_mapping(seq, position, position + 1))
        return self._prepared_batch(
            token_ids=proposal_ids.to(torch.long), positions=positions,
            cu_q=cu_q, cu_k=_cuda_tensor(cu_k_host, torch.int32),
            slots=_cuda_tensor(slots, torch.int32),
            context_lens=_cuda_tensor(contexts, torch.int32),
            logits_indices=torch.arange(size, dtype=torch.long, device="cuda"),
            block_tables=self._block_tables(sampled_seqs),
            max_q=1, max_k=max(contexts), uniform_q=1,
        )

    def _recursive_step(
        self,
        sampled_seqs: list[Sequence],
        proposal_ids: torch.Tensor,
        sampled_hidden: torch.Tensor,
        next_positions: list[int],
        use_kv_cache: bool,
        temperatures: list[float],
        generators: list[torch.Generator | None],
    ) -> tuple[torch.Tensor, torch.Tensor, list[int], torch.Tensor]:
        batch = self._recursive_batch(
            sampled_seqs, proposal_ids, next_positions, use_kv_cache
        )
        result = self._run_draft(batch, sampled_hidden, temperatures, generators)
        if result is None:
            raise RuntimeError("recursive MTP step produced no sampled logits")
        proposal_ids, sampled_hidden, logits = result
        return proposal_ids, sampled_hidden, [item + 1 for item in next_positions], logits

    @staticmethod
    def _format_proposals(
        chains: list[list[int]],
        logit_steps: list[torch.Tensor],
        temperatures: list[float],
    ) -> list[DraftProposal]:
        vocab = logit_steps[-1].shape[-1]
        return [
            DraftProposal(
                chain,
                torch.stack([step[index] for step in logit_steps]).detach()
                if temperatures[index] > 0
                else torch.empty(
                    0, vocab, dtype=logit_steps[-1].dtype,
                    device=logit_steps[-1].device,
                ),
            )
            for index, chain in enumerate(chains)
        ]

    @torch.inference_mode()
    def propose(
        self,
        seqs: list[Sequence],
        target_hidden_states: torch.Tensor,
        token_groups: list[list[int]],
        accepted_counts: list[int],
        temperatures: list[float],
        generators: list[torch.Generator | None],
    ) -> list[DraftProposal | None]:
        sampled_seqs = [seq for seq in seqs if seq.will_sample]
        if len(temperatures) != len(sampled_seqs) or len(generators) != len(sampled_seqs):
            raise ValueError("MTP sampling inputs do not match sampled requests")
        use_kv_cache = self._cache_mode(seqs)
        host = self._collect_initial_host(
            seqs, target_hidden_states, token_groups, accepted_counts, use_kv_cache
        )
        if not host.token_ids:
            return []
        batch, hidden = self._initial_batch(host, seqs, use_kv_cache)
        initial = self._run_draft(batch, hidden, temperatures, generators)
        if initial is None:
            return []
        proposal_ids, sampled_hidden, logits = initial
        chains = [[token] for token in proposal_ids.tolist()]
        logit_steps = [logits]
        for _ in range(1, self.num_steps):
            proposal_ids, sampled_hidden, host.next_positions, logits = self._recursive_step(
                sampled_seqs, proposal_ids, sampled_hidden, host.next_positions,
                use_kv_cache, temperatures, generators,
            )
            logit_steps.append(logits)
            for chain, token in zip(chains, proposal_ids.tolist()):
                chain.append(token)
        return self._format_proposals(chains, logit_steps, temperatures)
