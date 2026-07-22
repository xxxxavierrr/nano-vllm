"""Qwen3.6 native MTP proposal implementation."""

from __future__ import annotations

from dataclasses import dataclass

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
            slot_start = block_start
            if block_index == start_block:
                slot_start += start % self.block_size
            slot_end = block_start + self.block_size
            if block_index == end_block - 1:
                slot_end = block_start + end - block_index * self.block_size
            slots.extend(range(slot_start, slot_end))
        return slots

    @staticmethod
    def _block_tables(seqs: list[Sequence]) -> torch.Tensor:
        max_len = max(len(seq.block_table) for seq in seqs)
        rows = [
            seq.block_table + [-1] * (max_len - len(seq.block_table))
            for seq in seqs
        ]
        return torch.tensor(
            rows,
            dtype=torch.int32,
            pin_memory=True,
        ).cuda(non_blocking=True)

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
        if len(temperatures) != len(sampled_seqs):
            raise ValueError("MTP temperatures do not match sampled requests")
        if len(generators) != len(sampled_seqs):
            raise ValueError("MTP generators do not match sampled requests")
        hidden_parts = []
        next_token_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        slot_mapping = []
        context_lens = []
        logits_indices = []
        sampled_next_positions = []
        offset = 0
        group_index = 0
        use_kv_cache = all(bool(seq.block_table) for seq in seqs)
        if any(bool(seq.block_table) for seq in seqs) != use_kv_cache:
            raise ValueError("MTP batch mixes cached and uncached sequences")

        for seq in seqs:
            scheduled = seq.num_scheduled_tokens
            sequence_hidden = target_hidden_states[offset : offset + scheduled]
            offset += scheduled
            if seq.will_sample:
                outputs = token_groups[group_index]
                accepted = accepted_counts[group_index]
                valid_inputs = 1 + accepted if seq.draft_token_ids else scheduled
                target_inputs = seq.scheduled_token_ids()[:valid_inputs]
                shifted_ids = target_inputs[1:] + [outputs[-1]]
                group_index += 1
            else:
                valid_inputs = scheduled
                start = seq.num_cached_tokens
                shifted_ids = seq.token_ids[start + 1 : start + valid_inputs + 1]
            if len(shifted_ids) != valid_inputs:
                raise ValueError(
                    f"cannot build shifted MTP inputs for sequence {seq.seq_id}"
                )

            start = seq.num_cached_tokens
            end = start + valid_inputs
            hidden_parts.append(sequence_hidden[:valid_inputs])
            next_token_ids.extend(shifted_ids)
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + valid_inputs)
            cu_seqlens_k.append(cu_seqlens_k[-1] + end)
            context_lens.append(end)
            if seq.will_sample:
                logits_indices.append(cu_seqlens_q[-1] - 1)
                sampled_next_positions.append(end)
            if use_kv_cache:
                slot_mapping.extend(self._slot_mapping(seq, start, end))

        num_tokens = len(next_token_ids)
        if num_tokens == 0:
            return []
        hidden_states = torch.cat(hidden_parts, dim=0)
        input_ids = torch.tensor(
            next_token_ids, dtype=torch.long, pin_memory=True
        ).cuda(non_blocking=True)
        positions_tensor = torch.tensor(
            positions, dtype=torch.long, pin_memory=True
        ).cuda(non_blocking=True)
        cu_q = torch.tensor(
            cu_seqlens_q, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        cu_k = torch.tensor(
            cu_seqlens_k, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        slots = torch.tensor(
            slot_mapping, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        context_lens_tensor = torch.tensor(
            context_lens, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        logits_indices_tensor = torch.tensor(
            logits_indices, dtype=torch.long, pin_memory=True
        ).cuda(non_blocking=True)
        block_tables = self._block_tables(seqs) if use_kv_cache else None
        query_lengths = [
            end - start
            for start, end in zip(cu_seqlens_q[:-1], cu_seqlens_q[1:])
        ]
        uniform_query_len = (
            query_lengths[0]
            if use_kv_cache
            and all(length == query_lengths[0] for length in query_lengths)
            else None
        )
        prepared = PreparedBatch(
            input_ids=input_ids,
            positions=positions_tensor,
            signature=ExecutionSignature(
                num_tokens=num_tokens,
                num_requests=len(seqs),
                num_padded_tokens=num_tokens,
                uniform_query_len=uniform_query_len,
            ),
            attention=AttentionMetadata(
                cu_seqlens_q=cu_q,
                cu_seqlens_k=cu_k,
                max_seqlen_q=max(query_lengths),
                max_seqlen_k=max(context_lens),
                slot_mapping=slots,
                block_tables=block_tables,
                context_lens=context_lens_tensor,
                use_kv_cache=use_kv_cache,
            ),
            sampling=SamplingMetadata(logits_indices=logits_indices_tensor),
        )
        with forward_context(prepared):
            embeddings = self.target_model.model.language_model.embed_tokens(
                input_ids
            )
            mtp_hidden = self.draft_model(
                positions_tensor,
                hidden_states,
                embeddings,
            )
            if not logits_indices:
                return []
            sampled_hidden = mtp_hidden.index_select(0, logits_indices_tensor)
            # The LM head consumes PreparedBatch.sampling.logits_indices and
            # selects the sampled rows from the full packed MTP output.
            logits = self.target_model.compute_logits(mtp_hidden)
            proposal_ids = torch.tensor(
                [
                    sample_from_logits(
                        row,
                        temperature,
                        generator=generator,
                    )
                    for row, temperature, generator in zip(
                        logits, temperatures, generators
                    )
                ],
                dtype=torch.long,
                device=logits.device,
            )

        draft_chains = [[token_id] for token_id in proposal_ids.tolist()]
        logit_steps = [logits]
        for _ in range(1, self.num_steps):
            proposal_ids, sampled_hidden, sampled_next_positions, logits = (
                self._recursive_step(
                    sampled_seqs,
                    proposal_ids,
                    sampled_hidden,
                    sampled_next_positions,
                    use_kv_cache,
                    temperatures,
                    generators,
                )
            )
            logit_steps.append(logits)
            for chain, token_id in zip(draft_chains, proposal_ids.tolist()):
                chain.append(token_id)
        return [
            DraftProposal(
                token_ids=chain,
                logits=(
                    torch.stack(
                        [step[index] for step in logit_steps], dim=0
                    ).detach()
                    if temperatures[index] > 0
                    else torch.empty(
                        0,
                        logits.shape[-1],
                        dtype=logits.dtype,
                        device=logits.device,
                    )
                ),
            )
            for index, chain in enumerate(draft_chains)
        ]

    def _recursive_step(
        self,
        sampled_seqs: list[Sequence],
        proposal_ids: torch.Tensor,
        sampled_hidden: torch.Tensor,
        sampled_next_positions: list[int],
        use_kv_cache: bool,
        temperatures: list[float],
        generators: list[torch.Generator | None],
    ) -> tuple[torch.Tensor, torch.Tensor, list[int], torch.Tensor]:
        batch_size = len(sampled_seqs)
        positions = torch.tensor(
            sampled_next_positions,
            dtype=torch.long,
            pin_memory=True,
        ).cuda(non_blocking=True)
        input_ids = proposal_ids.to(dtype=torch.long)
        cu_q = torch.arange(batch_size + 1, dtype=torch.int32, device="cuda")
        if use_kv_cache:
            cu_k_host = [0]
            slots = []
            context_lens = []
            for seq, position in zip(sampled_seqs, sampled_next_positions):
                cu_k_host.append(cu_k_host[-1] + position + 1)
                context_lens.append(position + 1)
                slots.extend(self._slot_mapping(seq, position, position + 1))
            cu_k = torch.tensor(
                cu_k_host, dtype=torch.int32, pin_memory=True
            ).cuda(non_blocking=True)
            slot_mapping = torch.tensor(
                slots, dtype=torch.int32, pin_memory=True
            ).cuda(non_blocking=True)
            context = torch.tensor(
                context_lens, dtype=torch.int32, pin_memory=True
            ).cuda(non_blocking=True)
            block_tables = self._block_tables(sampled_seqs)
        else:
            cu_k = cu_q
            slot_mapping = torch.empty(0, dtype=torch.int32, device="cuda")
            context = torch.ones(batch_size, dtype=torch.int32, device="cuda")
            block_tables = None
        logits_indices = torch.arange(batch_size, dtype=torch.long, device="cuda")
        prepared = PreparedBatch(
            input_ids=input_ids,
            positions=positions,
            signature=ExecutionSignature(
                num_tokens=batch_size,
                num_requests=batch_size,
                num_padded_tokens=batch_size,
                uniform_query_len=1 if use_kv_cache else None,
            ),
            attention=AttentionMetadata(
                cu_seqlens_q=cu_q,
                cu_seqlens_k=cu_k,
                max_seqlen_q=1,
                max_seqlen_k=(
                    max(sampled_next_positions) + 1 if use_kv_cache else 1
                ),
                slot_mapping=slot_mapping,
                block_tables=block_tables,
                context_lens=context,
                use_kv_cache=use_kv_cache,
            ),
            sampling=SamplingMetadata(logits_indices=logits_indices),
        )
        with forward_context(prepared):
            embeddings = self.target_model.model.language_model.embed_tokens(
                input_ids
            )
            sampled_hidden = self.draft_model(
                positions,
                sampled_hidden,
                embeddings,
            )
            logits = self.target_model.compute_logits(sampled_hidden)
            proposal_ids = torch.tensor(
                [
                    sample_from_logits(
                        row,
                        temperature,
                        generator=generator,
                    )
                    for row, temperature, generator in zip(
                        logits, temperatures, generators
                    )
                ],
                dtype=torch.long,
                device=logits.device,
            )
        return (
            proposal_ids,
            sampled_hidden,
            [position + 1 for position in sampled_next_positions],
            logits,
        )
