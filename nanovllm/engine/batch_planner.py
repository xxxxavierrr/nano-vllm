from __future__ import annotations

from dataclasses import dataclass, field

import torch

from nanovllm.engine.batch import (
    AttentionMetadata,
    GDNMetadata,
    PreparedBatch,
    SamplingMetadata,
)
from nanovllm.engine.cudagraph import BatchDescriptor, ExecutionMode
from nanovllm.engine.hybrid_state import HybridStateManager
from nanovllm.engine.sequence import Sequence
from nanovllm.layers.deltanet_chunk import DELTA_CHUNK_MIN_TOKENS, DELTA_CHUNK_SIZE
from nanovllm.layers.fp8_attention import FP8_QUERY_TILE_SIZE


@dataclass(slots=True)
class _HostBatch:
    input_ids: list[int] = field(default_factory=list)
    positions: list[int] = field(default_factory=list)
    cu_q: list[int] = field(default_factory=lambda: [0])
    cu_k: list[int] = field(default_factory=lambda: [0])
    slot_mapping: list[int] = field(default_factory=list)
    logits_indices: list[int] = field(default_factory=list)
    context_lens: list[int] = field(default_factory=list)
    tile_seq_ids: list[int] = field(default_factory=list)
    tile_starts: list[int] = field(default_factory=list)
    tile_lens: list[int] = field(default_factory=list)
    tile_positions: list[int] = field(default_factory=list)
    chunk_indices: list[tuple[int, int]] = field(default_factory=list)
    cu_chunks: list[int] = field(default_factory=lambda: [0])
    chunk_sequences: list[int] = field(default_factory=list)
    recurrent_sequences: list[int] = field(default_factory=list)
    max_q: int = 0
    max_k: int = 0


def _cuda_tensor(values, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(values, dtype=dtype, pin_memory=True).cuda(non_blocking=True)


class BatchPlanner:
    """Translate scheduler-owned sequences into one typed model batch."""

    def __init__(
        self,
        *,
        block_size: int,
        use_fp8_kv: bool,
        hybrid_state: HybridStateManager,
    ):
        self.block_size = block_size
        self.use_fp8_kv = use_fp8_kv
        self.hybrid_state = hybrid_state

    @staticmethod
    def _uses_kv_cache(seqs: list[Sequence]) -> bool:
        cached = [bool(seq.block_table) for seq in seqs]
        if any(cached) and not all(cached):
            raise ValueError(
                "all sequences in a model batch must either use KV cache "
                "or be warmup sequences"
            )
        return all(cached)

    def _append_slots(
        self,
        host: _HostBatch,
        seq: Sequence,
        start: int,
        end: int,
    ) -> None:
        start_block = start // self.block_size
        end_block = (end + self.block_size - 1) // self.block_size
        for index in range(start_block, end_block):
            block_start = seq.block_table[index] * self.block_size
            slot_start = block_start + (start % self.block_size if index == start_block else 0)
            slot_end = block_start + self.block_size
            if index == end_block - 1:
                slot_end = block_start + end - index * self.block_size
            host.slot_mapping.extend(range(slot_start, slot_end))

    def _append_tiles(
        self,
        host: _HostBatch,
        sequence: int,
        query_start: int,
        position: int,
        query_len: int,
    ) -> None:
        if not self.use_fp8_kv:
            return
        for offset in range(0, query_len, FP8_QUERY_TILE_SIZE):
            host.tile_seq_ids.append(sequence)
            host.tile_starts.append(query_start + offset)
            host.tile_lens.append(min(FP8_QUERY_TILE_SIZE, query_len - offset))
            host.tile_positions.append(position + offset)

    @staticmethod
    def _append_gdn_partition(host: _HostBatch, sequence: int, query_len: int) -> None:
        if query_len < DELTA_CHUNK_MIN_TOKENS:
            host.recurrent_sequences.append(sequence)
            return
        host.chunk_sequences.append(sequence)
        chunks = (query_len + DELTA_CHUNK_SIZE - 1) // DELTA_CHUNK_SIZE
        host.chunk_indices.extend((sequence, chunk) for chunk in range(chunks))
        host.cu_chunks.append(host.cu_chunks[-1] + chunks)

    def _append_sequence(
        self,
        host: _HostBatch,
        seq: Sequence,
        sequence: int,
        use_kv_cache: bool,
    ) -> None:
        start = seq.num_cached_tokens
        query_len = seq.num_scheduled_tokens
        if query_len <= 0:
            raise ValueError(f"sequence {getattr(seq, 'seq_id', 'unknown')} has no scheduled tokens")
        end = start + query_len
        query_start = host.cu_q[-1]
        if query_len == 1 and not seq.draft_token_ids:
            host.input_ids.append(seq.last_token)
        else:
            host.input_ids.extend(seq.scheduled_token_ids())
        host.positions.extend(range(start, end))
        host.cu_q.append(query_start + query_len)
        host.cu_k.append(host.cu_k[-1] + end)
        host.context_lens.append(end)
        host.max_q = max(host.max_q, query_len)
        host.max_k = max(host.max_k, end)
        self._append_tiles(host, sequence, query_start, start, query_len)
        self._append_gdn_partition(host, sequence, query_len)
        if seq.will_sample:
            indices = range(query_start, query_start + query_len)
            host.logits_indices.extend(indices if seq.draft_token_ids else [query_start + query_len - 1])
        if use_kv_cache:
            self._append_slots(host, seq, start, end)

    def _pad_inputs(self, host: _HostBatch, descriptor: BatchDescriptor) -> None:
        actual = len(host.input_ids)
        if actual != descriptor.num_tokens:
            raise ValueError(
                f"dispatcher expected {descriptor.num_tokens} tokens, prepared {actual}"
            )
        model_tokens = (
            descriptor.num_padded_tokens
            if descriptor.execution_mode is ExecutionMode.PIECEWISE
            else actual
        )
        padding = model_tokens - actual
        if padding < 0:
            raise ValueError("CUDA Graph bucket is smaller than the real token batch")
        host.input_ids.extend([0] * padding)
        host.positions.extend([0] * padding)

    def _block_tables(self, seqs: list[Sequence]) -> torch.Tensor:
        width = max(len(seq.block_table) for seq in seqs)
        rows = [seq.block_table + [-1] * (width - len(seq.block_table)) for seq in seqs]
        return _cuda_tensor(rows, torch.int32)

    def _attention(
        self,
        host: _HostBatch,
        seqs: list[Sequence],
        use_kv_cache: bool,
    ) -> AttentionMetadata:
        tiles = (None, None, None, None)
        if self.use_fp8_kv:
            tiles = tuple(
                _cuda_tensor(values, torch.int32)
                for values in (
                    host.tile_seq_ids, host.tile_starts,
                    host.tile_lens, host.tile_positions,
                )
            )
        return AttentionMetadata(
            cu_seqlens_q=_cuda_tensor(host.cu_q, torch.int32),
            cu_seqlens_k=_cuda_tensor(host.cu_k, torch.int32),
            max_seqlen_q=host.max_q,
            max_seqlen_k=host.max_k,
            slot_mapping=_cuda_tensor(host.slot_mapping, torch.int32),
            block_tables=self._block_tables(seqs) if use_kv_cache else None,
            context_lens=_cuda_tensor(host.context_lens, torch.int32),
            query_tile_seq_ids=tiles[0],
            query_tile_starts=tiles[1],
            query_tile_lens=tiles[2],
            query_tile_positions=tiles[3],
            use_kv_cache=use_kv_cache,
        )

    def _gdn(
        self,
        host: _HostBatch,
        seqs: list[Sequence],
        cu_q: torch.Tensor,
    ) -> GDNMetadata | None:
        if not self.hybrid_state.enabled:
            return None
        seq_ids = tuple(seq.seq_id for seq in seqs)
        state_view = self.hybrid_state.batch_view(seq_ids)
        if state_view is None:
            raise RuntimeError("hybrid state manager returned no state view")
        conv_slab, recurrent_slab, state_slots = state_view
        branch_width = host.max_q if self.hybrid_state.branches else 1
        return GDNMetadata(
            cu_seqlens=cu_q,
            conv_slab=conv_slab,
            recurrent_slab=recurrent_slab,
            state_slots=state_slots,
            branch_state_slots=self.hybrid_state.branch_slots_view(seq_ids, branch_width),
            chunk_indices=_cuda_tensor(host.chunk_indices, torch.int32).reshape(-1, 2),
            cu_chunks=_cuda_tensor(host.cu_chunks, torch.int32),
            chunk_sequences=_cuda_tensor(host.chunk_sequences, torch.int32),
            recurrent_sequences=_cuda_tensor(host.recurrent_sequences, torch.int32),
        )

    @staticmethod
    def _validate_signature(
        seqs: list[Sequence],
        descriptor: BatchDescriptor,
        use_kv_cache: bool,
    ) -> None:
        lengths = [seq.num_scheduled_tokens for seq in seqs]
        uniform = (
            lengths[0]
            if use_kv_cache
            and all(not seq.is_prefill for seq in seqs)
            and all(length == lengths[0] for length in lengths)
            else None
        )
        if uniform != descriptor.uniform_query_len:
            raise ValueError(
                "dispatcher and prepared attention metadata disagree on uniform query length"
            )

    def prepare(
        self,
        seqs: list[Sequence],
        descriptor: BatchDescriptor,
    ) -> PreparedBatch:
        use_kv_cache = self._uses_kv_cache(seqs)
        host = _HostBatch()
        for sequence, seq in enumerate(seqs):
            self._append_sequence(host, seq, sequence, use_kv_cache)
        self._pad_inputs(host, descriptor)
        self._validate_signature(seqs, descriptor, use_kv_cache)
        attention = self._attention(host, seqs, use_kv_cache)
        gdn = self._gdn(host, seqs, attention.cu_seqlens_q)
        return PreparedBatch(
            input_ids=_cuda_tensor(host.input_ids, torch.int64),
            positions=_cuda_tensor(host.positions, torch.int64),
            signature=descriptor.signature,
            attention=attention,
            sampling=SamplingMetadata(
                logits_indices=_cuda_tensor(host.logits_indices, torch.int64)
            ),
            gdn=gdn,
        )
