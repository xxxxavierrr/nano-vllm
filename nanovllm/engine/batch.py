"""Typed rank-local model batch metadata.

The scheduler owns request/token policy.  These records own the tensor metadata
for one model invocation and are intentionally independent of model family and
CUDA Graph implementation classes.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class ExecutionSignature:
    """Semantic shape used to select an execution strategy."""

    num_tokens: int
    num_requests: int
    num_padded_tokens: int
    uniform_query_len: int | None

    def __post_init__(self) -> None:
        if self.num_tokens <= 0 or self.num_requests <= 0:
            raise ValueError("an execution signature requires tokens and requests")
        if self.num_padded_tokens < self.num_tokens:
            raise ValueError("padded token count cannot be smaller than real tokens")
        if self.uniform_query_len is not None:
            if self.uniform_query_len <= 0:
                raise ValueError("uniform query length must be positive")
            if self.num_tokens != self.num_requests * self.uniform_query_len:
                raise ValueError(
                    "uniform query length does not match token/request counts"
                )

    @property
    def is_single_token_decode(self) -> bool:
        return self.uniform_query_len == 1


@dataclass(frozen=True, slots=True)
class AttentionMetadata:
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    query_tile_seq_ids: torch.Tensor | None = None
    query_tile_starts: torch.Tensor | None = None
    query_tile_lens: torch.Tensor | None = None
    query_tile_positions: torch.Tensor | None = None
    use_kv_cache: bool = False


@dataclass(frozen=True, slots=True)
class GDNMetadata:
    cu_seqlens: torch.Tensor
    conv_slab: torch.Tensor
    recurrent_slab: torch.Tensor
    state_slots: torch.Tensor
    chunk_indices: torch.Tensor
    cu_chunks: torch.Tensor
    chunk_sequences: torch.Tensor
    recurrent_sequences: torch.Tensor


@dataclass(frozen=True, slots=True)
class SamplingMetadata:
    logits_indices: torch.Tensor | None = None


@dataclass(frozen=True, slots=True)
class PreparedBatch:
    input_ids: torch.Tensor
    positions: torch.Tensor
    signature: ExecutionSignature
    attention: AttentionMetadata
    sampling: SamplingMetadata
    gdn: GDNMetadata | None = None

    def __post_init__(self) -> None:
        tensor_tokens = self.input_ids.size(0)
        if self.positions.size(0) != tensor_tokens:
            raise ValueError("prepared input and position tensors disagree")
        if tensor_tokens not in (
            self.signature.num_tokens,
            self.signature.num_padded_tokens,
        ):
            raise ValueError("prepared tensors do not match real or padded size")
