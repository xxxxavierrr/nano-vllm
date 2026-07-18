from dataclasses import dataclass

import torch


@dataclass(slots=True)
class Context:
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    logits_indices: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    use_kv_cache: bool = False
    is_uniform_decode: bool = False
    num_actual_tokens: int = 0
    num_padded_tokens: int = 0


_CONTEXT = Context()


def get_context():
    return _CONTEXT


def set_context(
    *,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=0,
    max_seqlen_k=0,
    slot_mapping=None,
    block_tables=None,
    logits_indices=None,
    context_lens=None,
    use_kv_cache=False,
    is_uniform_decode=False,
    num_actual_tokens=0,
    num_padded_tokens=0,
):
    global _CONTEXT
    _CONTEXT = Context(
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        slot_mapping=slot_mapping,
        block_tables=block_tables,
        logits_indices=logits_indices,
        context_lens=context_lens,
        use_kv_cache=use_kv_cache,
        is_uniform_decode=is_uniform_decode,
        num_actual_tokens=num_actual_tokens,
        num_padded_tokens=num_padded_tokens,
    )


def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
