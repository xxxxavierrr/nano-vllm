"""Chunked Gated DeltaNet numerical backend.

The public chunk operator mirrors the boundary used by vLLM/FLA: projected
and normalized ``q/k/v/gating`` tensors plus an initial recurrent state go in;
attention output is returned and the final state is committed in place.

This file deliberately does not know about requests, scheduler policy,
prefill/decode labels, convolution state, or model projections.  Packed mixed
execution is a thin composition of the recurrent backend from ``deltanet``
and the chunk backend below; the runner supplies the partition metadata.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from nanovllm.layers.deltanet import (
    _launch_recurrent,
    _validate_delta_tensors,
    _validate_packed_metadata,
)


DELTA_CHUNK_SIZE = 32
DELTA_CHUNK_MIN_TOKENS = 512


@triton.jit
def _prepare_delta_chunk_kernel(
    q_ptr,
    k_ptr,
    value_ptr,
    beta_ptr,
    decay_ptr,
    cu_seqlens_ptr,
    chunk_indices_ptr,
    w_ptr,
    u_ptr,
    output_weights_ptr,
    cumulative_decay_ptr,
    final_decay_ratio_ptr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    CHUNK: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Build the per-chunk triangular solve and transformed K/V tensors."""
    packed_chunk = tl.program_id(axis=0)
    head = tl.program_id(axis=1)
    sequence = tl.load(chunk_indices_ptr + packed_chunk * 2)
    local_chunk = tl.load(chunk_indices_ptr + packed_chunk * 2 + 1)
    begin = tl.load(cu_seqlens_ptr + sequence)
    end = tl.load(cu_seqlens_ptr + sequence + 1)

    offs_t = tl.arange(0, CHUNK)
    offs_k = tl.arange(0, BLOCK_K)
    tokens = begin + local_chunk * CHUNK + offs_t
    token_mask = tokens < end

    q = tl.load(
        q_ptr + (tokens[:, None] * H + head) * K + offs_k[None, :],
        mask=token_mask[:, None] & (offs_k[None, :] < K),
        other=0.0,
    ).to(tl.float32)
    k = tl.load(
        k_ptr + (tokens[:, None] * H + head) * K + offs_k[None, :],
        mask=token_mask[:, None] & (offs_k[None, :] < K),
        other=0.0,
    ).to(tl.float32)
    beta = tl.load(
        beta_ptr + tokens * H + head,
        mask=token_mask,
        other=0.0,
    ).to(tl.float32)
    # Padding has neutral decay so a partial final chunk has the same final
    # state as the equivalent recurrent sequence.
    decay = tl.load(
        decay_ptr + tokens * H + head,
        mask=token_mask,
        other=1.0,
    ).to(tl.float32)

    log_cumulative = tl.cumsum(tl.log(decay), axis=0)
    cumulative = tl.exp(log_cumulative)
    decay_ratio = tl.exp(log_cumulative[:, None] - log_cumulative[None, :])

    rows = offs_t[:, None]
    cols = offs_t[None, :]
    gram = tl.dot(k, tl.trans(k), input_precision="tf32")
    lower = tl.where(
        rows > cols,
        gram * decay_ratio * beta[:, None],
        0.0,
    )

    # Solve (I + strictly_lower)^-1.  Keeping this stage explicit makes the
    # data flow match the FLA decomposition (KKT -> triangular solve -> W/U).
    inverse = tl.where(rows == cols, 1.0, 0.0)
    for row in range(0, CHUNK):
        lower_row = tl.sum(
            tl.where((offs_t == row)[:, None], lower, 0.0),
            axis=0,
        )
        correction = tl.sum(lower_row[:, None] * inverse, axis=0)
        target = tl.where(offs_t == row, 1.0, 0.0)
        inverse_row = target - correction
        inverse = tl.where(
            (offs_t == row)[:, None],
            inverse_row[None, :],
            inverse,
        )

    qk = tl.dot(q, tl.trans(k), input_precision="tf32")
    output_weights = tl.where(rows >= cols, qk * decay_ratio, 0.0)
    matrix_base = (packed_chunk * H + head) * CHUNK * CHUNK
    tl.store(
        output_weights_ptr + matrix_base + rows * CHUNK + cols,
        output_weights,
    )

    vector_base = (packed_chunk * H + head) * CHUNK
    tl.store(cumulative_decay_ptr + vector_base + offs_t, cumulative)
    final_log_cumulative = tl.sum(
        tl.where(offs_t == CHUNK - 1, log_cumulative, 0.0),
        axis=0,
    )
    final_ratio = tl.exp(final_log_cumulative - log_cumulative)
    tl.store(final_decay_ratio_ptr + vector_base + offs_t, final_ratio)

    for block in range(0, tl.cdiv(K, BLOCK_D)):
        offs_d = block * BLOCK_D + tl.arange(0, BLOCK_D)
        k_block = tl.load(
            k_ptr + (tokens[:, None] * H + head) * K + offs_d[None, :],
            mask=token_mask[:, None] & (offs_d[None, :] < K),
            other=0.0,
        ).to(tl.float32)
        weighted_k = k_block * (beta * cumulative)[:, None]
        w = tl.dot(inverse, weighted_k, input_precision="tf32")
        tl.store(
            w_ptr + (tokens[:, None] * H + head) * K + offs_d[None, :],
            w,
            mask=token_mask[:, None] & (offs_d[None, :] < K),
        )

    for block in range(0, tl.cdiv(V, BLOCK_D)):
        offs_d = block * BLOCK_D + tl.arange(0, BLOCK_D)
        value = tl.load(
            value_ptr + (tokens[:, None] * H + head) * V + offs_d[None, :],
            mask=token_mask[:, None] & (offs_d[None, :] < V),
            other=0.0,
        ).to(tl.float32)
        u = tl.dot(inverse, value * beta[:, None], input_precision="tf32")
        tl.store(
            u_ptr + (tokens[:, None] * H + head) * V + offs_d[None, :],
            u,
            mask=token_mask[:, None] & (offs_d[None, :] < V),
        )


@triton.jit
def _apply_delta_chunk_kernel(
    q_ptr,
    k_ptr,
    w_ptr,
    u_ptr,
    cu_seqlens_ptr,
    chunk_indices_ptr,
    cu_chunks_ptr,
    chunk_sequences_ptr,
    slots_ptr,
    output_weights_ptr,
    cumulative_decay_ptr,
    final_decay_ratio_ptr,
    state_ptr,
    output_ptr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    CHUNK: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """Apply prepared chunks in sequence order and commit the final state."""
    value_block = tl.program_id(axis=0)
    packed_sequence_head = tl.program_id(axis=1)
    packed_sequence = packed_sequence_head // H
    head = packed_sequence_head % H
    sequence = tl.load(chunk_sequences_ptr + packed_sequence)
    begin = tl.load(cu_seqlens_ptr + sequence)
    end = tl.load(cu_seqlens_ptr + sequence + 1)
    chunk_begin = tl.load(cu_chunks_ptr + packed_sequence)
    chunk_end = tl.load(cu_chunks_ptr + packed_sequence + 1)
    slot = tl.load(slots_ptr + sequence)

    offs_t = tl.arange(0, CHUNK)
    offs_k = tl.arange(0, BLOCK_K)
    offs_v = value_block * BLOCK_V + tl.arange(0, BLOCK_V)
    state_offsets = (
        (slot * H + head) * K * V
        + offs_k[:, None] * V
        + offs_v[None, :]
    )
    state = tl.load(
        state_ptr + state_offsets,
        mask=(offs_k[:, None] < K) & (offs_v[None, :] < V),
        other=0.0,
    ).to(tl.float32)

    packed_chunk = chunk_begin
    while packed_chunk < chunk_end:
        local_chunk = tl.load(chunk_indices_ptr + packed_chunk * 2 + 1)
        tokens = begin + local_chunk * CHUNK + offs_t
        token_mask = tokens < end

        q = tl.load(
            q_ptr + (tokens[:, None] * H + head) * K + offs_k[None, :],
            mask=token_mask[:, None] & (offs_k[None, :] < K),
            other=0.0,
        ).to(tl.float32)
        k = tl.load(
            k_ptr + (tokens[:, None] * H + head) * K + offs_k[None, :],
            mask=token_mask[:, None] & (offs_k[None, :] < K),
            other=0.0,
        ).to(tl.float32)
        w = tl.load(
            w_ptr + (tokens[:, None] * H + head) * K + offs_k[None, :],
            mask=token_mask[:, None] & (offs_k[None, :] < K),
            other=0.0,
        ).to(tl.float32)
        u = tl.load(
            u_ptr + (tokens[:, None] * H + head) * V + offs_v[None, :],
            mask=token_mask[:, None] & (offs_v[None, :] < V),
            other=0.0,
        ).to(tl.float32)

        vector_base = (packed_chunk * H + head) * CHUNK
        cumulative = tl.load(cumulative_decay_ptr + vector_base + offs_t)
        final_ratio = tl.load(final_decay_ratio_ptr + vector_base + offs_t)
        v_new = u - tl.dot(w, state, input_precision="tf32")

        matrix_base = (packed_chunk * H + head) * CHUNK * CHUNK
        output_weights = tl.load(
            output_weights_ptr
            + matrix_base
            + offs_t[:, None] * CHUNK
            + offs_t[None, :]
        )
        initial_output = tl.dot(q, state, input_precision="tf32")
        local_output = tl.dot(output_weights, v_new, input_precision="tf32")
        output = cumulative[:, None] * initial_output + local_output
        tl.store(
            output_ptr + (tokens[:, None] * H + head) * V + offs_v[None, :],
            output,
            mask=token_mask[:, None] & (offs_v[None, :] < V),
        )

        state_update = tl.dot(
            tl.trans(k),
            v_new * final_ratio[:, None],
            input_precision="tf32",
        )
        final_cumulative = tl.sum(
            tl.where(offs_t == CHUNK - 1, cumulative, 0.0),
            axis=0,
        )
        state = final_cumulative * state + state_update
        packed_chunk += 1

    tl.store(
        state_ptr + state_offsets,
        state,
        mask=(offs_k[:, None] < K) & (offs_v[None, :] < V),
    )


def _allocate_chunk_workspace(
    q: torch.Tensor,
    k: torch.Tensor,
    value: torch.Tensor,
    num_chunks: int,
    heads: int,
) -> tuple[torch.Tensor, ...]:
    w = torch.empty_like(k)
    u = torch.empty_like(value)
    output_weights = torch.empty(
        num_chunks,
        heads,
        DELTA_CHUNK_SIZE,
        DELTA_CHUNK_SIZE,
        device=q.device,
        dtype=torch.float32,
    )
    cumulative_decay = torch.empty(
        num_chunks,
        heads,
        DELTA_CHUNK_SIZE,
        device=q.device,
        dtype=torch.float32,
    )
    final_decay_ratio = torch.empty_like(cumulative_decay)
    return w, u, output_weights, cumulative_decay, final_decay_ratio


def _launch_chunk(
    q: torch.Tensor,
    k: torch.Tensor,
    value: torch.Tensor,
    beta: torch.Tensor,
    decay: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
    cu_chunks: torch.Tensor,
    chunk_sequences: torch.Tensor,
    slots: torch.Tensor,
    state: torch.Tensor,
    output: torch.Tensor,
) -> None:
    _, heads, key_dim = q.shape
    value_dim = value.shape[-1]
    num_chunks = chunk_indices.shape[0]
    num_sequences = chunk_sequences.numel()
    if num_chunks == 0 or num_sequences == 0:
        return

    w, u, output_weights, cumulative_decay, final_decay_ratio = (
        _allocate_chunk_workspace(q, k, value, num_chunks, heads)
    )
    block_k = triton.next_power_of_2(key_dim)
    _prepare_delta_chunk_kernel[(num_chunks, heads)](
        q,
        k,
        value,
        beta,
        decay,
        cu_seqlens,
        chunk_indices,
        w,
        u,
        output_weights,
        cumulative_decay,
        final_decay_ratio,
        H=heads,
        K=key_dim,
        V=value_dim,
        CHUNK=DELTA_CHUNK_SIZE,
        BLOCK_K=block_k,
        BLOCK_D=64,
        num_warps=4,
        num_stages=2,
    )
    grid = (triton.cdiv(value_dim, 32), num_sequences * heads)
    _apply_delta_chunk_kernel[grid](
        q,
        k,
        w,
        u,
        cu_seqlens,
        chunk_indices,
        cu_chunks,
        chunk_sequences,
        slots,
        output_weights,
        cumulative_decay,
        final_decay_ratio,
        state,
        output,
        H=heads,
        K=key_dim,
        V=value_dim,
        CHUNK=DELTA_CHUNK_SIZE,
        BLOCK_K=block_k,
        BLOCK_V=32,
        num_warps=2,
        num_stages=1,
    )


def _validate_packed_partition(
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
    cu_chunks: torch.Tensor,
    chunk_sequences: torch.Tensor,
    recurrent_sequences: torch.Tensor,
    slots: torch.Tensor,
) -> None:
    _validate_packed_metadata(
        cu_seqlens,
        slots,
        operation="packed DeltaNet",
    )
    metadata = (
        chunk_indices,
        cu_chunks,
        chunk_sequences,
        recurrent_sequences,
    )
    if not all(tensor.dtype is torch.int32 for tensor in metadata):
        raise TypeError("packed DeltaNet metadata must be int32")
    if not all(
        tensor.is_cuda and tensor.is_contiguous() for tensor in metadata
    ):
        raise ValueError("packed DeltaNet metadata must be contiguous CUDA tensors")
    if chunk_indices.ndim != 2 or chunk_indices.shape[1] != 2:
        raise ValueError("chunk_indices must have shape [num_chunks, 2]")
    if any(
        tensor.ndim != 1
        for tensor in (cu_chunks, chunk_sequences, recurrent_sequences)
    ):
        raise ValueError("packed DeltaNet sequence metadata must be rank-1")
    if cu_chunks.numel() != chunk_sequences.numel() + 1:
        raise ValueError("cu_chunks must match indexed chunk sequences")
    if chunk_sequences.numel() + recurrent_sequences.numel() != slots.numel():
        raise ValueError("packed partition must cover every sequence")


def gated_delta_packed(
    q: torch.Tensor,
    k: torch.Tensor,
    value: torch.Tensor,
    beta: torch.Tensor,
    decay: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
    cu_chunks: torch.Tensor,
    chunk_sequences: torch.Tensor,
    recurrent_sequences: torch.Tensor,
    slots: torch.Tensor,
    state_slab: torch.Tensor,
) -> torch.Tensor:
    """Execute a model-supplied recurrent/chunk partition of a packed batch."""
    q = q.contiguous()
    k = k.contiguous()
    value = value.contiguous()
    beta = beta.contiguous()
    decay = decay.contiguous()
    cu_seqlens = cu_seqlens.contiguous()
    chunk_indices = chunk_indices.contiguous()
    cu_chunks = cu_chunks.contiguous()
    chunk_sequences = chunk_sequences.contiguous()
    recurrent_sequences = recurrent_sequences.contiguous()
    slots = slots.contiguous()
    _validate_delta_tensors(
        q,
        k,
        value,
        beta,
        decay,
        state_slab=state_slab,
        operation="packed DeltaNet kernel",
    )
    _validate_packed_partition(
        cu_seqlens,
        chunk_indices,
        cu_chunks,
        chunk_sequences,
        recurrent_sequences,
        slots,
    )

    output = torch.empty_like(value)
    _launch_recurrent(
        q,
        k,
        value,
        beta,
        decay,
        cu_seqlens,
        recurrent_sequences,
        slots,
        state_slab,
        output,
    )
    _launch_chunk(
        q,
        k,
        value,
        beta,
        decay,
        cu_seqlens,
        chunk_indices,
        cu_chunks,
        chunk_sequences,
        slots,
        state_slab,
        output,
    )
    return output
