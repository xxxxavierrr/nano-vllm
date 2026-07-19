from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton


_CHUNK_SIZE = 32


@triton.jit(do_not_specialize=["T"])
def _prepare_delta_chunk_kernel(
    q_ptr,
    k_ptr,
    value_ptr,
    beta_ptr,
    decay_ptr,
    w_ptr,
    u_ptr,
    output_weights_ptr,
    cumulative_decay_ptr,
    final_decay_ratio_ptr,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    CHUNK: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    chunk = tl.program_id(axis=0)
    head = tl.program_id(axis=1)
    offs_t = tl.arange(0, CHUNK)
    offs_k = tl.arange(0, BLOCK_K)
    tokens = chunk * CHUNK + offs_t
    token_mask = tokens < T

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
    decay = tl.load(
        decay_ptr + tokens * H + head,
        mask=token_mask,
        other=1.0,
    ).to(tl.float32)

    log_cumulative = tl.cumsum(tl.log(decay), axis=0)
    cumulative = tl.exp(log_cumulative)
    decay_ratio = tl.exp(
        log_cumulative[:, None] - log_cumulative[None, :]
    )
    gram = tl.dot(k, tl.trans(k), input_precision="tf32")
    qk = tl.dot(q, tl.trans(k), input_precision="tf32")
    rows = offs_t[:, None]
    cols = offs_t[None, :]
    lower = tl.where(
        rows > cols,
        gram * decay_ratio * beta[:, None],
        0.0,
    )
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

    output_weights = tl.where(
        rows >= cols,
        qk * decay_ratio,
        0.0,
    )
    matrix_base = (chunk * H + head) * CHUNK * CHUNK
    matrix_offsets = rows * CHUNK + cols
    tl.store(
        output_weights_ptr + matrix_base + matrix_offsets,
        output_weights,
    )

    vector_base = (chunk * H + head) * CHUNK
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
        weighted_value = value * beta[:, None]
        u = tl.dot(inverse, weighted_value, input_precision="tf32")
        tl.store(
            u_ptr + (tokens[:, None] * H + head) * V + offs_d[None, :],
            u,
            mask=token_mask[:, None] & (offs_d[None, :] < V),
        )


@triton.jit(do_not_specialize=["T", "chunks"])
def _apply_delta_chunk_kernel(
    q_ptr,
    k_ptr,
    w_ptr,
    u_ptr,
    output_weights_ptr,
    cumulative_decay_ptr,
    final_decay_ratio_ptr,
    state_ptr,
    output_ptr,
    T,
    chunks,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    CHUNK: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    value_block = tl.program_id(axis=0)
    head = tl.program_id(axis=1)
    offs_t = tl.arange(0, CHUNK)
    offs_k = tl.arange(0, BLOCK_K)
    offs_v = value_block * BLOCK_V + tl.arange(0, BLOCK_V)

    state_offsets = head * K * V + offs_k[:, None] * V + offs_v[None, :]
    state = tl.load(
        state_ptr + state_offsets,
        mask=(offs_k[:, None] < K) & (offs_v[None, :] < V),
        other=0.0,
    ).to(tl.float32)

    for chunk in tl.range(0, chunks):
        tokens = chunk * CHUNK + offs_t
        token_mask = tokens < T
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

        vector_base = (chunk * H + head) * CHUNK
        cumulative = tl.load(cumulative_decay_ptr + vector_base + offs_t)
        final_ratio = tl.load(final_decay_ratio_ptr + vector_base + offs_t)
        v_new = u - tl.dot(w, state, input_precision="tf32")

        matrix_base = (chunk * H + head) * CHUNK * CHUNK
        output_weights = tl.load(
            output_weights_ptr
            + matrix_base
            + offs_t[:, None] * CHUNK
            + offs_t[None, :]
        )
        initial_output = tl.dot(q, state, input_precision="tf32")
        local_output = tl.dot(
            output_weights,
            v_new,
            input_precision="tf32",
        )
        output = cumulative[:, None] * initial_output + local_output
        tl.store(
            output_ptr + (tokens[:, None] * H + head) * V + offs_v[None, :],
            output,
            mask=token_mask[:, None] & (offs_v[None, :] < V),
        )

        weighted_delta = v_new * final_ratio[:, None]
        state_update = tl.dot(
            tl.trans(k),
            weighted_delta,
            input_precision="tf32",
        )
        final_cumulative = tl.sum(
            tl.where(offs_t == CHUNK - 1, cumulative, 0.0),
            axis=0,
        )
        state = final_cumulative * state + state_update

    tl.store(
        state_ptr + state_offsets,
        state,
        mask=(offs_k[:, None] < K) & (offs_v[None, :] < V),
    )


def _validate_chunk_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    value: torch.Tensor,
    beta: torch.Tensor,
    decay: torch.Tensor,
    state: torch.Tensor,
) -> tuple[int, int, int, int]:
    tensors = (q, k, value, beta, decay, state)
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("DeltaNet chunk kernel requires CUDA tensors")
    if not all(tensor.dtype is torch.float32 for tensor in tensors):
        raise TypeError("DeltaNet chunk kernel requires float32 tensors")
    if q.ndim != 3 or k.shape != q.shape or value.ndim != 3:
        raise ValueError("q/k/value must have shapes [T, H, K/V]")
    tokens, heads, key_dim = q.shape
    value_dim = value.shape[-1]
    if value.shape[:2] != (tokens, heads):
        raise ValueError("value T/H dimensions must match q/k")
    if beta.shape != (tokens, heads) or decay.shape != (tokens, heads):
        raise ValueError("beta and decay must have shape [T, H]")
    if state.shape != (heads, key_dim, value_dim):
        raise ValueError("state must have shape [H, K, V]")
    if not all(tensor.is_contiguous() for tensor in tensors):
        raise ValueError("DeltaNet chunk tensors must be contiguous")
    return tokens, heads, key_dim, value_dim


@triton_op("nanovllm::gated_delta_chunk", mutates_args={"state"})
def _gated_delta_chunk(
    q: torch.Tensor,
    k: torch.Tensor,
    value: torch.Tensor,
    beta: torch.Tensor,
    decay: torch.Tensor,
    state: torch.Tensor,
) -> torch.Tensor:
    tokens, heads, key_dim, value_dim = _validate_chunk_inputs(
        q, k, value, beta, decay, state
    )
    chunks = triton.cdiv(tokens, _CHUNK_SIZE)
    w = torch.empty_like(k)
    u = torch.empty_like(value)
    output_weights = torch.empty(
        chunks,
        heads,
        _CHUNK_SIZE,
        _CHUNK_SIZE,
        device=q.device,
        dtype=torch.float32,
    )
    cumulative_decay = torch.empty(
        chunks,
        heads,
        _CHUNK_SIZE,
        device=q.device,
        dtype=torch.float32,
    )
    final_decay_ratio = torch.empty_like(cumulative_decay)
    output = torch.empty_like(value)
    block_k = triton.next_power_of_2(key_dim)

    wrap_triton(_prepare_delta_chunk_kernel)[(chunks, heads)](
        q,
        k,
        value,
        beta,
        decay,
        w,
        u,
        output_weights,
        cumulative_decay,
        final_decay_ratio,
        T=tokens,
        H=heads,
        K=key_dim,
        V=value_dim,
        CHUNK=_CHUNK_SIZE,
        BLOCK_K=block_k,
        BLOCK_D=64,
        num_warps=4,
        num_stages=2,
    )
    grid = (triton.cdiv(value_dim, 32), heads)
    wrap_triton(_apply_delta_chunk_kernel)[grid](
        q,
        k,
        w,
        u,
        output_weights,
        cumulative_decay,
        final_decay_ratio,
        state,
        output,
        T=tokens,
        chunks=chunks,
        H=heads,
        K=key_dim,
        V=value_dim,
        CHUNK=_CHUNK_SIZE,
        BLOCK_K=block_k,
        BLOCK_V=32,
        num_warps=2,
        num_stages=1,
    )
    return output


def gated_delta_chunk(
    q: torch.Tensor,
    k: torch.Tensor,
    value: torch.Tensor,
    beta: torch.Tensor,
    decay: torch.Tensor,
    state: torch.Tensor,
) -> torch.Tensor:
    return _gated_delta_chunk(
        q.contiguous(),
        k.contiguous(),
        value.contiguous(),
        beta.contiguous(),
        decay.contiguous(),
        state,
    )
