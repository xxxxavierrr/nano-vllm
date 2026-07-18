from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton


_CHUNK_SIZE = 32


@triton.jit
def _prepare_delta_chunk_kernel(
    q_ptr,
    k_ptr,
    beta_ptr,
    decay_ptr,
    solve_ptr,
    output_weights_ptr,
    cumulative_decay_ptr,
    final_decay_ratio_ptr,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    CHUNK: tl.constexpr,
    BLOCK_K: tl.constexpr,
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
    gram = tl.dot(k, tl.trans(k), input_precision="ieee")
    qk = tl.dot(q, tl.trans(k), input_precision="ieee")
    rows = offs_t[:, None]
    cols = offs_t[None, :]
    solve = tl.where(rows > cols, gram * decay_ratio * beta[:, None], 0.0)
    output_weights = tl.where(rows >= cols, qk * decay_ratio, 0.0)

    matrix_base = (chunk * H + head) * CHUNK * CHUNK
    matrix_offsets = rows * CHUNK + cols
    tl.store(solve_ptr + matrix_base + matrix_offsets, solve)
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


@triton.jit
def _apply_delta_chunk_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    beta_ptr,
    solve_ptr,
    output_weights_ptr,
    cumulative_decay_ptr,
    final_decay_ratio_ptr,
    state_ptr,
    output_ptr,
    chunk,
    T,
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
    value = tl.load(
        v_ptr + (tokens[:, None] * H + head) * V + offs_v[None, :],
        mask=token_mask[:, None] & (offs_v[None, :] < V),
        other=0.0,
    ).to(tl.float32)
    beta = tl.load(
        beta_ptr + tokens * H + head,
        mask=token_mask,
        other=0.0,
    ).to(tl.float32)

    state_offsets = head * K * V + offs_k[:, None] * V + offs_v[None, :]
    state = tl.load(
        state_ptr + state_offsets,
        mask=(offs_k[:, None] < K) & (offs_v[None, :] < V),
        other=0.0,
    ).to(tl.float32)

    vector_base = (chunk * H + head) * CHUNK
    cumulative = tl.load(cumulative_decay_ptr + vector_base + offs_t)
    final_ratio = tl.load(final_decay_ratio_ptr + vector_base + offs_t)
    initial_memory = tl.dot(k, state, input_precision="ieee")
    rhs = beta[:, None] * (
        value - cumulative[:, None] * initial_memory
    )

    delta = tl.zeros((CHUNK, BLOCK_V), dtype=tl.float32)
    matrix_base = (chunk * H + head) * CHUNK * CHUNK
    for row in range(0, CHUNK):
        solve_row = tl.load(
            solve_ptr + matrix_base + row * CHUNK + offs_t
        )
        correction = tl.sum(solve_row[:, None] * delta, axis=0)
        rhs_row = tl.sum(
            tl.where((offs_t == row)[:, None], rhs, 0.0),
            axis=0,
        )
        row_delta = rhs_row - correction
        delta = tl.where(
            (offs_t == row)[:, None],
            row_delta[None, :],
            delta,
        )

    output_weights = tl.load(
        output_weights_ptr
        + matrix_base
        + offs_t[:, None] * CHUNK
        + offs_t[None, :]
    )
    initial_output = tl.dot(q, state, input_precision="ieee")
    local_output = tl.dot(
        output_weights,
        delta,
        input_precision="ieee",
    )
    output = cumulative[:, None] * initial_output + local_output
    tl.store(
        output_ptr + (tokens[:, None] * H + head) * V + offs_v[None, :],
        output,
        mask=token_mask[:, None] & (offs_v[None, :] < V),
    )

    weighted_delta = delta * final_ratio[:, None]
    state_update = tl.dot(
        tl.trans(k),
        weighted_delta,
        input_precision="ieee",
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
    solve = torch.empty(
        chunks,
        heads,
        _CHUNK_SIZE,
        _CHUNK_SIZE,
        device=q.device,
        dtype=torch.float32,
    )
    output_weights = torch.empty_like(solve)
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
        beta,
        decay,
        solve,
        output_weights,
        cumulative_decay,
        final_decay_ratio,
        T=tokens,
        H=heads,
        K=key_dim,
        CHUNK=_CHUNK_SIZE,
        BLOCK_K=block_k,
        num_warps=4,
        num_stages=3,
    )
    grid = (triton.cdiv(value_dim, 16), heads)
    for chunk in range(chunks):
        wrap_triton(_apply_delta_chunk_kernel)[grid](
            q,
            k,
            value,
            beta,
            solve,
            output_weights,
            cumulative_decay,
            final_decay_ratio,
            state,
            output,
            chunk,
            T=tokens,
            H=heads,
            K=key_dim,
            V=value_dim,
            CHUNK=_CHUNK_SIZE,
            BLOCK_K=block_k,
            BLOCK_V=16,
            num_warps=4,
            num_stages=3,
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
