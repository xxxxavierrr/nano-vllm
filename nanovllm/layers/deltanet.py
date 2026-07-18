from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton


@triton.jit(do_not_specialize=["T"])
def _gated_delta_recurrent_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    beta_ptr,
    decay_ptr,
    state_ptr,
    output_ptr,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    value_block = tl.program_id(axis=0)
    head = tl.program_id(axis=1)
    offs_k = tl.arange(0, BLOCK_K)
    offs_v = value_block * BLOCK_V + tl.arange(0, BLOCK_V)
    state_offsets = head * K * V + offs_k[:, None] * V + offs_v[None, :]
    state = tl.load(
        state_ptr + state_offsets,
        mask=(offs_k[:, None] < K) & (offs_v[None, :] < V),
        other=0.0,
    ).to(tl.float32)

    for token in range(0, T):
        q = tl.load(
            q_ptr + (token * H + head) * K + offs_k,
            mask=offs_k < K,
            other=0.0,
        ).to(tl.float32)
        k = tl.load(
            k_ptr + (token * H + head) * K + offs_k,
            mask=offs_k < K,
            other=0.0,
        ).to(tl.float32)
        value = tl.load(
            v_ptr + (token * H + head) * V + offs_v,
            mask=offs_v < V,
            other=0.0,
        ).to(tl.float32)
        beta = tl.load(beta_ptr + token * H + head).to(tl.float32)
        decay = tl.load(decay_ptr + token * H + head).to(tl.float32)

        state *= decay
        memory = tl.sum(state * k[:, None], axis=0)
        delta = (value - memory) * beta
        state += k[:, None] * delta[None, :]
        output = tl.sum(state * q[:, None], axis=0)
        tl.store(
            output_ptr + (token * H + head) * V + offs_v,
            output,
            mask=offs_v < V,
        )

    tl.store(
        state_ptr + state_offsets,
        state,
        mask=(offs_k[:, None] < K) & (offs_v[None, :] < V),
    )


def _validate_recurrent_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    value: torch.Tensor,
    beta: torch.Tensor,
    decay: torch.Tensor,
    state: torch.Tensor,
) -> tuple[int, int, int, int]:
    tensors = (q, k, value, beta, decay, state)
    if not all(t.is_cuda for t in tensors):
        raise ValueError("DeltaNet recurrent kernel requires CUDA tensors")
    if not all(t.dtype is torch.float32 for t in tensors):
        raise TypeError("DeltaNet recurrent kernel requires float32 tensors")
    if q.ndim != 3 or k.shape != q.shape or value.ndim != 3:
        raise ValueError("q/k/value must have shapes [T, H, K/V]")
    t, h, key_dim = q.shape
    if value.shape[:2] != (t, h):
        raise ValueError("value T/H dimensions must match q/k")
    value_dim = value.shape[-1]
    if beta.shape != (t, h) or decay.shape != (t, h):
        raise ValueError("beta and decay must have shape [T, H]")
    if state.shape != (h, key_dim, value_dim):
        raise ValueError("state must have shape [H, K, V]")
    if not all(tensor.is_contiguous() for tensor in tensors):
        raise ValueError("DeltaNet recurrent tensors must be contiguous")
    return t, h, key_dim, value_dim


@triton_op("nanovllm::gated_delta_recurrent", mutates_args={"state"})
def _gated_delta_recurrent(
    q: torch.Tensor,
    k: torch.Tensor,
    value: torch.Tensor,
    beta: torch.Tensor,
    decay: torch.Tensor,
    state: torch.Tensor,
) -> torch.Tensor:
    t, h, key_dim, value_dim = _validate_recurrent_inputs(
        q, k, value, beta, decay, state
    )
    output = torch.empty_like(value)
    block_k = triton.next_power_of_2(key_dim)
    block_v = min(8, triton.next_power_of_2(value_dim))
    grid = (triton.cdiv(value_dim, block_v), h)
    wrap_triton(_gated_delta_recurrent_kernel)[grid](
        q,
        k,
        value,
        beta,
        decay,
        state,
        output,
        T=t,
        H=h,
        K=key_dim,
        V=value_dim,
        BLOCK_K=block_k,
        BLOCK_V=block_v,
        num_warps=1,
        num_stages=3,
    )
    return output


def gated_delta_recurrent(
    q: torch.Tensor,
    k: torch.Tensor,
    value: torch.Tensor,
    beta: torch.Tensor,
    decay: torch.Tensor,
    state: torch.Tensor,
) -> torch.Tensor:
    return _gated_delta_recurrent(
        q.contiguous(),
        k.contiguous(),
        value.contiguous(),
        beta.contiguous(),
        decay.contiguous(),
        state,
    )

