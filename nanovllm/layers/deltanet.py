"""Recurrent Gated DeltaNet backend.

This module contains one recurrent Triton kernel and the tensor contracts
shared with chunk execution.  It intentionally has no scheduler, request, or
model-projection logic.

Compiler integration lives at the model's single stateful GDN-core custom-op
boundary, not on this numerical kernel.

Layouts:

* q, k: ``[tokens, heads, key_dim]``
* value: ``[tokens, heads, value_dim]``
* beta, decay: ``[tokens, heads]``
* state slab: ``[capacity, heads, key_dim, value_dim]``
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import triton
import triton.language as tl


class DeltaNetShape(NamedTuple):
    tokens: int
    heads: int
    key_dim: int
    value_dim: int


@triton.jit
def _packed_causal_conv_kernel(
    input_ptr,
    weight_ptr,
    cu_seqlens_ptr,
    slots_ptr,
    state_ptr,
    output_ptr,
    D: tl.constexpr,
    KERNEL: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Apply depthwise causal convolution while reading the old state."""
    local_token = tl.program_id(axis=0)
    sequence = tl.program_id(axis=1)
    channel_block = tl.program_id(axis=2)

    begin = tl.load(cu_seqlens_ptr + sequence)
    end = tl.load(cu_seqlens_ptr + sequence + 1)
    token = begin + local_token
    token_active = token < end
    slot = tl.load(slots_ptr + sequence)
    channels = channel_block * BLOCK_D + tl.arange(0, BLOCK_D)
    channel_mask = channels < D

    accumulator = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for tap in range(KERNEL):
        source_local = local_token + tap + 1 - KERNEL
        history_position = KERNEL + source_local
        history = tl.load(
            state_ptr + (slot * D + channels) * KERNEL + history_position,
            mask=channel_mask & token_active & (source_local < 0),
            other=0.0,
        ).to(tl.float32)
        current = tl.load(
            input_ptr + (begin + source_local) * D + channels,
            mask=channel_mask & token_active & (source_local >= 0),
            other=0.0,
        ).to(tl.float32)
        weight = tl.load(
            weight_ptr + channels * KERNEL + tap,
            mask=channel_mask,
            other=0.0,
        ).to(tl.float32)
        accumulator += tl.where(source_local < 0, history, current) * weight

    activated = accumulator * tl.sigmoid(accumulator)
    tl.store(
        output_ptr + token * D + channels,
        activated,
        mask=channel_mask & token_active,
    )


@triton.jit
def _packed_causal_conv_state_kernel(
    input_ptr,
    cu_seqlens_ptr,
    slots_ptr,
    state_ptr,
    D: tl.constexpr,
    KERNEL: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Commit the last K inputs after convolution consumed the old state."""
    sequence = tl.program_id(axis=0)
    channel_block = tl.program_id(axis=1)
    begin = tl.load(cu_seqlens_ptr + sequence)
    end = tl.load(cu_seqlens_ptr + sequence + 1)
    length = end - begin
    slot = tl.load(slots_ptr + sequence)

    channels = channel_block * BLOCK_D + tl.arange(0, BLOCK_D)
    positions = tl.arange(0, BLOCK_K)
    channel_mask = channels < D
    position_mask = positions < KERNEL
    concatenated_position = length + positions

    # Every old-state element is loaded before this program writes any state
    # element, which makes short-query left shifts race-free.
    old_state = tl.load(
        state_ptr
        + (slot * D + channels[:, None]) * KERNEL
        + concatenated_position[None, :],
        mask=(
            channel_mask[:, None]
            & position_mask[None, :]
            & (concatenated_position[None, :] < KERNEL)
        ),
        other=0.0,
    )
    input_position = begin + concatenated_position - KERNEL
    new_input = tl.load(
        input_ptr + input_position[None, :] * D + channels[:, None],
        mask=(
            channel_mask[:, None]
            & position_mask[None, :]
            & (concatenated_position[None, :] >= KERNEL)
        ),
        other=0.0,
    )
    updated = tl.where(
        concatenated_position[None, :] < KERNEL,
        old_state,
        new_input,
    )
    tl.store(
        state_ptr
        + (slot * D + channels[:, None]) * KERNEL
        + positions[None, :],
        updated,
        mask=channel_mask[:, None] & position_mask[None, :],
    )


def packed_causal_conv1d(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    cu_seqlens: torch.Tensor,
    slots: torch.Tensor,
    state_slab: torch.Tensor,
    max_seqlen: int,
) -> torch.Tensor:
    """Run packed depthwise causal convolution and update request states."""
    if inputs.ndim != 2:
        raise ValueError("packed causal convolution input must have shape [T, D]")
    tokens, channels = inputs.shape
    if weight.ndim != 3 or weight.shape[:2] != (channels, 1):
        raise ValueError("packed causal convolution weight must have shape [D, 1, K]")
    kernel_size = weight.shape[2]
    if state_slab.ndim != 3 or state_slab.shape[1:] != (
        channels,
        kernel_size,
    ):
        raise ValueError(
            "packed causal convolution state must have shape [capacity, D, K]"
        )
    num_sequences = _validate_packed_metadata(
        cu_seqlens,
        slots,
        operation="packed causal convolution",
    )
    if tokens <= 0 or num_sequences <= 0 or max_seqlen <= 0:
        raise ValueError("packed causal convolution requires a non-empty batch")
    if not all(
        tensor.is_cuda for tensor in (inputs, weight, state_slab)
    ):
        raise ValueError("packed causal convolution requires CUDA tensors")
    if inputs.dtype != weight.dtype or inputs.dtype != state_slab.dtype:
        raise TypeError("packed causal convolution tensors must share one dtype")
    if not all(
        tensor.is_contiguous() for tensor in (inputs, weight, state_slab)
    ):
        raise ValueError("packed causal convolution tensors must be contiguous")

    output = torch.empty_like(inputs)
    block_d = min(128, triton.next_power_of_2(channels))
    _packed_causal_conv_kernel[
        (max_seqlen, num_sequences, triton.cdiv(channels, block_d))
    ](
        inputs,
        weight,
        cu_seqlens,
        slots,
        state_slab,
        output,
        D=channels,
        KERNEL=kernel_size,
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=2,
    )
    block_k = triton.next_power_of_2(kernel_size)
    _packed_causal_conv_state_kernel[
        (num_sequences, triton.cdiv(channels, block_d))
    ](
        inputs,
        cu_seqlens,
        slots,
        state_slab,
        D=channels,
        KERNEL=kernel_size,
        BLOCK_D=block_d,
        BLOCK_K=block_k,
        num_warps=4,
        num_stages=2,
    )
    return output


def _validate_delta_tensors(
    q: torch.Tensor,
    k: torch.Tensor,
    value: torch.Tensor,
    beta: torch.Tensor,
    decay: torch.Tensor,
    *,
    state_slab: torch.Tensor,
    operation: str = "DeltaNet",
) -> DeltaNetShape:
    """Validate the numerical contract shared by every DeltaNet backend."""
    if q.ndim != 3 or k.shape != q.shape or value.ndim != 3:
        raise ValueError(f"{operation} q/k/value must have shapes [T, H, K/V]")
    tokens, heads, key_dim = q.shape
    value_dim = value.shape[-1]
    if value.shape[:2] != (tokens, heads):
        raise ValueError(f"{operation} value T/H dimensions must match q/k")
    if beta.shape != (tokens, heads) or decay.shape != (tokens, heads):
        raise ValueError(f"{operation} beta and decay must have shape [T, H]")
    if (
        state_slab.ndim != 4
        or state_slab.shape[1:] != (heads, key_dim, value_dim)
    ):
        raise ValueError(
            f"{operation} state slab must have shape [capacity, H, K, V]"
        )

    tensors = [q, k, value, beta, decay]
    tensors.append(state_slab)
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError(f"{operation} requires CUDA tensors")
    if not all(tensor.dtype is torch.float32 for tensor in tensors):
        raise TypeError(f"{operation} requires float32 tensors")
    if not all(tensor.is_contiguous() for tensor in tensors):
        raise ValueError(f"{operation} tensors must be contiguous")
    return DeltaNetShape(tokens, heads, key_dim, value_dim)


def _validate_packed_metadata(
    cu_seqlens: torch.Tensor,
    slots: torch.Tensor,
    *,
    operation: str = "packed DeltaNet",
) -> int:
    """Validate sequence-to-state metadata without synchronizing the GPU."""
    if cu_seqlens.dtype is not torch.int32 or slots.dtype is not torch.int32:
        raise TypeError(f"{operation} cu_seqlens and state slots must be int32")
    if cu_seqlens.ndim != 1 or slots.ndim != 1:
        raise ValueError(f"{operation} cu_seqlens and state slots must be rank-1")
    if cu_seqlens.numel() != slots.numel() + 1:
        raise ValueError(
            f"{operation} cu_seqlens must contain one more entry than slots"
        )
    if not all(
        tensor.is_cuda and tensor.is_contiguous()
        for tensor in (cu_seqlens, slots)
    ):
        raise ValueError(f"{operation} metadata must be contiguous CUDA tensors")
    return slots.numel()


@triton.jit
def _gated_delta_recurrent_kernel(
    q_ptr,
    k_ptr,
    value_ptr,
    beta_ptr,
    decay_ptr,
    cu_seqlens_ptr,
    sequence_indices_ptr,
    slots_ptr,
    state_ptr,
    output_ptr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """Run an indexed subset of sequences from one packed token buffer."""
    value_block = tl.program_id(axis=0)
    packed_sequence_head = tl.program_id(axis=1)
    packed_sequence = packed_sequence_head // H
    head = packed_sequence_head % H

    sequence = tl.load(sequence_indices_ptr + packed_sequence)
    begin = tl.load(cu_seqlens_ptr + sequence)
    end = tl.load(cu_seqlens_ptr + sequence + 1)
    slot = tl.load(slots_ptr + sequence)

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

    token = begin
    while token < end:
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
            value_ptr + (token * H + head) * V + offs_v,
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
        token += 1

    tl.store(
        state_ptr + state_offsets,
        state,
        mask=(offs_k[:, None] < K) & (offs_v[None, :] < V),
    )


def _launch_recurrent(
    q: torch.Tensor,
    k: torch.Tensor,
    value: torch.Tensor,
    beta: torch.Tensor,
    decay: torch.Tensor,
    cu_seqlens: torch.Tensor,
    sequence_indices: torch.Tensor,
    slots: torch.Tensor,
    state: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """Launch recurrent execution for selected packed sequences."""
    _, heads, key_dim = q.shape
    value_dim = value.shape[-1]
    num_sequences = sequence_indices.numel()
    if num_sequences == 0:
        return

    block_k = triton.next_power_of_2(key_dim)
    block_v = min(8, triton.next_power_of_2(value_dim))
    grid = (triton.cdiv(value_dim, block_v), num_sequences * heads)
    _gated_delta_recurrent_kernel[grid](
        q,
        k,
        value,
        beta,
        decay,
        cu_seqlens,
        sequence_indices,
        slots,
        state,
        output,
        H=heads,
        K=key_dim,
        V=value_dim,
        BLOCK_K=block_k,
        BLOCK_V=block_v,
        num_warps=1,
        num_stages=3,
    )
