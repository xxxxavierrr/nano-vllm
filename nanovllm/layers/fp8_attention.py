from __future__ import annotations

import torch
import triton
import triton.language as tl

from nanovllm.layers.fp8_decode_attention import fp8_paged_attention_decode


FP8_MAX = 448.0
FP16_MIN_SUBNORMAL = 2.0**-24
FP8_QUERY_TILE_SIZE = 16


@triton.jit
def _store_fp8_kvcache_kernel(
    key_ptr,
    value_ptr,
    k_cache_ptr,
    v_cache_ptr,
    k_scale_ptr,
    v_scale_ptr,
    slot_mapping_ptr,
    key_stride_t,
    key_stride_h,
    value_stride_t,
    value_stride_h,
    num_kv_heads: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    slot = tl.load(slot_mapping_ptr + token_idx)
    if slot == -1:
        return

    offsets_d = tl.arange(0, HEAD_DIM)
    key = tl.load(
        key_ptr
        + token_idx * key_stride_t
        + head_idx * key_stride_h
        + offsets_d
    ).to(tl.float32)
    value = tl.load(
        value_ptr
        + token_idx * value_stride_t
        + head_idx * value_stride_h
        + offsets_d
    ).to(tl.float32)

    key_absmax = tl.max(tl.abs(key), axis=0)
    value_absmax = tl.max(tl.abs(value), axis=0)
    key_scale = tl.maximum(key_absmax / 448.0, 2.0**-24)
    value_scale = tl.maximum(value_absmax / 448.0, 2.0**-24)
    # Quantize with the exact rounded value that is persisted in the scale cache.
    key_scale = key_scale.to(tl.float16).to(tl.float32)
    value_scale = value_scale.to(tl.float16).to(tl.float32)

    cache_offsets = (
        slot * num_kv_heads * HEAD_DIM
        + head_idx * HEAD_DIM
        + offsets_d
    )
    scale_offset = slot * num_kv_heads + head_idx
    quant_key = tl.maximum(tl.minimum(key / key_scale, 448.0), -448.0)
    quant_value = tl.maximum(tl.minimum(value / value_scale, 448.0), -448.0)
    tl.store(k_cache_ptr + cache_offsets, quant_key)
    tl.store(v_cache_ptr + cache_offsets, quant_value)
    tl.store(k_scale_ptr + scale_offset, key_scale)
    tl.store(v_scale_ptr + scale_offset, value_scale)


def store_fp8_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    num_tokens, num_kv_heads, head_dim = key.shape
    if value.shape != key.shape:
        raise ValueError("key and value must have the same shape")
    if head_dim not in (64, 128, 256):
        raise ValueError(f"FP8 KV cache does not support head_dim={head_dim}")
    if key.dtype != torch.bfloat16 or value.dtype != torch.bfloat16:
        raise ValueError("FP8 KV cache requires BF16 key/value activations")
    if k_cache.dtype != torch.float8_e4m3fn or v_cache.dtype != torch.float8_e4m3fn:
        raise ValueError("FP8 KV cache tensors must use torch.float8_e4m3fn")
    if k_scale.dtype != torch.float16 or v_scale.dtype != torch.float16:
        raise ValueError("FP8 KV scales must use float16")
    if slot_mapping.numel() != num_tokens:
        raise ValueError("slot_mapping must contain one entry per token")
    if not (
        key.stride(-1) == value.stride(-1) == 1
        and k_cache.is_contiguous()
        and v_cache.is_contiguous()
        and k_scale.is_contiguous()
        and v_scale.is_contiguous()
    ):
        raise ValueError("FP8 KV cache requires contiguous head/cache dimensions")

    _store_fp8_kvcache_kernel[(num_tokens, num_kv_heads)](
        key,
        value,
        k_cache,
        v_cache,
        k_scale,
        v_scale,
        slot_mapping,
        key.stride(0),
        key.stride(1),
        value.stride(0),
        value.stride(1),
        num_kv_heads,
        HEAD_DIM=head_dim,
        num_warps=4,
    )


@triton.jit
def _fp8_paged_attention_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    k_scale_ptr,
    v_scale_ptr,
    block_tables_ptr,
    context_lens_ptr,
    tile_seq_ids_ptr,
    tile_starts_ptr,
    tile_lens_ptr,
    tile_positions_ptr,
    out_ptr,
    q_stride_t,
    q_stride_h,
    out_stride_t,
    out_stride_h,
    block_table_stride,
    num_kv_heads: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    Q_PER_KV: tl.constexpr,
    SOFTMAX_SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    tile_idx = tl.program_id(0)
    query_head = tl.program_id(1)
    seq_idx = tl.load(tile_seq_ids_ptr + tile_idx)
    query_start = tl.load(tile_starts_ptr + tile_idx)
    query_rows = tl.load(tile_lens_ptr + tile_idx)
    query_position = tl.load(tile_positions_ptr + tile_idx)
    context_len = tl.load(context_lens_ptr + seq_idx)
    kv_head = query_head // Q_PER_KV

    offsets_m = tl.arange(0, BLOCK_M)
    offsets_n = tl.arange(0, BLOCK_N)
    offsets_d = tl.arange(0, HEAD_DIM)
    active_rows = offsets_m < query_rows
    q_offsets = (
        (query_start + offsets_m[:, None]) * q_stride_t
        + query_head * q_stride_h
        + offsets_d[None, :]
    )
    q = tl.load(q_ptr + q_offsets, mask=active_rows[:, None], other=0.0)

    neg_inf = float("-inf")
    log2e = 1.4426950408889634
    running_max = tl.full((BLOCK_M,), neg_inf, tl.float32)
    running_sum = tl.zeros((BLOCK_M,), tl.float32)
    accumulator = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)
    key_start = 0

    while key_start < context_len:
        key_positions = key_start + offsets_n
        valid_keys = key_positions < context_len
        logical_blocks = key_positions // BLOCK_SIZE
        block_offsets = key_positions % BLOCK_SIZE
        physical_blocks = tl.load(
            block_tables_ptr
            + seq_idx * block_table_stride
            + logical_blocks,
            mask=valid_keys,
            other=0,
        )
        slots = physical_blocks * BLOCK_SIZE + block_offsets

        cache_offsets = (
            slots[:, None] * num_kv_heads * HEAD_DIM
            + kv_head * HEAD_DIM
            + offsets_d[None, :]
        )
        scale_offsets = slots * num_kv_heads + kv_head
        key_scale = tl.load(
            k_scale_ptr + scale_offsets, mask=valid_keys, other=0.0
        ).to(tl.float32)
        key = tl.load(
            k_cache_ptr + cache_offsets,
            mask=valid_keys[:, None],
            other=0.0,
        ).to(tl.float32)
        key = (key * key_scale[:, None]).to(tl.bfloat16)

        scores = tl.dot(q, tl.trans(key)) * SOFTMAX_SCALE
        absolute_q = query_position + offsets_m
        causal = key_positions[None, :] <= absolute_q[:, None]
        valid = active_rows[:, None] & valid_keys[None, :] & causal
        scores = tl.where(valid, scores, neg_inf)

        tile_max = tl.max(scores, axis=1)
        new_max = tl.maximum(running_max, tile_max)
        new_max = tl.where(active_rows, new_max, 0.0)
        alpha = tl.where(
            active_rows,
            tl.exp2((running_max - new_max) * log2e),
            0.0,
        )
        probabilities = tl.where(
            valid,
            tl.exp2((scores - new_max[:, None]) * log2e),
            0.0,
        )
        new_sum = running_sum * alpha + tl.sum(probabilities, axis=1)

        value_scale = tl.load(
            v_scale_ptr + scale_offsets, mask=valid_keys, other=0.0
        ).to(tl.float32)
        value = tl.load(
            v_cache_ptr + cache_offsets,
            mask=valid_keys[:, None],
            other=0.0,
        ).to(tl.float32)
        value = (value * value_scale[:, None]).to(tl.bfloat16)
        accumulator = (
            accumulator * alpha[:, None]
            + tl.dot(probabilities.to(tl.bfloat16), value)
        )
        running_max = new_max
        running_sum = new_sum
        key_start += BLOCK_N

    output = accumulator / running_sum[:, None]
    out_offsets = (
        (query_start + offsets_m[:, None]) * out_stride_t
        + query_head * out_stride_h
        + offsets_d[None, :]
    )
    tl.store(out_ptr + out_offsets, output, mask=active_rows[:, None])


def _validate_fp8_attention(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    tile_seq_ids: torch.Tensor,
    tile_starts: torch.Tensor,
    tile_lens: torch.Tensor,
    tile_positions: torch.Tensor,
    block_size: int,
) -> tuple[int, int, int, int]:
    num_tokens, num_query_heads, head_dim = query.shape
    num_kv_heads = k_cache.shape[-2]
    if query.dtype != torch.bfloat16:
        raise ValueError("FP8 paged attention requires BF16 query activations")
    if head_dim not in (64, 128, 256):
        raise ValueError(f"FP8 paged attention does not support head_dim={head_dim}")
    if num_query_heads % num_kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    if block_size % 32:
        raise ValueError("FP8 paged attention requires block_size divisible by 32")
    if k_cache.dtype != torch.float8_e4m3fn or v_cache.dtype != torch.float8_e4m3fn:
        raise ValueError("FP8 paged attention requires E4M3 cache tensors")
    if any(t is None for t in (
        block_tables,
        context_lens,
        tile_seq_ids,
        tile_starts,
        tile_lens,
        tile_positions,
    )):
        raise ValueError("FP8 paged attention metadata is incomplete")
    if not (
        query.stride(-1) == 1
        and k_cache.is_contiguous()
        and v_cache.is_contiguous()
        and k_scale.is_contiguous()
        and v_scale.is_contiguous()
    ):
        raise ValueError("FP8 paged attention requires contiguous cache tensors")
    return num_tokens, num_query_heads, num_kv_heads, head_dim


def _launch_fp8_attention(
    query, k_cache, v_cache, k_scale, v_scale, block_tables, context_lens,
    tile_seq_ids, tile_starts, tile_lens, tile_positions, softmax_scale,
    block_size, output, num_query_heads, num_kv_heads, head_dim,
) -> None:
    block_n = 32 if head_dim == 256 else 64
    num_warps = 8 if head_dim == 256 else 4
    grid = (tile_seq_ids.numel(), num_query_heads)
    _fp8_paged_attention_kernel[grid](
        query, k_cache, v_cache, k_scale, v_scale, block_tables, context_lens,
        tile_seq_ids, tile_starts, tile_lens, tile_positions, output,
        query.stride(0), query.stride(1), output.stride(0), output.stride(1),
        block_tables.stride(0), num_kv_heads,
        BLOCK_SIZE=block_size, HEAD_DIM=head_dim,
        Q_PER_KV=num_query_heads // num_kv_heads, SOFTMAX_SCALE=softmax_scale,
        BLOCK_M=FP8_QUERY_TILE_SIZE, BLOCK_N=block_n,
        num_warps=num_warps, num_stages=2,
    )


def fp8_paged_attention(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    tile_seq_ids: torch.Tensor,
    tile_starts: torch.Tensor,
    tile_lens: torch.Tensor,
    tile_positions: torch.Tensor,
    softmax_scale: float,
    block_size: int,
) -> torch.Tensor:
    num_tokens, num_query_heads, num_kv_heads, head_dim = _validate_fp8_attention(
        query, k_cache, v_cache, k_scale, v_scale, block_tables, context_lens,
        tile_seq_ids, tile_starts, tile_lens, tile_positions, block_size,
    )

    num_tiles = tile_seq_ids.numel()
    if num_tiles == 0:
        return torch.empty_like(query)
    if num_tokens == context_lens.numel() and num_tiles == num_tokens:
        return fp8_paged_attention_decode(
            query,
            k_cache,
            v_cache,
            k_scale,
            v_scale,
            block_tables,
            context_lens,
            softmax_scale,
            block_size,
        )

    output = torch.empty_like(query)
    _launch_fp8_attention(
        query, k_cache, v_cache, k_scale, v_scale, block_tables, context_lens,
        tile_seq_ids, tile_starts, tile_lens, tile_positions, softmax_scale,
        block_size, output, num_query_heads, num_kv_heads, head_dim,
    )
    return output


def quantize_fp8_reference(
    tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if tensor.dtype != torch.bfloat16:
        raise ValueError("reference FP8 KV quantization requires BF16 input")
    scale = tensor.float().abs().amax(dim=-1) / FP8_MAX
    scale = scale.clamp_min(FP16_MIN_SUBNORMAL).to(torch.float16)
    quantized = (
        tensor.float() / scale.float().unsqueeze(-1)
    ).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return quantized, scale


def dequantize_fp8_reference(
    tensor: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    return (tensor.float() * scale.float().unsqueeze(-1)).to(dtype)
