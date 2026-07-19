import torch
import triton
import triton.language as tl


@triton.jit
def _fp8_decode_split_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    k_scale_ptr,
    v_scale_ptr,
    block_tables_ptr,
    context_lens_ptr,
    partial_max_ptr,
    partial_sum_ptr,
    partial_acc_ptr,
    q_stride_t,
    q_stride_h,
    block_table_stride,
    num_query_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    Q_PER_KV: tl.constexpr,
    SOFTMAX_SCALE: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    token_idx = tl.program_id(0)
    query_head = tl.program_id(1)
    split_idx = tl.program_id(2)
    kv_head = query_head // Q_PER_KV
    context_len = tl.load(context_lens_ptr + token_idx)
    split_start = split_idx * BLOCK_N

    offsets_d = tl.arange(0, HEAD_DIM)
    offsets_n = tl.arange(0, BLOCK_N)
    query = tl.load(
        q_ptr
        + token_idx * q_stride_t
        + query_head * q_stride_h
        + offsets_d
    ).to(tl.float32)
    running_max = tl.full((), float("-inf"), tl.float32)
    running_sum = tl.zeros((), tl.float32)
    accumulator = tl.zeros((HEAD_DIM,), tl.float32)
    key_start = split_start

    while key_start < context_len:
        key_positions = key_start + offsets_n
        valid = key_positions < context_len
        logical_blocks = key_positions // BLOCK_SIZE
        block_offsets = key_positions % BLOCK_SIZE
        physical_blocks = tl.load(
            block_tables_ptr
            + token_idx * block_table_stride
            + logical_blocks,
            mask=valid,
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
            k_scale_ptr + scale_offsets, mask=valid, other=0.0
        ).to(tl.float32)
        key = tl.load(
            k_cache_ptr + cache_offsets,
            mask=valid[:, None],
            other=0.0,
        ).to(tl.float32)
        key *= key_scale[:, None]
        scores = tl.sum(key * query[None, :], axis=1) * SOFTMAX_SCALE
        scores = tl.where(valid, scores, float("-inf"))

        tile_max = tl.max(scores, axis=0)
        new_max = tl.maximum(running_max, tile_max)
        alpha = tl.exp2((running_max - new_max) * 1.4426950408889634)
        probabilities = tl.where(
            valid,
            tl.exp2((scores - new_max) * 1.4426950408889634),
            0.0,
        )
        running_sum = running_sum * alpha + tl.sum(probabilities, axis=0)

        value_scale = tl.load(
            v_scale_ptr + scale_offsets, mask=valid, other=0.0
        ).to(tl.float32)
        value = tl.load(
            v_cache_ptr + cache_offsets,
            mask=valid[:, None],
            other=0.0,
        ).to(tl.float32)
        value *= value_scale[:, None]
        accumulator = (
            accumulator * alpha
            + tl.sum(probabilities[:, None] * value, axis=0)
        )
        running_max = new_max
        key_start += BLOCK_N * NUM_SPLITS

    partial_offset = (
        token_idx * num_query_heads * NUM_SPLITS
        + query_head * NUM_SPLITS
        + split_idx
    )
    tl.store(partial_max_ptr + partial_offset, running_max)
    tl.store(partial_sum_ptr + partial_offset, running_sum)
    tl.store(
        partial_acc_ptr + partial_offset * HEAD_DIM + offsets_d,
        accumulator,
    )


@triton.jit
def _fp8_decode_reduce_kernel(
    partial_max_ptr,
    partial_sum_ptr,
    partial_acc_ptr,
    output_ptr,
    output_stride_t,
    output_stride_h,
    num_query_heads: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    token_idx = tl.program_id(0)
    query_head = tl.program_id(1)
    offsets_s = tl.arange(0, NUM_SPLITS)
    offsets_d = tl.arange(0, HEAD_DIM)
    partial_base = (
        token_idx * num_query_heads * NUM_SPLITS
        + query_head * NUM_SPLITS
    )
    local_max = tl.load(partial_max_ptr + partial_base + offsets_s)
    local_sum = tl.load(partial_sum_ptr + partial_base + offsets_s)
    global_max = tl.max(local_max, axis=0)
    weights = tl.where(
        local_sum > 0,
        tl.exp2((local_max - global_max) * 1.4426950408889634),
        0.0,
    )
    denominator = tl.sum(local_sum * weights, axis=0)
    partial_acc = tl.load(
        partial_acc_ptr
        + (partial_base + offsets_s[:, None]) * HEAD_DIM
        + offsets_d[None, :]
    )
    numerator = tl.sum(partial_acc * weights[:, None], axis=0)
    output = numerator / denominator
    tl.store(
        output_ptr
        + token_idx * output_stride_t
        + query_head * output_stride_h
        + offsets_d,
        output,
    )


def fp8_paged_attention_decode(
    query,
    k_cache,
    v_cache,
    k_scale,
    v_scale,
    block_tables,
    context_lens,
    softmax_scale,
    block_size,
):
    num_tokens, num_query_heads, head_dim = query.shape
    num_kv_heads = k_cache.shape[-2]
    num_splits = 16
    partial_max = torch.empty(
        num_tokens,
        num_query_heads,
        num_splits,
        device=query.device,
        dtype=torch.float32,
    )
    partial_sum = torch.empty_like(partial_max)
    partial_acc = torch.empty(
        num_tokens,
        num_query_heads,
        num_splits,
        head_dim,
        device=query.device,
        dtype=torch.float32,
    )
    output = torch.empty_like(query)
    block_n = 16 if head_dim == 256 else 32
    _fp8_decode_split_kernel[
        (num_tokens, num_query_heads, num_splits)
    ](
        query,
        k_cache,
        v_cache,
        k_scale,
        v_scale,
        block_tables,
        context_lens,
        partial_max,
        partial_sum,
        partial_acc,
        query.stride(0),
        query.stride(1),
        block_tables.stride(0),
        num_query_heads,
        num_kv_heads,
        BLOCK_SIZE=block_size,
        HEAD_DIM=head_dim,
        Q_PER_KV=num_query_heads // num_kv_heads,
        SOFTMAX_SCALE=softmax_scale,
        NUM_SPLITS=num_splits,
        BLOCK_N=block_n,
        num_warps=4,
        num_stages=2,
    )
    _fp8_decode_reduce_kernel[(num_tokens, num_query_heads)](
        partial_max,
        partial_sum,
        partial_acc,
        output,
        output.stride(0),
        output.stride(1),
        num_query_heads,
        NUM_SPLITS=num_splits,
        HEAD_DIM=head_dim,
        num_warps=4,
    )
    return output