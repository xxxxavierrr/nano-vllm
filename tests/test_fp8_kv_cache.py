import math

import pytest
import torch

from nanovllm.layers.fp8_attention import (
    dequantize_fp8_reference,
    fp8_paged_attention,
    quantize_fp8_reference,
    store_fp8_kvcache,
)


def test_fp8_reference_zero_and_boundary_values():
    values = torch.tensor(
        [[[0.0, -448.0, 448.0, 1.0] * 16]],
        dtype=torch.bfloat16,
    )
    quantized, scale = quantize_fp8_reference(values)
    restored = dequantize_fp8_reference(quantized, scale)
    assert torch.isfinite(restored).all()
    assert scale.dtype == torch.float16
    assert scale.item() > 0
    torch.testing.assert_close(restored, values, rtol=0.07, atol=0.07)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("head_dim", [64, 128, 256])
def test_store_fp8_kvcache_matches_reference(head_dim):
    torch.manual_seed(0)
    key = torch.randn(5, 2, head_dim, device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    slot_mapping = torch.tensor([3, -1, 0, 7, 4], device="cuda", dtype=torch.int32)
    k_cache = torch.zeros(
        2, 4, 2, head_dim, device="cuda", dtype=torch.float8_e4m3fn
    )
    v_cache = torch.zeros_like(k_cache)
    k_scale = torch.zeros(2, 4, 2, device="cuda", dtype=torch.float16)
    v_scale = torch.zeros_like(k_scale)

    store_fp8_kvcache(
        key, value, k_cache, v_cache, k_scale, v_scale, slot_mapping
    )
    torch.cuda.synchronize()
    expected_k, expected_ks = quantize_fp8_reference(key)
    expected_v, expected_vs = quantize_fp8_reference(value)
    flat_k = k_cache.view(-1, 2, head_dim)
    flat_v = v_cache.view(-1, 2, head_dim)
    flat_ks = k_scale.view(-1, 2)
    flat_vs = v_scale.view(-1, 2)
    for token, slot in enumerate(slot_mapping.tolist()):
        if slot == -1:
            continue
        torch.testing.assert_close(flat_ks[slot], expected_ks[token], rtol=0, atol=0)
        torch.testing.assert_close(flat_vs[slot], expected_vs[token], rtol=0, atol=0)
        stored_k = dequantize_fp8_reference(flat_k[slot], flat_ks[slot])
        stored_v = dequantize_fp8_reference(flat_v[slot], flat_vs[slot])
        reference_k = dequantize_fp8_reference(expected_k[token], expected_ks[token])
        reference_v = dequantize_fp8_reference(expected_v[token], expected_vs[token])
        torch.testing.assert_close(stored_k, reference_k, rtol=0.13, atol=0.03)
        torch.testing.assert_close(stored_v, reference_v, rtol=0.13, atol=0.03)
        torch.testing.assert_close(stored_k, key[token], rtol=0.13, atol=0.03)
        torch.testing.assert_close(stored_v, value[token], rtol=0.13, atol=0.03)


def _paged_attention_reference(
    query,
    k_cache,
    v_cache,
    k_scale,
    v_scale,
    block_tables,
    contexts,
    seq_query_ranges,
    query_positions,
    softmax_scale,
):
    output = torch.empty_like(query)
    num_query_heads = query.shape[1]
    num_kv_heads = k_cache.shape[-2]
    queries_per_kv = num_query_heads // num_kv_heads
    block_size = k_cache.shape[1]
    for seq_idx, (query_start, query_end) in enumerate(seq_query_ranges):
        context_len = contexts[seq_idx]
        logical = torch.arange(context_len, device=query.device)
        pages = block_tables[seq_idx, logical // block_size].long()
        offsets = logical % block_size
        keys = dequantize_fp8_reference(
            k_cache[pages, offsets], k_scale[pages, offsets]
        ).float()
        values = dequantize_fp8_reference(
            v_cache[pages, offsets], v_scale[pages, offsets]
        ).float()
        for query_idx in range(query_start, query_end):
            absolute_position = query_positions[query_idx]
            valid = logical <= absolute_position
            for query_head in range(num_query_heads):
                kv_head = query_head // queries_per_kv
                scores = (
                    query[query_idx, query_head].float()
                    @ keys[valid, kv_head].T
                ) * softmax_scale
                probabilities = torch.softmax(scores, dim=-1)
                output[query_idx, query_head] = (
                    probabilities @ values[valid, kv_head]
                ).to(output.dtype)
    return output


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("head_dim", [64, 128, 256])
def test_fp8_paged_attention_mixed_cross_page(head_dim):
    torch.manual_seed(1)
    block_size = 32
    num_blocks = 4
    num_kv_heads = 2
    num_query_heads = 4
    k_source = torch.randn(
        num_blocks, block_size, num_kv_heads, head_dim,
        device="cuda", dtype=torch.bfloat16,
    )
    v_source = torch.randn_like(k_source)
    k_cache, k_scale = quantize_fp8_reference(k_source)
    v_cache, v_scale = quantize_fp8_reference(v_source)
    block_tables = torch.tensor(
        [[1, 3], [0, 2]], device="cuda", dtype=torch.int32
    )
    contexts_host = [40, 4]
    contexts = torch.tensor(contexts_host, device="cuda", dtype=torch.int32)
    query = torch.randn(
        4, num_query_heads, head_dim, device="cuda", dtype=torch.bfloat16
    )
    tile_seq_ids = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    tile_starts = torch.tensor([0, 3], device="cuda", dtype=torch.int32)
    tile_lens = torch.tensor([3, 1], device="cuda", dtype=torch.int32)
    tile_positions = torch.tensor([37, 3], device="cuda", dtype=torch.int32)
    query_positions = [37, 38, 39, 3]
    softmax_scale = 1.0 / math.sqrt(head_dim)

    actual = fp8_paged_attention(
        query,
        k_cache,
        v_cache,
        k_scale,
        v_scale,
        block_tables,
        contexts,
        tile_seq_ids,
        tile_starts,
        tile_lens,
        tile_positions,
        softmax_scale,
        block_size,
    )
    expected = _paged_attention_reference(
        query,
        k_cache,
        v_cache,
        k_scale,
        v_scale,
        block_tables,
        contexts_host,
        [(0, 3), (3, 4)],
        query_positions,
        softmax_scale,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
    cosine = torch.nn.functional.cosine_similarity(
        actual.float().flatten(), expected.float().flatten(), dim=0
    )
    assert cosine.item() >= 0.999


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("head_dim", [64, 128, 256])
def test_fp8_paged_attention_uniform_decode(head_dim):
    torch.manual_seed(2)
    block_size = 32
    batch_size = 3
    blocks_per_seq = 3
    num_kv_heads = 2
    num_query_heads = 4
    num_blocks = batch_size * blocks_per_seq
    k_source = torch.randn(
        num_blocks, block_size, num_kv_heads, head_dim,
        device="cuda", dtype=torch.bfloat16,
    )
    v_source = torch.randn_like(k_source)
    k_cache, k_scale = quantize_fp8_reference(k_source)
    v_cache, v_scale = quantize_fp8_reference(v_source)
    block_tables = torch.arange(
        num_blocks, device="cuda", dtype=torch.int32
    ).reshape(batch_size, blocks_per_seq)
    contexts_host = [70, 61, 33]
    contexts = torch.tensor(contexts_host, device="cuda", dtype=torch.int32)
    query = torch.randn(
        batch_size, num_query_heads, head_dim,
        device="cuda", dtype=torch.bfloat16,
    )
    tile_seq_ids = torch.arange(batch_size, device="cuda", dtype=torch.int32)
    tile_starts = tile_seq_ids.clone()
    tile_lens = torch.ones(batch_size, device="cuda", dtype=torch.int32)
    tile_positions = contexts - 1
    softmax_scale = 1.0 / math.sqrt(head_dim)

    actual = fp8_paged_attention(
        query,
        k_cache,
        v_cache,
        k_scale,
        v_scale,
        block_tables,
        contexts,
        tile_seq_ids,
        tile_starts,
        tile_lens,
        tile_positions,
        softmax_scale,
        block_size,
    )
    expected = _paged_attention_reference(
        query,
        k_cache,
        v_cache,
        k_scale,
        v_scale,
        block_tables,
        contexts_host,
        [(0, 1), (1, 2), (2, 3)],
        [length - 1 for length in contexts_host],
        softmax_scale,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
    cosine = torch.nn.functional.cosine_similarity(
        actual.float().flatten(), expected.float().flatten(), dim=0
    )
    assert cosine.item() >= 0.999