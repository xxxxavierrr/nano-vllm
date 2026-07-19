import argparse
import json
import math
from statistics import mean, median

import torch
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

from nanovllm.layers.fp8_attention import (
    FP8_QUERY_TILE_SIZE,
    fp8_paged_attention,
    quantize_fp8_reference,
)


def _parse_ints(value):
    return [int(item) for item in value.split(",")]


def _timed(fn, warmup, repeats):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return {
        "mean_ms": mean(samples),
        "median_ms": median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def _metadata(batch_size, query_len, context_len):
    tile_seq_ids = []
    tile_starts = []
    tile_lens = []
    tile_positions = []
    prefix_len = context_len - query_len
    for seq_idx in range(batch_size):
        query_start = seq_idx * query_len
        for offset in range(0, query_len, FP8_QUERY_TILE_SIZE):
            tile_seq_ids.append(seq_idx)
            tile_starts.append(query_start + offset)
            tile_lens.append(min(FP8_QUERY_TILE_SIZE, query_len - offset))
            tile_positions.append(prefix_len + offset)
    return [
        torch.tensor(values, device="cuda", dtype=torch.int32)
        for values in (
            tile_seq_ids,
            tile_starts,
            tile_lens,
            tile_positions,
        )
    ]


def _run_case(args, batch_size, query_len, context_len, head_dim):
    if context_len < query_len:
        raise ValueError("context length must be at least query length")
    block_size = args.block_size
    blocks_per_seq = math.ceil(context_len / block_size)
    num_blocks = batch_size * blocks_per_seq
    block_tables = torch.arange(
        num_blocks, device="cuda", dtype=torch.int32
    ).reshape(batch_size, blocks_per_seq)
    context_lens = torch.full(
        (batch_size,), context_len, device="cuda", dtype=torch.int32
    )
    query = torch.randn(
        batch_size * query_len,
        args.num_query_heads,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    key = torch.randn(
        num_blocks,
        block_size,
        args.num_kv_heads,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    value = torch.randn_like(key)
    scale = 1.0 / math.sqrt(head_dim)
    cu_q = torch.arange(
        0,
        (batch_size + 1) * query_len,
        query_len,
        device="cuda",
        dtype=torch.int32,
    )
    cu_k = torch.arange(
        0,
        (batch_size + 1) * context_len,
        context_len,
        device="cuda",
        dtype=torch.int32,
    )

    if query_len == 1:
        def bf16_call():
            return flash_attn_with_kvcache(
                query.view(batch_size, 1, args.num_query_heads, head_dim),
                key,
                value,
                cache_seqlens=context_lens,
                block_table=block_tables,
                softmax_scale=scale,
                causal=True,
            )
    else:
        def bf16_call():
            return flash_attn_varlen_func(
                query,
                key,
                value,
                max_seqlen_q=query_len,
                cu_seqlens_q=cu_q,
                max_seqlen_k=context_len,
                cu_seqlens_k=cu_k,
                softmax_scale=scale,
                causal=True,
                block_table=block_tables,
            )

    bf16 = _timed(bf16_call, args.warmup, args.repeats)
    fp8_key, key_scale = quantize_fp8_reference(key)
    fp8_value, value_scale = quantize_fp8_reference(value)
    del key, value
    tiles = _metadata(batch_size, query_len, context_len)

    def fp8_call():
        return fp8_paged_attention(
            query,
            fp8_key,
            fp8_value,
            key_scale,
            value_scale,
            block_tables,
            context_lens,
            *tiles,
            scale,
            block_size,
        )

    fp8 = _timed(fp8_call, args.warmup, args.repeats)
    operations = (
        4
        * batch_size
        * query_len
        * context_len
        * args.num_query_heads
        * head_dim
    )
    cache_bytes = (
        batch_size
        * context_len
        * args.num_kv_heads
        * (2 * head_dim + 4)
    )
    for metrics in (bf16, fp8):
        metrics["tflops"] = operations / (metrics["median_ms"] * 1e9)
    fp8["effective_cache_gbps"] = cache_bytes / (fp8["median_ms"] * 1e6)
    return {
        "batch_size": batch_size,
        "query_len": query_len,
        "context_len": context_len,
        "head_dim": head_dim,
        "bf16": bf16,
        "fp8_e4m3": fp8,
        "speedup": bf16["median_ms"] / fp8["median_ms"],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark BF16 FlashAttention against fused FP8 paged attention"
    )
    parser.add_argument("--batch-sizes", type=_parse_ints, default=[1, 8, 32])
    parser.add_argument("--query-lens", type=_parse_ints, default=[1, 16, 128, 512])
    parser.add_argument(
        "--context-lens", type=_parse_ints, default=[128, 1024, 4096, 16384]
    )
    parser.add_argument("--head-dims", type=_parse_ints, default=[64, 128, 256])
    parser.add_argument("--num-query-heads", type=int, default=4)
    parser.add_argument("--num-kv-heads", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.num_query_heads % args.num_kv_heads:
        parser.error("query heads must be divisible by KV heads")

    results = []
    for head_dim in args.head_dims:
        for batch_size in args.batch_sizes:
            for query_len in args.query_lens:
                for context_len in args.context_lens:
                    if query_len > context_len:
                        continue
                    result = _run_case(
                        args, batch_size, query_len, context_len, head_dim
                    )
                    results.append(result)
                    print(
                        f"D={head_dim} B={batch_size} Q={query_len} "
                        f"K={context_len} bf16={result['bf16']['median_ms']:.3f}ms "
                        f"fp8={result['fp8_e4m3']['median_ms']:.3f}ms "
                        f"speedup={result['speedup']:.3f}x",
                        flush=True,
                    )
    payload = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "results": results,
    }
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)


if __name__ == "__main__":
    main()
