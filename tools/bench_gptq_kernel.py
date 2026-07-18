from __future__ import annotations

import argparse
import json
import time

import torch
import torch.nn.functional as F

from nanovllm.layers.gptq import dequantize_gptq_weight
from nanovllm.layers.gptq_kernel import gptq_w4a16_linear


def pack_int4(values: torch.Tensor, dim: int) -> torch.Tensor:
    shape = list(values.shape)
    shape[dim] //= 8
    shape.insert(dim + 1, 8)
    chunks = values.to(torch.int64).reshape(shape)
    shifts_shape = [1] * chunks.ndim
    shifts_shape[dim + 1] = 8
    shifts = (torch.arange(8, device=values.device) * 4).view(shifts_shape)
    return torch.sum(chunks << shifts, dim=dim + 1).to(torch.int32)


def measure(fn, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000 / repeats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=3584)
    parser.add_argument("--n", type=int, default=3584)
    parser.add_argument("--m", type=int, nargs="+", default=[1, 8, 19, 64, 128, 512])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--ordered-g-idx", action="store_true")
    args = parser.parse_args()

    if args.k % 128 or args.k % 8 or args.n % 8:
        raise ValueError("K must be divisible by 128 and N by 8")
    torch.manual_seed(0)
    groups = args.k // 128
    values = torch.randint(0, 16, (args.k, args.n), device="cuda", dtype=torch.int32)
    zeros = torch.randint(1, 17, (groups, args.n), device="cuda", dtype=torch.int32)
    scales = (
        torch.rand(groups, args.n, device="cuda", dtype=torch.float32) * 0.05 + 0.005
    ).to(torch.bfloat16)
    if args.ordered_g_idx:
        g_idx = torch.arange(args.k, device="cuda", dtype=torch.int32) // 128
    else:
        group_order = torch.randperm(groups, device="cuda")
        g_idx = group_order[torch.arange(args.k, device="cuda") // 128].to(torch.int32)
    qweight = pack_int4(values, 0)
    qzeros = pack_int4(zeros - 1, 1)
    bf16_weight = dequantize_gptq_weight(qweight, scales, qzeros, g_idx)

    rows = []
    for m in args.m:
        x = torch.randn(m, args.k, device="cuda", dtype=torch.bfloat16)
        gptq_fn = lambda: gptq_w4a16_linear(x, qweight, scales, qzeros, g_idx)
        bf16_fn = lambda: F.linear(x, bf16_weight)
        gptq_ms = measure(gptq_fn, args.warmup, args.repeats)
        bf16_ms = measure(bf16_fn, args.warmup, args.repeats)
        expected = bf16_fn()
        actual = gptq_fn()
        error = (actual - expected).float()
        max_abs_error = error.abs().max().item()
        relative_l2_error = error.norm().div(expected.float().norm()).item()
        flops = 2 * m * args.k * args.n
        packed_bytes = (
            qweight.numel() * qweight.element_size()
            + scales.numel() * scales.element_size()
            + qzeros.numel() * qzeros.element_size()
            + g_idx.numel() * g_idx.element_size()
        )
        bf16_weight_bytes = args.k * args.n * torch.bfloat16.itemsize
        compression_ratio = bf16_weight_bytes / packed_bytes
        rows.append(
            {
                "m": m,
                "k": args.k,
                "n": args.n,
                "gptq_ms": gptq_ms,
                "bf16_ms": bf16_ms,
                "speedup_vs_bf16": bf16_ms / gptq_ms,
                "gptq_tflops": flops / (gptq_ms * 1e9),
                "effective_weight_gb_s": packed_bytes / (gptq_ms * 1e6),
                "max_abs_error": max_abs_error,
                "relative_l2_error": relative_l2_error,
                "packed_weight_bytes": packed_bytes,
                "bf16_weight_bytes": bf16_weight_bytes,
                "weight_compression_ratio": compression_ratio,
            }
        )
    print(json.dumps({"ordered_g_idx": args.ordered_g_idx, "results": rows}, indent=2))


if __name__ == "__main__":
    main()
