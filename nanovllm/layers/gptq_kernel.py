from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton


_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 32, "BLOCK_K": 32}, num_warps=4),
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8),
]


@triton.autotune(configs=_AUTOTUNE_CONFIGS, key=["M", "N", "K"])
@triton.jit
def _gptq_w4a16_kernel(
    x_ptr,
    qweight_ptr,
    scales_ptr,
    qzeros_ptr,
    g_idx_ptr,
    output_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    ZERO_PACKED_N: tl.constexpr,
    SYMMETRIC_ZERO: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        x = tl.load(
            x_ptr + offs_m[:, None] * K + offs_k[None, :],
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        )

        # Load each packed int32 exactly once, then expand its eight INT4
        # values in registers. The reference kernel addressed qweight with
        # offs_k // 8 and therefore presented every packed address eight
        # times to the compiler.
        packed_k = k_start // 8 + tl.arange(0, BLOCK_K // 8)
        packed_weight = tl.load(
            qweight_ptr + packed_k[:, None] * N + offs_n[None, :],
            mask=(packed_k[:, None] < K // 8) & (offs_n[None, :] < N),
            other=0,
        )
        weight_shift = tl.arange(0, 8) * 4
        weight = (
            (packed_weight[:, None, :] >> weight_shift[None, :, None]) & 0xF
        ).reshape((BLOCK_K, BLOCK_N))

        group = tl.load(g_idx_ptr + offs_k, mask=offs_k < K, other=0)
        if SYMMETRIC_ZERO:
            zero = 8.0
        else:
            packed_n = offs_n // 8
            zero_shift = (offs_n % 8) * 4
            packed_zero = tl.load(
                qzeros_ptr + group[:, None] * ZERO_PACKED_N + packed_n[None, :],
                mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
                other=0,
            )
            zero = ((packed_zero >> zero_shift[None, :]) & 0xF) + 1
        scale = tl.load(
            scales_ptr + group[:, None] * N + offs_n[None, :],
            mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        dequantized = ((weight - zero).to(tl.float32) * scale).to(tl.bfloat16)
        acc = tl.dot(x, dequantized, acc)

    tl.store(
        output_ptr + offs_m[:, None] * N + offs_n[None, :],
        acc.to(tl.bfloat16),
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def _validate_inputs(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    g_idx: torch.Tensor,
) -> tuple[int, int, int]:
    if not x.is_cuda:
        raise ValueError("GPTQ W4A16 kernel requires CUDA tensors")
    if x.dtype is not torch.bfloat16:
        raise TypeError(f"x must be bfloat16, got {x.dtype}")
    if qweight.dtype is not torch.int32 or qweight.ndim != 2:
        raise TypeError("qweight must be a rank-2 int32 tensor")
    if qzeros.dtype is not torch.int32 or qzeros.ndim != 2:
        raise TypeError("qzeros must be a rank-2 int32 tensor")
    if scales.dtype not in (torch.bfloat16, torch.float16) or scales.ndim != 2:
        raise TypeError("scales must be a rank-2 bfloat16/float16 tensor")
    if g_idx.dtype is not torch.int32 or g_idx.ndim != 1:
        raise TypeError("g_idx must be a rank-1 int32 tensor")
    tensors = (qweight, scales, qzeros, g_idx)
    if any(t.device != x.device for t in tensors):
        raise ValueError("all GPTQ tensors must be on the same CUDA device")

    k = qweight.shape[0] * 8
    n = qweight.shape[1]
    if x.shape[-1] != k:
        raise ValueError(f"x K={x.shape[-1]} does not match qweight K={k}")
    if scales.shape[1] != n:
        raise ValueError("scales output width does not match qweight")
    if qzeros.shape != (scales.shape[0], triton.cdiv(n, 8)):
        raise ValueError("qzeros shape does not match scales/output packing")
    if g_idx.numel() != k:
        raise ValueError(f"g_idx must contain {k} entries")
    if not all(t.is_contiguous() for t in tensors):
        raise ValueError("packed GPTQ tensors must be contiguous")
    m = x.numel() // k
    return m, n, k


@triton_op("nanovllm::gptq_w4a16", mutates_args={})
def _gptq_w4a16(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    g_idx: torch.Tensor,
    symmetric_zero: bool,
) -> torch.Tensor:
    m, n, k = _validate_inputs(x, qweight, scales, qzeros, g_idx)
    x_2d = x.reshape(m, k)
    if not x_2d.is_contiguous():
        x_2d = x_2d.contiguous()
    output = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    grid = lambda meta: (
        triton.cdiv(m, meta["BLOCK_M"]),
        triton.cdiv(n, meta["BLOCK_N"]),
    )
    wrap_triton(_gptq_w4a16_kernel)[grid](
        x_2d,
        qweight,
        scales,
        qzeros,
        g_idx,
        output,
        M=m,
        N=n,
        K=k,
        ZERO_PACKED_N=triton.cdiv(n, 8),
        SYMMETRIC_ZERO=symmetric_zero,
    )
    return output


def gptq_w4a16_linear(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    g_idx: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    symmetric_zero: bool = False,
) -> torch.Tensor:
    input_shape = x.shape
    output = _gptq_w4a16(x, qweight, scales, qzeros, g_idx, symmetric_zero)
    if bias is not None:
        output = output + bias
    return output.reshape(*input_shape[:-1], qweight.shape[1])
