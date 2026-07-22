from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

from nanovllm.layers.gptq_native import (
    W4Backend,
    W4Kernel,
    native_w4a16_linear,
    select_w4_kernel,
    validate_native_w4_layout,
)


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
    input_perm_ptr,
    output_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    ZERO_PACKED_N: tl.constexpr,
    SYMMETRIC_ZERO: tl.constexpr,
    USE_INPUT_PERM: tl.constexpr,
    DIRECT_GROUPS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
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
        if USE_INPUT_PERM:
            input_k = tl.load(
                input_perm_ptr + offs_k,
                mask=offs_k < K,
                other=0,
            )
        else:
            input_k = offs_k
        x = tl.load(
            x_ptr + offs_m[:, None] * K + input_k[None, :],
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

        if DIRECT_GROUPS:
            group = offs_k // GROUP_SIZE
        else:
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


@triton.jit
def _repack_gptq_qweight_kernel(
    qweight_ptr,
    input_perm_ptr,
    output_ptr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Repack K-packed GPTQ weights into argsort(g_idx) runtime order."""
    packed_k = tl.program_id(axis=0)
    offs_n = tl.program_id(axis=1) * BLOCK_N + tl.arange(0, BLOCK_N)
    packed = tl.zeros((BLOCK_N,), dtype=tl.int32)
    for lane in range(8):
        runtime_k = packed_k * 8 + lane
        checkpoint_k = tl.load(input_perm_ptr + runtime_k)
        checkpoint_word = tl.load(
            qweight_ptr + (checkpoint_k // 8) * N + offs_n,
            mask=offs_n < N,
            other=0,
        )
        nibble = (checkpoint_word >> ((checkpoint_k % 8) * 4)) & 0xF
        packed |= nibble << (lane * 4)
    tl.store(
        output_ptr + packed_k * N + offs_n,
        packed,
        mask=offs_n < N,
    )


def _validate_inputs(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    g_idx: torch.Tensor,
    input_perm: torch.Tensor,
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
    if input_perm.dtype is not torch.int32 or input_perm.ndim != 1:
        raise TypeError("input_perm must be a rank-1 int32 tensor")
    tensors = (qweight, scales, qzeros, g_idx, input_perm)
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
    if input_perm.numel() not in (0, k):
        raise ValueError(f"input_perm must be empty or contain {k} entries")
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
    input_perm: torch.Tensor,
    symmetric_zero: bool,
    direct_groups: bool,
    group_size: int,
) -> torch.Tensor:
    m, n, k = _validate_inputs(
        x, qweight, scales, qzeros, g_idx, input_perm
    )
    if group_size <= 0 or k % group_size:
        raise ValueError("runtime GPTQ group_size must divide K")
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
        input_perm,
        output,
        M=m,
        N=n,
        K=k,
        ZERO_PACKED_N=triton.cdiv(n, 8),
        SYMMETRIC_ZERO=symmetric_zero,
        USE_INPUT_PERM=input_perm.numel() != 0,
        DIRECT_GROUPS=direct_groups,
        GROUP_SIZE=group_size,
    )
    return output


@triton_op("nanovllm::repack_gptq_qweight", mutates_args={})
def repack_gptq_qweight(
    qweight: torch.Tensor,
    input_perm: torch.Tensor,
) -> torch.Tensor:
    if (
        not qweight.is_cuda
        or qweight.dtype is not torch.int32
        or qweight.ndim != 2
        or not qweight.is_contiguous()
    ):
        raise ValueError("qweight repack requires contiguous CUDA int32 [K/8, N]")
    k = qweight.shape[0] * 8
    if (
        input_perm.device != qweight.device
        or input_perm.dtype is not torch.int32
        or input_perm.shape != (k,)
        or not input_perm.is_contiguous()
    ):
        raise ValueError("input permutation must be contiguous CUDA int32 [K]")
    output = torch.empty_like(qweight)
    block_n = 128
    wrap_triton(_repack_gptq_qweight_kernel)[
        (qweight.shape[0], triton.cdiv(qweight.shape[1], block_n))
    ](
        qweight,
        input_perm,
        output,
        N=qweight.shape[1],
        BLOCK_N=block_n,
        num_warps=4,
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
    input_perm: torch.Tensor | None = None,
    direct_groups: bool = False,
    group_size: int = 128,
    backend: W4Backend | str = W4Backend.AUTO,
) -> torch.Tensor:
    input_shape = x.shape
    m = x.numel() // x.shape[-1]
    capability = torch.cuda.get_device_capability(x.device) if x.is_cuda else None
    selected = select_w4_kernel(m, backend, capability=capability)
    if selected is not W4Kernel.TRITON:
        validate_native_w4_layout(
            symmetric_zero=symmetric_zero,
            direct_groups=direct_groups,
            input_perm_numel=0 if input_perm is None else input_perm.numel(),
            k=x.shape[-1],
            group_size=group_size,
        )
        return native_w4a16_linear(
            x,
            qweight,
            scales,
            bias,
            group_size=group_size,
            kernel=selected,
        )
    if input_perm is None:
        input_perm = torch.empty(0, dtype=torch.int32, device=x.device)
    output = _gptq_w4a16(
        x,
        qweight,
        scales,
        qzeros,
        g_idx,
        input_perm,
        symmetric_zero,
        direct_groups,
        group_size,
    )
    if bias is not None:
        output = output + bias
    return output.reshape(*input_shape[:-1], qweight.shape[1])
