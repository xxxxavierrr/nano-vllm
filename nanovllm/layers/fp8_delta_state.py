from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from nanovllm.layers.delta_state_layout import (
    DeltaStateLayout,
    DeltaStateShape,
    make_delta_state_layout,
)
from nanovllm.layers.delta_state_pool import FP8DeltaStatePool
from nanovllm.layers.delta_state_reference import (
    FP8_MAX,
    FP16_MIN_SUBNORMAL,
    dequantize_delta_state_reference,
    quantize_conv_state_reference,
    quantize_recurrent_state_reference,
)



@triton.jit
def _experimental_quantize_state_rows_kernel(
    input_ptr,
    payload_ptr,
    scale_ptr,
    ROW_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < ROW_SIZE
    values = tl.load(input_ptr + row * ROW_SIZE + offsets, mask=mask, other=0.0).to(tl.float32)
    absmax = tl.max(tl.abs(values), axis=0)
    scale = tl.maximum(absmax / FP8_MAX, FP16_MIN_SUBNORMAL)
    quantized = tl.maximum(tl.minimum(values / scale, FP8_MAX), -FP8_MAX)
    tl.store(payload_ptr + row * ROW_SIZE + offsets, quantized, mask=mask)
    tl.store(scale_ptr + row, scale)


@triton.jit
def _experimental_dequantize_state_rows_kernel(
    payload_ptr,
    scale_ptr,
    output_ptr,
    ROW_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < ROW_SIZE
    scale = tl.load(scale_ptr + row).to(tl.float32)
    values = tl.load(payload_ptr + row * ROW_SIZE + offsets, mask=mask, other=0.0)
    tl.store(output_ptr + row * ROW_SIZE + offsets, values.to(tl.float32) * scale, mask=mask)


def experimental_state_kernels_enabled() -> bool:
    return os.getenv("NANOVLLM_ENABLE_EXPERIMENTAL_FP8_DELTA_STATE_KERNELS") == "1"


def experimental_quantize_state_rows(
    values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Standalone GPU validation entry; not wired into production GDN."""
    if not experimental_state_kernels_enabled():
        raise RuntimeError(
            "experimental FP8 DeltaNet kernels are disabled; set "
            "NANOVLLM_ENABLE_EXPERIMENTAL_FP8_DELTA_STATE_KERNELS=1 for GPU validation"
        )
    if not values.is_cuda or values.ndim != 2 or not values.is_contiguous():
        raise ValueError("experimental state quantization requires contiguous CUDA [rows, width]")
    payload = torch.empty_like(values, dtype=torch.float8_e4m3fn)
    scale = torch.empty(values.shape[0], device=values.device, dtype=torch.float16)
    block = triton.next_power_of_2(values.shape[1])
    _experimental_quantize_state_rows_kernel[(values.shape[0],)](
        values,
        payload,
        scale,
        ROW_SIZE=values.shape[1],
        BLOCK_SIZE=block,
        num_warps=4,
    )
    return payload, scale
