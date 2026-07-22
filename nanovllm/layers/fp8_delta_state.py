from __future__ import annotations

import os
from dataclasses import asdict, dataclass

import torch
import triton
import triton.language as tl


FP8_MAX = 448.0
FP16_MIN_SUBNORMAL = 2.0**-24


@dataclass(frozen=True, slots=True)
class DeltaStateShape:
    layers: int
    conv_channels: int
    conv_kernel_size: int
    recurrent_heads: int
    recurrent_key_dim: int
    recurrent_value_dim: int

    def validate(self) -> None:
        if any(value <= 0 for value in asdict(self).values()):
            raise ValueError("DeltaNet state dimensions must be positive")


@dataclass(frozen=True, slots=True)
class DeltaStateLayout:
    dtype: str
    scale_mode: str
    conv_payload_bytes: int
    conv_scale_bytes: int
    recurrent_payload_bytes: int
    recurrent_scale_bytes: int
    bytes_per_slot: int

    def report(self, *, request_capacity: int, branch_slots_per_request: int) -> dict:
        if request_capacity < 0 or branch_slots_per_request < 0:
            raise ValueError("DeltaNet state capacities cannot be negative")
        total_slots = request_capacity * (1 + branch_slots_per_request)
        return {
            **asdict(self),
            "request_capacity": request_capacity,
            "branch_slots_per_request": branch_slots_per_request,
            "total_slots": total_slots,
            "total_bytes": total_slots * self.bytes_per_slot,
            "scale_overhead_ratio": (
                (self.conv_scale_bytes + self.recurrent_scale_bytes)
                / self.bytes_per_slot
            ),
        }


def make_delta_state_layout(
    shape: DeltaStateShape,
    *,
    dtype: str,
    native_conv_bytes: int = 2,
) -> DeltaStateLayout:
    shape.validate()
    if dtype not in {"auto", "fp8_e4m3"}:
        raise ValueError("delta state dtype must be auto or fp8_e4m3")
    if native_conv_bytes <= 0:
        raise ValueError("native_conv_bytes must be positive")
    conv_values = (
        shape.layers * shape.conv_channels * shape.conv_kernel_size
    )
    recurrent_values = (
        shape.layers
        * shape.recurrent_heads
        * shape.recurrent_key_dim
        * shape.recurrent_value_dim
    )
    if dtype == "auto":
        conv_payload = conv_values * native_conv_bytes
        recurrent_payload = recurrent_values * 4
        return DeltaStateLayout(
            dtype="native",
            scale_mode="none",
            conv_payload_bytes=conv_payload,
            conv_scale_bytes=0,
            recurrent_payload_bytes=recurrent_payload,
            recurrent_scale_bytes=0,
            bytes_per_slot=conv_payload + recurrent_payload,
        )
    conv_payload = conv_values
    conv_scales = shape.layers * shape.conv_channels * 2
    recurrent_payload = recurrent_values
    recurrent_scales = (
        shape.layers * shape.recurrent_heads * shape.recurrent_key_dim * 2
    )
    return DeltaStateLayout(
        dtype="fp8_e4m3",
        scale_mode="conv_per_channel,recurrent_per_head_k_row",
        conv_payload_bytes=conv_payload,
        conv_scale_bytes=conv_scales,
        recurrent_payload_bytes=recurrent_payload,
        recurrent_scale_bytes=recurrent_scales,
        bytes_per_slot=(
            conv_payload + conv_scales + recurrent_payload + recurrent_scales
        ),
    )


def _quantize_rows_reference(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.nan_to_num(
        values.float(), nan=0.0, posinf=FP8_MAX, neginf=-FP8_MAX
    )
    scale = (
        values.abs().amax(dim=-1)
        .div(FP8_MAX)
        .clamp_min(FP16_MIN_SUBNORMAL)
        .to(torch.float16)
    )
    payload = (
        values.div(scale.float().unsqueeze(-1))
        .clamp(-FP8_MAX, FP8_MAX)
        .to(torch.float8_e4m3fn)
    )
    return payload, scale


def quantize_conv_state_reference(
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize [..., channel, kernel] with one FP16 scale per channel."""
    if state.ndim < 2 or not state.is_floating_point():
        raise TypeError("conv state must be floating [..., channel, kernel]")
    return _quantize_rows_reference(state)


def quantize_recurrent_state_reference(
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize [..., head, K, V] with one FP16 scale per head/K row."""
    if state.ndim < 3 or not state.is_floating_point():
        raise TypeError("recurrent state must be floating [..., head, K, V]")
    return _quantize_rows_reference(state)


def dequantize_delta_state_reference(
    payload: torch.Tensor,
    scale: torch.Tensor,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if payload.dtype is not torch.float8_e4m3fn:
        raise TypeError("DeltaNet FP8 payload must use float8_e4m3fn")
    if scale.dtype is not torch.float16 or scale.shape != payload.shape[:-1]:
        raise TypeError("DeltaNet FP8 scale shape/dtype mismatch")
    return (payload.float() * scale.float().unsqueeze(-1)).to(dtype)


class FP8DeltaStatePool:
    """CPU/reference slot pool sharing production committed/branch slot IDs."""

    def __init__(
        self,
        shape: DeltaStateShape,
        capacity: int,
        *,
        device: torch.device | str = "cpu",
    ):
        shape.validate()
        if capacity <= 0:
            raise ValueError("FP8 DeltaNet state pool capacity must be positive")
        self.shape = shape
        self.capacity = capacity
        self.conv_payload = torch.zeros(
            shape.layers,
            capacity,
            shape.conv_channels,
            shape.conv_kernel_size,
            dtype=torch.float8_e4m3fn,
            device=device,
        )
        self.conv_scale = torch.full(
            (shape.layers, capacity, shape.conv_channels),
            FP16_MIN_SUBNORMAL,
            dtype=torch.float16,
            device=device,
        )
        self.recurrent_payload = torch.zeros(
            shape.layers,
            capacity,
            shape.recurrent_heads,
            shape.recurrent_key_dim,
            shape.recurrent_value_dim,
            dtype=torch.float8_e4m3fn,
            device=device,
        )
        self.recurrent_scale = torch.full(
            (
                shape.layers,
                capacity,
                shape.recurrent_heads,
                shape.recurrent_key_dim,
            ),
            FP16_MIN_SUBNORMAL,
            dtype=torch.float16,
            device=device,
        )

    def store(
        self,
        slot: int,
        conv_state: torch.Tensor,
        recurrent_state: torch.Tensor,
    ) -> None:
        if not 0 <= slot < self.capacity:
            raise IndexError("FP8 DeltaNet state slot is out of range")
        expected_conv = (
            self.shape.layers,
            self.shape.conv_channels,
            self.shape.conv_kernel_size,
        )
        expected_recurrent = (
            self.shape.layers,
            self.shape.recurrent_heads,
            self.shape.recurrent_key_dim,
            self.shape.recurrent_value_dim,
        )
        if conv_state.shape != expected_conv or recurrent_state.shape != expected_recurrent:
            raise ValueError("FP8 DeltaNet state shape mismatch")
        conv_payload, conv_scale = quantize_conv_state_reference(conv_state)
        recurrent_payload, recurrent_scale = quantize_recurrent_state_reference(
            recurrent_state
        )
        self.conv_payload[:, slot].copy_(conv_payload)
        self.conv_scale[:, slot].copy_(conv_scale)
        self.recurrent_payload[:, slot].copy_(recurrent_payload)
        self.recurrent_scale[:, slot].copy_(recurrent_scale)

    def load(
        self,
        slot: int,
        *,
        conv_dtype: torch.dtype = torch.bfloat16,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not 0 <= slot < self.capacity:
            raise IndexError("FP8 DeltaNet state slot is out of range")
        return (
            dequantize_delta_state_reference(
                self.conv_payload[:, slot], self.conv_scale[:, slot], dtype=conv_dtype
            ),
            dequantize_delta_state_reference(
                self.recurrent_payload[:, slot],
                self.recurrent_scale[:, slot],
                dtype=torch.float32,
            ),
        )

    def zero(self, slot: int) -> None:
        self.conv_payload[:, slot].zero_()
        self.conv_scale[:, slot].fill_(FP16_MIN_SUBNORMAL)
        self.recurrent_payload[:, slot].zero_()
        self.recurrent_scale[:, slot].fill_(FP16_MIN_SUBNORMAL)


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
