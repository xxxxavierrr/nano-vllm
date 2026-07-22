from __future__ import annotations

import torch


FP8_MAX = 448.0
FP16_MIN_SUBNORMAL = 2.0**-24


def _quantize_rows(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
    return _quantize_rows(state)


def quantize_recurrent_state_reference(
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize [..., head, K, V] with one FP16 scale per head/K row."""
    if state.ndim < 3 or not state.is_floating_point():
        raise TypeError("recurrent state must be floating [..., head, K, V]")
    return _quantize_rows(state)


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
