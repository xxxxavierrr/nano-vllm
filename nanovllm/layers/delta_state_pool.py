from __future__ import annotations

import torch

from nanovllm.layers.delta_state_layout import DeltaStateShape
from nanovllm.layers.delta_state_reference import (
    FP16_MIN_SUBNORMAL,
    dequantize_delta_state_reference,
    quantize_conv_state_reference,
    quantize_recurrent_state_reference,
)


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
            shape.layers, capacity, shape.conv_channels, shape.conv_kernel_size,
            dtype=torch.float8_e4m3fn, device=device,
        )
        self.conv_scale = torch.full(
            (shape.layers, capacity, shape.conv_channels), FP16_MIN_SUBNORMAL,
            dtype=torch.float16, device=device,
        )
        self.recurrent_payload = torch.zeros(
            shape.layers, capacity, shape.recurrent_heads,
            shape.recurrent_key_dim, shape.recurrent_value_dim,
            dtype=torch.float8_e4m3fn, device=device,
        )
        self.recurrent_scale = torch.full(
            (shape.layers, capacity, shape.recurrent_heads, shape.recurrent_key_dim),
            FP16_MIN_SUBNORMAL, dtype=torch.float16, device=device,
        )

    def _validate_slot(self, slot: int) -> None:
        if not 0 <= slot < self.capacity:
            raise IndexError("FP8 DeltaNet state slot is out of range")

    def store(
        self,
        slot: int,
        conv_state: torch.Tensor,
        recurrent_state: torch.Tensor,
    ) -> None:
        self._validate_slot(slot)
        expected_conv = (
            self.shape.layers, self.shape.conv_channels, self.shape.conv_kernel_size
        )
        expected_recurrent = (
            self.shape.layers, self.shape.recurrent_heads,
            self.shape.recurrent_key_dim, self.shape.recurrent_value_dim,
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
        self._validate_slot(slot)
        conv = dequantize_delta_state_reference(
            self.conv_payload[:, slot], self.conv_scale[:, slot], dtype=conv_dtype
        )
        recurrent = dequantize_delta_state_reference(
            self.recurrent_payload[:, slot], self.recurrent_scale[:, slot],
            dtype=torch.float32,
        )
        return conv, recurrent

    def zero(self, slot: int) -> None:
        self._validate_slot(slot)
        self.conv_payload[:, slot].zero_()
        self.conv_scale[:, slot].fill_(FP16_MIN_SUBNORMAL)
        self.recurrent_payload[:, slot].zero_()
        self.recurrent_scale[:, slot].fill_(FP16_MIN_SUBNORMAL)
