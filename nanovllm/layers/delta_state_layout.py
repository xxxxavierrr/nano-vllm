from __future__ import annotations

from dataclasses import asdict, dataclass


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


def _state_value_counts(shape: DeltaStateShape) -> tuple[int, int]:
    conv = shape.layers * shape.conv_channels * shape.conv_kernel_size
    recurrent = (
        shape.layers
        * shape.recurrent_heads
        * shape.recurrent_key_dim
        * shape.recurrent_value_dim
    )
    return conv, recurrent


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
    conv_values, recurrent_values = _state_value_counts(shape)
    if dtype == "auto":
        conv_payload = conv_values * native_conv_bytes
        recurrent_payload = recurrent_values * 4
        return DeltaStateLayout(
            "native", "none", conv_payload, 0, recurrent_payload, 0,
            conv_payload + recurrent_payload,
        )
    conv_scales = shape.layers * shape.conv_channels * 2
    recurrent_scales = shape.layers * shape.recurrent_heads * shape.recurrent_key_dim * 2
    return DeltaStateLayout(
        "fp8_e4m3",
        "conv_per_channel,recurrent_per_head_k_row",
        conv_values,
        conv_scales,
        recurrent_values,
        recurrent_scales,
        conv_values + conv_scales + recurrent_values + recurrent_scales,
    )
