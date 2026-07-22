from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


GPTQ_BITS = 4
GPTQ_VALUES_PER_INT32 = 32 // GPTQ_BITS


@dataclass(frozen=True, slots=True)
class GPTQConfig:
    bits: int = GPTQ_BITS
    group_size: int = 128
    sym: bool = True
    desc_act: bool = False
    pack_dtype: str = "int32"

    @classmethod
    def from_dict(cls, value: dict) -> "GPTQConfig":
        method = value.get("quant_method", value.get("format", "gptq"))
        checkpoint_format = value.get("checkpoint_format", value.get("format", "gptq"))
        if str(method).lower() != "gptq" or str(checkpoint_format).lower() != "gptq":
            raise ValueError(
                "only GPTQ checkpoint format is supported, got "
                f"quant_method={method!r}, checkpoint_format={checkpoint_format!r}"
            )
        config = cls(
            bits=int(value.get("bits", GPTQ_BITS)),
            group_size=int(value.get("group_size", 128)),
            sym=bool(value.get("sym", True)),
            desc_act=bool(value.get("desc_act", False)),
            pack_dtype=str(value.get("pack_dtype", "int32")).lower(),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.bits != GPTQ_BITS:
            raise ValueError(f"GPTQ v1 requires 4 bits, got {self.bits}")
        if self.group_size != 128:
            raise ValueError(
                f"GPTQ v1 requires group_size=128, got {self.group_size}"
            )
        if not self.sym:
            raise ValueError("GPTQ v1 requires symmetric quantization")
        if self.pack_dtype not in ("int32", "torch.int32"):
            raise ValueError(
                f"GPTQ v1 requires int32 packing, got {self.pack_dtype!r}"
            )

    @property
    def values_per_int32(self) -> int:
        return 32 // self.bits


def _unpack_int4(packed: torch.Tensor, dim: int) -> torch.Tensor:
    """Unpack unsigned INT4 values stored eight-per-INT32."""
    if packed.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"packed GPTQ tensor must be int32/int64, got {packed.dtype}")
    if dim < 0:
        dim += packed.ndim
    if not 0 <= dim < packed.ndim:
        raise IndexError(f"invalid unpack dimension {dim} for rank {packed.ndim}")

    shifts_shape = [1] * (packed.ndim + 1)
    shifts_shape[dim + 1] = GPTQ_VALUES_PER_INT32
    shifts = (
        torch.arange(GPTQ_VALUES_PER_INT32, device=packed.device, dtype=torch.int64)
        .mul_(GPTQ_BITS)
        .view(shifts_shape)
    )
    expanded = packed.to(torch.int64).unsqueeze(dim + 1)
    unpacked = torch.bitwise_and(torch.bitwise_right_shift(expanded, shifts), 0xF)
    shape = list(packed.shape)
    shape[dim] *= GPTQ_VALUES_PER_INT32
    return unpacked.reshape(shape).to(torch.int32)


def unpack_gptq_qweight(qweight: torch.Tensor) -> torch.Tensor:
    """Return GPTQ qweight as logical unsigned INT4 values shaped [K, N]."""
    if qweight.ndim != 2:
        raise ValueError(f"qweight must be rank 2, got shape {tuple(qweight.shape)}")
    return _unpack_int4(qweight, dim=0)


def unpack_gptq_qzeros(qzeros: torch.Tensor, out_features: int | None = None) -> torch.Tensor:
    """Return logical GPTQ zero points shaped [groups, N].

    GPTQ stores ``zero_point - 1`` in qzeros, so one is added after
    unpacking. The optional output width removes packing padding.
    """
    if qzeros.ndim != 2:
        raise ValueError(f"qzeros must be rank 2, got shape {tuple(qzeros.shape)}")
    zeros = _unpack_int4(qzeros, dim=1).add_(1)
    if out_features is not None:
        if not 0 < out_features <= zeros.shape[1]:
            raise ValueError(
                f"out_features must be in [1, {zeros.shape[1]}], got {out_features}"
            )
        zeros = zeros[:, :out_features]
    return zeros


def default_g_idx(
    in_features: int,
    num_groups: int,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    if in_features <= 0 or num_groups <= 0 or in_features % num_groups != 0:
        raise ValueError(
            "in_features must be positive and divisible by the number of groups"
        )
    group_size = in_features // num_groups
    return torch.arange(in_features, device=device, dtype=torch.int32).div(
        group_size, rounding_mode="floor"
    )


def repack_gptq_qweight_reference(
    qweight: torch.Tensor,
    input_perm: torch.Tensor,
) -> torch.Tensor:
    """CPU/test reference for K-order GPTQ repacking without BF16 weights."""
    if qweight.dtype is not torch.int32 or qweight.ndim != 2:
        raise TypeError("qweight must be rank-2 int32")
    k = qweight.shape[0] * GPTQ_VALUES_PER_INT32
    if input_perm.dtype is not torch.int32 or input_perm.shape != (k,):
        raise TypeError("input_perm must be int32 [K]")
    output = torch.zeros_like(qweight, dtype=torch.int64)
    source = qweight.to(torch.int64)
    for lane in range(GPTQ_VALUES_PER_INT32):
        checkpoint_k = input_perm[lane::GPTQ_VALUES_PER_INT32].long()
        checkpoint_word = source.index_select(0, checkpoint_k // 8)
        shift = ((checkpoint_k % 8) * GPTQ_BITS).unsqueeze(1)
        nibble = torch.bitwise_and(
            torch.bitwise_right_shift(checkpoint_word, shift),
            0xF,
        )
        output.bitwise_or_(nibble << (lane * GPTQ_BITS))
    return output.to(torch.int32)


def dequantize_gptq_weight(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    g_idx: torch.Tensor | None = None,
) -> torch.Tensor:
    """Materialize a GPTQ INT4 weight as [out_features, in_features].

    This is a correctness reference only. Runtime inference must keep qweight
    packed and dequantize tiles inside the W4A16 kernel.
    """
    if qweight.ndim != 2:
        raise ValueError(f"qweight must be rank 2, got shape {tuple(qweight.shape)}")
    if scales.ndim != 2:
        raise ValueError(f"scales must be rank 2, got shape {tuple(scales.shape)}")
    if not scales.is_floating_point():
        raise TypeError(f"scales must be floating point, got {scales.dtype}")
    if qzeros.ndim != 2:
        raise ValueError(f"qzeros must be rank 2, got shape {tuple(qzeros.shape)}")
    in_features = qweight.shape[0] * GPTQ_VALUES_PER_INT32
    num_groups, out_features = scales.shape
    if qweight.shape[1] != out_features:
        raise ValueError(
            f"qweight output width {qweight.shape[1]} does not match scales {out_features}"
        )
    if qzeros.shape[0] != num_groups:
        raise ValueError(
            f"qzeros group count {qzeros.shape[0]} does not match scales {num_groups}"
        )
    if g_idx is None:
        g_idx = default_g_idx(in_features, num_groups, qweight.device)
    if g_idx.ndim != 1 or g_idx.numel() != in_features:
        raise ValueError(f"g_idx must have shape [{in_features}], got {tuple(g_idx.shape)}")
    g_idx = g_idx.to(device=scales.device, dtype=torch.long)
    if g_idx.numel() and (g_idx.min().item() < 0 or g_idx.max().item() >= num_groups):
        raise ValueError(f"g_idx values must be in [0, {num_groups})")

    values = unpack_gptq_qweight(qweight).to(device=scales.device)
    zeros = unpack_gptq_qzeros(qzeros, out_features).to(device=scales.device)
    selected_scales = scales.index_select(0, g_idx)
    selected_zeros = zeros.index_select(0, g_idx)
    weight_kn = (values.to(scales.dtype) - selected_zeros.to(scales.dtype)) * selected_scales
    return weight_kn.t().contiguous()


def gptq_linear_reference(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    g_idx: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Slow reference implementation used by tests and kernel validation."""
    weight = dequantize_gptq_weight(qweight, scales, qzeros, g_idx).to(x.dtype)
    return F.linear(x, weight, bias)
