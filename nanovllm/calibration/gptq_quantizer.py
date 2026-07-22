from __future__ import annotations

from dataclasses import dataclass

import torch

from nanovllm.layers.gptq import GPTQ_VALUES_PER_INT32


@dataclass(frozen=True, slots=True)
class GPTQQuantizerConfig:
    bits: int = 4
    group_size: int = 128
    block_size: int = 128
    damping_percent: float = 0.01
    sym: bool = True
    desc_act: bool = False

    def validate(self) -> None:
        if self.bits != 4 or not self.sym or self.desc_act:
            raise ValueError("draft GPTQ requires symmetric 4-bit desc_act=false")
        if self.group_size != 128 or self.block_size != 128:
            raise ValueError("draft GPTQ requires group_size=block_size=128")
        if not 0 < self.damping_percent < 1:
            raise ValueError("GPTQ damping_percent must be in (0, 1)")

    def as_checkpoint_dict(self) -> dict:
        self.validate()
        return {
            "quant_method": "gptq",
            "checkpoint_format": "gptq",
            "bits": self.bits,
            "group_size": self.group_size,
            "sym": self.sym,
            "desc_act": self.desc_act,
            "pack_dtype": "int32",
            "damp_percent": self.damping_percent,
            "block_size": self.block_size,
        }


class HessianAccumulator:
    def __init__(self, in_features: int):
        if in_features <= 0:
            raise ValueError("in_features must be positive")
        self.in_features = in_features
        self.hessian = torch.zeros(
            (in_features, in_features), dtype=torch.float32, device="cpu"
        )
        self.num_samples = 0

    def add(self, activations: torch.Tensor) -> None:
        if activations.shape[-1] != self.in_features:
            raise ValueError("calibration activation width mismatch")
        values = activations.detach().reshape(-1, self.in_features).float().cpu()
        if not torch.isfinite(values).all():
            raise ValueError("calibration activations contain non-finite values")
        self.hessian.addmm_(values.t(), values)
        self.num_samples += values.shape[0]

    def finalize(self, damping_percent: float) -> torch.Tensor:
        if self.num_samples == 0:
            raise ValueError("cannot finalize an empty Hessian")
        hessian = self.hessian / self.num_samples
        diagonal_mean = torch.diag(hessian).mean()
        damping = (diagonal_mean * damping_percent).clamp_min(
            torch.finfo(torch.float32).eps
        )
        indices = torch.arange(self.in_features)
        hessian[indices, indices] += damping
        return hessian


def _pack_int4(values: torch.Tensor, dim: int, pad_value: int = 0) -> torch.Tensor:
    padding = (-values.shape[dim]) % GPTQ_VALUES_PER_INT32
    if padding:
        pad_shape = list(values.shape)
        pad_shape[dim] = padding
        values = torch.cat(
            (values, torch.full(pad_shape, pad_value, dtype=values.dtype)), dim=dim
        )
    shape = list(values.shape)
    shape[dim] //= GPTQ_VALUES_PER_INT32
    shape.insert(dim + 1, GPTQ_VALUES_PER_INT32)
    lanes = values.to(torch.int64).reshape(shape)
    shift_shape = [1] * lanes.ndim
    shift_shape[dim + 1] = GPTQ_VALUES_PER_INT32
    shifts = (torch.arange(GPTQ_VALUES_PER_INT32) * 4).view(shift_shape)
    return torch.sum(lanes << shifts, dim=dim + 1).to(torch.int32)


def _prepare_quantization(
    weight: torch.Tensor,
    hessian: torch.Tensor,
    config: GPTQQuantizerConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    config.validate()
    if weight.ndim != 2 or not weight.is_floating_point():
        raise TypeError("GPTQ weight must be floating [out_features, in_features]")
    out_features, in_features = weight.shape
    if in_features % config.group_size:
        raise ValueError("GPTQ in_features must be divisible by group_size")
    if hessian.shape != (in_features, in_features) or hessian.dtype is not torch.float32:
        raise TypeError("GPTQ Hessian must be FP32 [in_features, in_features]")

    working = weight.detach().float().cpu().clone()
    conditioned = hessian.detach().float().cpu().clone()
    dead = torch.diag(conditioned) <= 0
    if bool(dead.any()):
        conditioned[dead, dead] = 1
        working[:, dead] = 0
    try:
        inverse = torch.linalg.cholesky(conditioned)
        inverse = torch.cholesky_inverse(inverse)
        return working, torch.linalg.cholesky(inverse, upper=True)
    except RuntimeError as exc:
        raise ValueError("damped GPTQ Hessian is not positive definite") from exc


def _group_scales(
    working: torch.Tensor,
    config: GPTQQuantizerConfig,
) -> torch.Tensor:
    out_features, in_features = working.shape
    groups = in_features // config.group_size
    scales = torch.empty((groups, out_features), dtype=torch.float32)
    for group in range(groups):
        begin = group * config.group_size
        end = begin + config.group_size
        # Stored zero eight represents signed values [-8, 7]. Dividing by
        # seven preserves the positive maximum while retaining -8.
        scales[group] = (
            working[:, begin:end].abs().amax(dim=1).clamp_min(1.0e-12) / 7
        )
    return scales


def _quantize_columns(
    working: torch.Tensor,
    inverse: torch.Tensor,
    scales: torch.Tensor,
    config: GPTQQuantizerConfig,
) -> torch.Tensor:
    out_features, in_features = working.shape
    quantized = torch.empty((in_features, out_features), dtype=torch.int32)
    for block_start in range(0, in_features, config.block_size):
        block_end = min(block_start + config.block_size, in_features)
        errors = torch.zeros((out_features, block_end - block_start))
        for column in range(block_start, block_end):
            scale = scales[column // config.group_size]
            values = torch.round(working[:, column] / scale).add(8).clamp(0, 15)
            quantized[column] = values.to(torch.int32)
            diagonal = inverse[column, column].clamp_min(torch.finfo(torch.float32).eps)
            error = (working[:, column] - (values - 8) * scale) / diagonal
            errors[:, column - block_start] = error
            working[:, column:block_end] -= error.unsqueeze(1) * inverse[column, column:block_end]
        if block_end < in_features:
            working[:, block_end:] -= errors @ inverse[block_start:block_end, block_end:]
    return quantized


def _pack_quantized(
    quantized: torch.Tensor,
    scales: torch.Tensor,
    config: GPTQQuantizerConfig,
) -> dict[str, torch.Tensor]:
    in_features, out_features = quantized.shape
    stored_zero = 7
    logical_zeros = torch.full(
        (scales.shape[0], out_features), stored_zero, dtype=torch.int32
    )
    return {
        "qweight": _pack_int4(quantized, dim=0),
        "scales": scales.to(torch.float16).contiguous(),
        "qzeros": _pack_int4(logical_zeros, dim=1, pad_value=stored_zero),
        "g_idx": torch.arange(in_features, dtype=torch.int32).div(
            config.group_size, rounding_mode="floor"
        ),
    }


@torch.inference_mode()
def quantize_linear_gptq(
    weight: torch.Tensor,
    hessian: torch.Tensor,
    config: GPTQQuantizerConfig = GPTQQuantizerConfig(),
) -> dict[str, torch.Tensor]:
    working, inverse = _prepare_quantization(weight, hessian, config)
    scales = _group_scales(working, config)
    quantized = _quantize_columns(working, inverse, scales, config)
    return _pack_quantized(quantized, scales, config)
