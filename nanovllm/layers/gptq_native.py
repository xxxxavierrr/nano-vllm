from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch


class W4Backend(str, Enum):
    AUTO = "auto"
    TRITON = "triton"
    MARLIN = "marlin"


class W4Kernel(str, Enum):
    TRITON = "triton"
    NATIVE_SMALL = "native-small-m"
    NATIVE_LARGE = "native-large-m"


@dataclass(frozen=True, slots=True)
class NativeExtensionStatus:
    available: bool
    reason: str | None


try:
    from nanovllm import _C as _native_extension
except ImportError as exc:
    _native_extension = None
    _native_import_error = str(exc)
else:
    _native_import_error = None

def _fake_native_w4(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    input_perm: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    del scales, input_perm, group_size
    return x.new_empty((x.shape[0], qweight.shape[1]))


if _native_extension is not None:
    for _op_name in ("w4a16_small", "w4a16_large", "w4a8_large"):
        torch.library.register_fake(
            f"nanovllm_native::{_op_name}"
        )(_fake_native_w4)


def native_extension_status() -> NativeExtensionStatus:
    return NativeExtensionStatus(
        available=_native_extension is not None,
        reason=_native_import_error,
    )


def select_w4_kernel(
    m: int,
    backend: W4Backend | str,
    *,
    extension_available: bool | None = None,
    capability: tuple[int, int] | None = None,
) -> W4Kernel:
    if m <= 0:
        raise ValueError("W4 dispatcher requires M > 0")
    backend = W4Backend(backend)
    if backend in {W4Backend.AUTO, W4Backend.TRITON}:
        # Native kernels remain opt-in until SM89 correctness/Graph/performance
        # validation has been completed on the target 4090D.
        return W4Kernel.TRITON
    if extension_available is None:
        extension_available = native_extension_status().available
    if not extension_available:
        raise RuntimeError(
            "Marlin W4 backend was requested but nanovllm._C is unavailable; "
            "reinstall with NANOVLLM_BUILD_CUDA_EXT=1"
        )
    if capability is not None and capability != (8, 9):
        raise RuntimeError(
            f"Marlin W4 v1 targets RTX 4090D SM89, got SM{capability[0]}{capability[1]}"
        )
    return W4Kernel.NATIVE_SMALL if m <= 64 else W4Kernel.NATIVE_LARGE


def validate_native_w4_layout(
    *,
    symmetric_zero: bool,
    direct_groups: bool,
    input_perm_numel: int,
    k: int,
    group_size: int,
) -> None:
    if not symmetric_zero:
        raise ValueError("native W4 v1 requires symmetric zero point 8")
    if not direct_groups:
        raise ValueError("native W4 v1 requires load-time runtime repacking")
    if input_perm_numel != k:
        raise ValueError("native W4 v1 requires a fused activation permutation")
    if group_size != 128 or k % group_size:
        raise ValueError("native W4 v1 requires group_size=128 dividing K")


def quantize_w4a8_activation_reference(
    x: torch.Tensor,
    group_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU correctness reference for experimental per-row/per-group W4A8."""
    if not x.is_floating_point() or x.ndim < 2:
        raise TypeError("W4A8 activation must be a floating tensor with rank >= 2")
    if group_size <= 0 or x.shape[-1] % group_size:
        raise ValueError("W4A8 group_size must divide K")
    shape = x.shape
    grouped = x.float().reshape(-1, shape[-1] // group_size, group_size)
    scales = grouped.abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-12) / 127
    quantized = torch.round(grouped / scales).clamp(-127, 127).to(torch.int8)
    return quantized.reshape(shape), scales.squeeze(-1)


def native_w4a16_linear(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    input_perm: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    group_size: int,
    kernel: W4Kernel,
) -> torch.Tensor:
    if _native_extension is None:
        raise RuntimeError(native_extension_status().reason or "native extension unavailable")
    input_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1]).contiguous()
    if kernel is W4Kernel.NATIVE_SMALL:
        output = torch.ops.nanovllm_native.w4a16_small.default(
            x_2d,
            qweight,
            scales,
            input_perm,
            group_size,
        )
    elif kernel is W4Kernel.NATIVE_LARGE:
        output = torch.ops.nanovllm_native.w4a16_large.default(
            x_2d,
            qweight,
            scales,
            input_perm,
            group_size,
        )
    else:
        raise ValueError(f"invalid native kernel {kernel}")
    if bias is not None:
        output.add_(bias)
    return output.reshape(*input_shape[:-1], qweight.shape[1])


def native_w4a8_linear(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    input_perm: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    group_size: int = 128,
) -> torch.Tensor:
    if x.numel() // x.shape[-1] <= 64:
        raise ValueError("experimental W4A8 is large-M only")
    select_w4_kernel(
        x.numel() // x.shape[-1],
        W4Backend.MARLIN,
        capability=torch.cuda.get_device_capability(x.device),
    )
    input_shape = x.shape
    output = torch.ops.nanovllm_native.w4a8_large.default(
        x.reshape(-1, x.shape[-1]).contiguous(),
        qweight,
        scales,
        input_perm,
        group_size,
    )
    if bias is not None:
        output.add_(bias)
    return output.reshape(*input_shape[:-1], qweight.shape[1])
