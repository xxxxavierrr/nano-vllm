from pathlib import Path

import pytest
import torch

from nanovllm.layers.gptq_native import (
    W4Kernel,
    quantize_w4a8_activation_reference,
    select_w4_kernel,
    validate_native_w4_layout,
)


def test_auto_and_triton_remain_safe_fallbacks_before_gpu_validation():
    assert select_w4_kernel(1, "auto", extension_available=True, capability=(8, 9)) is W4Kernel.TRITON
    assert select_w4_kernel(512, "triton", extension_available=True, capability=(8, 9)) is W4Kernel.TRITON


def test_explicit_native_dispatch_uses_shape_threshold():
    assert select_w4_kernel(1, "marlin", extension_available=True, capability=(8, 9)) is W4Kernel.NATIVE_SMALL
    assert select_w4_kernel(64, "marlin", extension_available=True, capability=(8, 9)) is W4Kernel.NATIVE_SMALL
    assert select_w4_kernel(65, "marlin", extension_available=True, capability=(8, 9)) is W4Kernel.NATIVE_LARGE


def test_explicit_native_dispatch_rejects_missing_extension_and_wrong_sm():
    with pytest.raises(RuntimeError, match="NANOVLLM_BUILD_CUDA_EXT"):
        select_w4_kernel(1, "marlin", extension_available=False, capability=(8, 9))
    with pytest.raises(RuntimeError, match="SM89"):
        select_w4_kernel(1, "marlin", extension_available=True, capability=(9, 0))


def test_native_layout_requires_repacked_symmetric_group128():
    validate_native_w4_layout(
        symmetric_zero=True,
        direct_groups=True,
        input_perm_numel=256,
        k=256,
        group_size=128,
    )
    with pytest.raises(ValueError, match="symmetric"):
        validate_native_w4_layout(
            symmetric_zero=False,
            direct_groups=True,
            input_perm_numel=256,
            k=256,
            group_size=128,
        )
    with pytest.raises(ValueError, match="repacking"):
        validate_native_w4_layout(
            symmetric_zero=True,
            direct_groups=False,
            input_perm_numel=256,
            k=256,
            group_size=128,
        )


def test_w4a8_reference_uses_per_row_per_group_scales():
    x = torch.linspace(-3, 3, 4 * 256, dtype=torch.float32).reshape(4, 256)
    quantized, scales = quantize_w4a8_activation_reference(x)
    restored = (
        quantized.reshape(4, 2, 128).float() * scales.unsqueeze(-1)
    ).reshape_as(x)

    assert quantized.dtype is torch.int8
    assert scales.shape == (4, 2)
    assert (x - restored).abs().max() <= scales.max() / 2 + 1e-6


def test_native_extension_build_is_explicit_opt_in():
    root = Path(__file__).parents[1]
    setup_source = (root / "setup.py").read_text(encoding="utf-8")
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    assert 'os.getenv("NANOVLLM_BUILD_CUDA_EXT") == "1"' in setup_source
    assert "torch.utils.cpp_extension" not in pyproject
