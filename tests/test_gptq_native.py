from pathlib import Path

import pytest
import torch

from nanovllm.layers.gptq_kernel import gptq_w4a16_linear
from nanovllm.layers.gptq_native import (
    W4Kernel,
    native_extension_status,
    quantize_w4a8_activation_reference,
    select_w4_kernel,
    validate_native_w4_layout,
)


def _pack_int4(values: torch.Tensor, dim: int) -> torch.Tensor:
    shape = list(values.shape)
    shape[dim] //= 8
    shape.insert(dim + 1, 8)
    chunks = values.to(torch.int64).reshape(shape)
    shifts_shape = [1] * chunks.ndim
    shifts_shape[dim + 1] = 8
    shifts = (torch.arange(8, device=values.device) * 4).view(shifts_shape)
    return torch.sum(chunks << shifts, dim=dim + 1).to(torch.int32)


def _native_gpu_ready() -> bool:
    return (
        torch.cuda.is_available()
        and native_extension_status().available
        and torch.cuda.get_device_capability() == (8, 9)
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


@pytest.mark.skipif(not _native_gpu_ready(), reason="SM89 native W4 extension required")
@pytest.mark.parametrize("m", [1, 8, 64, 65, 128, 512])
def test_native_w4a16_matches_repacked_triton(m):
    torch.manual_seed(100 + m)
    k, n, group_size = 256, 192, 128
    groups = k // group_size
    qweight = _pack_int4(
        torch.randint(0, 16, (k, n), device="cuda", dtype=torch.int32),
        0,
    )
    scales = (
        torch.rand(groups, n, device="cuda", dtype=torch.float32) * 0.05 + 0.005
    ).to(torch.bfloat16)
    qzeros = _pack_int4(
        torch.full((groups, n), 7, device="cuda", dtype=torch.int32),
        1,
    )
    g_idx = torch.arange(k, device="cuda", dtype=torch.int32) // group_size
    input_perm = torch.arange(k, device="cuda", dtype=torch.int32)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)

    common = dict(
        symmetric_zero=True,
        input_perm=input_perm,
        direct_groups=True,
        group_size=group_size,
    )
    expected = gptq_w4a16_linear(
        x, qweight, scales, qzeros, g_idx, backend="triton", **common
    )
    actual = gptq_w4a16_linear(
        x, qweight, scales, qzeros, g_idx, backend="marlin", **common
    )

    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)


@pytest.mark.skipif(not _native_gpu_ready(), reason="SM89 native W4 extension required")
def test_native_w4a16_is_fullgraph_and_cuda_graph_safe():
    k, n = 256, 192
    qweight = _pack_int4(
        torch.randint(0, 16, (k, n), device="cuda", dtype=torch.int32), 0
    )
    scales = torch.ones(2, n, device="cuda", dtype=torch.bfloat16)
    qzeros = _pack_int4(
        torch.full((2, n), 7, device="cuda", dtype=torch.int32), 1
    )
    g_idx = torch.arange(k, device="cuda", dtype=torch.int32) // 128
    input_perm = torch.arange(k, device="cuda", dtype=torch.int32)
    x = torch.randn(19, k, device="cuda", dtype=torch.bfloat16)

    def invoke(activation: torch.Tensor) -> torch.Tensor:
        return gptq_w4a16_linear(
            activation,
            qweight,
            scales,
            qzeros,
            g_idx,
            symmetric_zero=True,
            input_perm=input_perm,
            direct_groups=True,
            group_size=128,
            backend="marlin",
        )

    eager = invoke(x)
    compiled = torch.compile(invoke, fullgraph=True)(x)
    static = torch.empty_like(eager)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static.copy_(invoke(x))
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(compiled, eager, rtol=0, atol=0)
    torch.testing.assert_close(static, eager, rtol=0, atol=0)
