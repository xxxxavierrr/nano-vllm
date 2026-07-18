import pytest
import torch

from nanovllm.layers.gptq import gptq_linear_reference
from nanovllm.layers.gptq_kernel import gptq_w4a16_linear


def _pack_int4(values: torch.Tensor, dim: int) -> torch.Tensor:
    shape = list(values.shape)
    shape[dim] //= 8
    shape.insert(dim + 1, 8)
    chunks = values.to(torch.int64).reshape(shape)
    shifts_shape = [1] * chunks.ndim
    shifts_shape[dim + 1] = 8
    shifts = (torch.arange(8, device=values.device) * 4).view(shifts_shape)
    return torch.sum(chunks << shifts, dim=dim + 1).to(torch.int32)


def _make_inputs(m: int, k: int, n: int, desc_act: bool = True):
    torch.manual_seed(m + k + n)
    groups = k // 128
    values = torch.randint(0, 16, (k, n), device="cuda", dtype=torch.int32)
    zeros = torch.randint(1, 17, (groups, n), device="cuda", dtype=torch.int32)
    scales = (
        torch.rand(groups, n, device="cuda", dtype=torch.float32) * 0.05 + 0.005
    ).to(torch.bfloat16)
    if desc_act:
        g_idx = torch.tensor(
            [(i * 17) % groups for i in range(k)],
            device="cuda",
            dtype=torch.int32,
        )
    else:
        g_idx = torch.arange(k, device="cuda", dtype=torch.int32) // 128
    qweight = _pack_int4(values, 0)
    qzeros = _pack_int4(zeros - 1, 1)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    return x, qweight, scales, qzeros, g_idx


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("m", [1, 8, 19, 64, 128, 512])
@pytest.mark.parametrize("desc_act", [False, True])
def test_gptq_w4a16_matches_reference(m, desc_act):
    x, qweight, scales, qzeros, g_idx = _make_inputs(m, 256, 192, desc_act)
    expected = gptq_linear_reference(x, qweight, scales, qzeros, g_idx)
    actual = gptq_w4a16_linear(x, qweight, scales, qzeros, g_idx)
    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_gptq_w4a16_bias_and_leading_dimensions():
    x, qweight, scales, qzeros, g_idx = _make_inputs(6, 256, 192)
    x = x.reshape(2, 3, 256)
    bias = torch.randn(192, device="cuda", dtype=torch.bfloat16)
    expected = gptq_linear_reference(x, qweight, scales, qzeros, g_idx, bias)
    actual = gptq_w4a16_linear(x, qweight, scales, qzeros, g_idx, bias)
    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_gptq_w4a16_is_torch_compile_traceable():
    x, qweight, scales, qzeros, g_idx = _make_inputs(19, 256, 192)

    def fn(value):
        return gptq_w4a16_linear(value, qweight, scales, qzeros, g_idx)

    expected = fn(x)
    compiled = torch.compile(fn, fullgraph=True)
    actual = compiled(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
