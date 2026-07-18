import pytest
import torch
import torch.nn.functional as F

from nanovllm.layers.gptq import (
    GPTQ_VALUES_PER_INT32,
    dequantize_gptq_weight,
    gptq_linear_reference,
    unpack_gptq_qweight,
    unpack_gptq_qzeros,
)


def _pack_int4(values: torch.Tensor, dim: int) -> torch.Tensor:
    if values.shape[dim] % GPTQ_VALUES_PER_INT32:
        raise ValueError("packed dimension must be divisible by 8")
    shape = list(values.shape)
    shape[dim] //= GPTQ_VALUES_PER_INT32
    shape.insert(dim + 1, GPTQ_VALUES_PER_INT32)
    chunks = values.to(torch.int64).reshape(shape)
    shifts_shape = [1] * chunks.ndim
    shifts_shape[dim + 1] = GPTQ_VALUES_PER_INT32
    shifts = (
        torch.arange(GPTQ_VALUES_PER_INT32, dtype=torch.int64)
        .mul(4)
        .view(shifts_shape)
    )
    return torch.sum(chunks << shifts, dim=dim + 1).to(torch.int32)


def _make_gptq_tensors(dtype: torch.dtype = torch.float32):
    torch.manual_seed(0)
    in_features, out_features, num_groups = 32, 24, 4
    values = torch.randint(0, 16, (in_features, out_features), dtype=torch.int32)
    zeros = torch.randint(1, 17, (num_groups, out_features), dtype=torch.int32)
    scales = torch.rand(num_groups, out_features, dtype=dtype).mul_(0.2).add_(0.01)
    # Exercise the desc_act representation: input channels need not map to
    # monotonically increasing groups.
    g_idx = torch.tensor([i % num_groups for i in range(in_features)], dtype=torch.int32)
    qweight = _pack_int4(values, dim=0)
    qzeros = _pack_int4(zeros - 1, dim=1)
    return values, zeros, scales, g_idx, qweight, qzeros


def test_unpack_gptq_int4_tensors():
    values, zeros, _, _, qweight, qzeros = _make_gptq_tensors()
    assert torch.equal(unpack_gptq_qweight(qweight), values)
    assert torch.equal(unpack_gptq_qzeros(qzeros, values.shape[1]), zeros)


def test_dequantize_gptq_weight_matches_formula():
    values, zeros, scales, g_idx, qweight, qzeros = _make_gptq_tensors()
    groups = g_idx.long()
    expected = (
        (values - zeros.index_select(0, groups)).to(scales.dtype)
        * scales.index_select(0, groups)
    ).t().contiguous()
    actual = dequantize_gptq_weight(qweight, scales, qzeros, g_idx)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_gptq_linear_reference_matches_bf16_linear(dtype):
    values, zeros, scales, g_idx, qweight, qzeros = _make_gptq_tensors(dtype)
    weight = (
        (values - zeros.index_select(0, g_idx.long())).to(dtype)
        * scales.index_select(0, g_idx.long())
    ).t().contiguous()
    x = torch.randn(7, values.shape[0], dtype=dtype)
    try:
        expected = F.linear(x, weight)
    except RuntimeError as exc:
        if dtype is torch.bfloat16:
            pytest.skip(f"CPU BF16 linear is unavailable: {exc}")
        raise
    actual = gptq_linear_reference(x, qweight, scales, qzeros, g_idx)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_qzeros_padding_is_trimmed_to_output_width():
    torch.manual_seed(1)
    in_features, out_features, packed_out_features, num_groups = 32, 23, 24, 4
    values = torch.randint(0, 16, (in_features, out_features), dtype=torch.int32)
    padded_zeros = torch.randint(
        1, 17, (num_groups, packed_out_features), dtype=torch.int32
    )
    scales = torch.rand(num_groups, out_features)
    g_idx = torch.arange(in_features, dtype=torch.int32) // (in_features // num_groups)
    qweight = _pack_int4(values, dim=0)
    qzeros = _pack_int4(padded_zeros - 1, dim=1)

    expected = (
        (values - padded_zeros[:, :out_features].index_select(0, g_idx.long())).float()
        * scales.index_select(0, g_idx.long())
    ).t().contiguous()
    actual = dequantize_gptq_weight(qweight, scales, qzeros, g_idx)
    torch.testing.assert_close(actual, expected)


def test_gptq_reference_rejects_non_floating_scales():
    _, _, scales, g_idx, qweight, qzeros = _make_gptq_tensors()
    with pytest.raises(TypeError, match="scales must be floating point"):
        dequantize_gptq_weight(qweight, scales.to(torch.int32), qzeros, g_idx)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_gptq_bf16_reference_matches_cuda_linear():
    values, zeros, scales, g_idx, qweight, qzeros = _make_gptq_tensors(torch.bfloat16)
    values = values.cuda()
    zeros = zeros.cuda()
    scales = scales.cuda()
    g_idx = g_idx.cuda()
    qweight = qweight.cuda()
    qzeros = qzeros.cuda()
    weight = (
        (values - zeros.index_select(0, g_idx.long())).to(torch.bfloat16)
        * scales.index_select(0, g_idx.long())
    ).t().contiguous()
    x = torch.randn(7, values.shape[0], dtype=torch.bfloat16, device="cuda")

    expected = F.linear(x, weight)
    actual = gptq_linear_reference(x, qweight, scales, qzeros, g_idx)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
