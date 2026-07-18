import torch

from nanovllm.layers.rotary_embedding import RotaryEmbedding
from nanovllm.models.qwen3_5 import Qwen3_5RMSNorm


def test_qwen35_rmsnorm_is_one_centered():
    norm = Qwen3_5RMSNorm(4, eps=1e-6)
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    expected = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)

    torch.testing.assert_close(norm(x), expected)
    with torch.no_grad():
        norm.weight.fill_(0.5)
    torch.testing.assert_close(norm(x), expected * 1.5)


def test_qwen35_rmsnorm_residual_path_matches_explicit_add():
    norm = Qwen3_5RMSNorm(4, eps=1e-6)
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    residual = torch.tensor([[0.5, -0.5, 1.0, -1.0]])

    output, updated_residual = norm(x, residual)

    torch.testing.assert_close(updated_residual, x + residual)
    torch.testing.assert_close(output, norm(x + residual))


def test_partial_rotary_embedding_preserves_non_rotary_tail():
    rope = RotaryEmbedding(
        head_size=8,
        rotary_dim=4,
        max_position_embeddings=16,
        base=10_000.0,
    )
    positions = torch.tensor([1, 2])
    query = torch.randn(2, 1, 8)
    key = torch.randn(2, 1, 8)

    rotated_query, rotated_key = rope(positions, query, key)

    torch.testing.assert_close(rotated_query[..., 4:], query[..., 4:])
    torch.testing.assert_close(rotated_key[..., 4:], key[..., 4:])
    assert not torch.equal(rotated_query[..., :4], query[..., :4])
