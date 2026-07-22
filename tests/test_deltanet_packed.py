import pytest
import torch

from nanovllm.layers.deltanet_chunk import gated_delta_packed


def _reference(q, k, value, beta, decay, state):
    outputs = []
    for token in range(q.shape[0]):
        state.mul_(decay[token, :, None, None])
        memory = (state * k[token, :, :, None]).sum(dim=-2)
        delta = (value[token] - memory) * beta[token, :, None]
        state.add_(k[token, :, :, None] * delta[:, None, :])
        outputs.append((state * q[token, :, :, None]).sum(dim=-2))
    return torch.stack(outputs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_gated_delta_packed_all_recurrent_matches_reference():
    torch.manual_seed(23)
    lengths = (1, 7, 19)
    slots = torch.tensor((2, 0, 3), device="cuda", dtype=torch.int32)
    cu_seqlens = torch.tensor((0, 1, 8, 27), device="cuda", dtype=torch.int32)
    tokens, heads, key_dim, value_dim = sum(lengths), 4, 16, 16
    q = torch.randn(tokens, heads, key_dim, device="cuda") * 0.1
    k = torch.randn_like(q) * 0.1
    value = torch.randn(tokens, heads, value_dim, device="cuda")
    beta = torch.sigmoid(torch.randn(tokens, heads, device="cuda"))
    decay = torch.sigmoid(torch.randn(tokens, heads, device="cuda"))
    initial_state = torch.randn(4, heads, key_dim, value_dim, device="cuda") * 0.1

    expected_state = initial_state.clone()
    expected_parts = []
    for sequence, slot in enumerate(slots.tolist()):
        begin = int(cu_seqlens[sequence])
        end = int(cu_seqlens[sequence + 1])
        expected_parts.append(
            _reference(
                q[begin:end],
                k[begin:end],
                value[begin:end],
                beta[begin:end],
                decay[begin:end],
                expected_state[slot],
            )
        )

    actual_state = initial_state.clone()
    empty = torch.empty(0, device="cuda", dtype=torch.int32)
    actual = gated_delta_packed(
        q,
        k,
        value,
        beta,
        decay,
        cu_seqlens,
        empty.reshape(0, 2),
        torch.zeros(1, device="cuda", dtype=torch.int32),
        empty,
        torch.arange(len(lengths), device="cuda", dtype=torch.int32),
        slots,
        actual_state,
    )
    torch.testing.assert_close(actual, torch.cat(expected_parts), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(actual_state, expected_state, rtol=1e-4, atol=1e-4)
