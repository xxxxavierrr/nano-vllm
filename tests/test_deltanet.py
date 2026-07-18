import pytest
import torch

from nanovllm.layers.deltanet import gated_delta_recurrent


def _reference(q, k, value, beta, decay, state):
    outputs = []
    for token in range(q.shape[0]):
        state.mul_(decay[token, :, None, None])
        memory = (state * k[token, :, :, None]).sum(dim=-2)
        delta = (value[token] - memory) * beta[token, :, None]
        state.add_(k[token, :, :, None] * delta[:, None, :])
        outputs.append((state * q[token, :, :, None]).sum(dim=-2))
    return torch.stack(outputs)


def _make_inputs(tokens):
    torch.manual_seed(17 + tokens)
    heads, key_dim, value_dim = 4, 16, 16
    q = torch.randn(tokens, heads, key_dim, device="cuda") * 0.1
    k = torch.randn_like(q) * 0.1
    value = torch.randn(tokens, heads, value_dim, device="cuda")
    beta = torch.sigmoid(torch.randn(tokens, heads, device="cuda"))
    decay = torch.sigmoid(torch.randn(tokens, heads, device="cuda"))
    state = torch.randn(heads, key_dim, value_dim, device="cuda") * 0.1
    return q, k, value, beta, decay, state


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("tokens", [1, 7, 19])
def test_gated_delta_recurrent_matches_reference(tokens):
    q, k, value, beta, decay, initial_state = _make_inputs(tokens)
    expected_state = initial_state.clone()
    expected = _reference(q, k, value, beta, decay, expected_state)
    actual_state = initial_state.clone()
    actual = gated_delta_recurrent(q, k, value, beta, decay, actual_state)

    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(actual_state, expected_state, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_gated_delta_recurrent_continuation_matches_full_sequence():
    q, k, value, beta, decay, initial_state = _make_inputs(19)
    full_state = initial_state.clone()
    full = gated_delta_recurrent(q, k, value, beta, decay, full_state)

    split_state = initial_state.clone()
    first = gated_delta_recurrent(
        q[:7], k[:7], value[:7], beta[:7], decay[:7], split_state
    )
    second = gated_delta_recurrent(
        q[7:], k[7:], value[7:], beta[7:], decay[7:], split_state
    )

    torch.testing.assert_close(torch.cat((first, second)), full, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(split_state, full_state, rtol=1e-4, atol=1e-4)

