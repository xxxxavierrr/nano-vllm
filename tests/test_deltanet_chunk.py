import pytest
import torch

from nanovllm.layers.deltanet import gated_delta_recurrent
from nanovllm.layers.deltanet_chunk import gated_delta_chunk


def _make_inputs(tokens: int):
    torch.manual_seed(41 + tokens)
    heads, key_dim, value_dim = 4, 16, 16
    query = torch.randn(tokens, heads, key_dim, device="cuda") * 0.1
    key = torch.randn_like(query) * 0.1
    value = torch.randn(tokens, heads, value_dim, device="cuda")
    beta = torch.sigmoid(torch.randn(tokens, heads, device="cuda"))
    decay = torch.sigmoid(torch.randn(tokens, heads, device="cuda"))
    state = torch.randn(heads, key_dim, value_dim, device="cuda") * 0.1
    return query, key, value, beta, decay, state


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("tokens", [1, 7, 19, 32, 64, 128])
def test_gated_delta_chunk_matches_recurrent(tokens):
    query, key, value, beta, decay, initial_state = _make_inputs(tokens)
    expected_state = initial_state.clone()
    expected = gated_delta_recurrent(
        query, key, value, beta, decay, expected_state
    )
    actual_state = initial_state.clone()
    actual = gated_delta_chunk(
        query, key, value, beta, decay, actual_state
    )

    torch.testing.assert_close(actual, expected, rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(
        actual_state, expected_state, rtol=5e-3, atol=5e-3
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_gated_delta_chunk_continuation_matches_full_sequence():
    query, key, value, beta, decay, initial_state = _make_inputs(128)
    full_state = initial_state.clone()
    full = gated_delta_chunk(
        query, key, value, beta, decay, full_state
    )

    split_state = initial_state.clone()
    first = gated_delta_chunk(
        query[:64],
        key[:64],
        value[:64],
        beta[:64],
        decay[:64],
        split_state,
    )
    second = gated_delta_chunk(
        query[64:],
        key[64:],
        value[64:],
        beta[64:],
        decay[64:],
        split_state,
    )

    torch.testing.assert_close(
        torch.cat((first, second)), full, rtol=5e-3, atol=5e-3
    )
    torch.testing.assert_close(
        split_state, full_state, rtol=5e-3, atol=5e-3
    )
