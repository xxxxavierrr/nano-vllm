import pytest
import torch
import torch.nn.functional as F

from nanovllm.layers.deltanet import packed_causal_conv1d
from nanovllm.layers.deltanet_chunk import gated_delta_packed


def _run_recurrent(q, k, value, beta, decay, state):
    device = q.device
    empty = torch.empty(0, device=device, dtype=torch.int32)
    return gated_delta_packed(
        q,
        k,
        value,
        beta,
        decay,
        torch.tensor((0, q.shape[0]), device=device, dtype=torch.int32),
        empty.reshape(0, 2),
        torch.zeros(1, device=device, dtype=torch.int32),
        empty,
        torch.zeros(1, device=device, dtype=torch.int32),
        torch.zeros(1, device=device, dtype=torch.int32),
        state.unsqueeze(0),
    )


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
def test_gated_delta_packed_recurrent_partition_matches_reference(tokens):
    q, k, value, beta, decay, initial_state = _make_inputs(tokens)
    expected_state = initial_state.clone()
    expected = _reference(q, k, value, beta, decay, expected_state)
    actual_state = initial_state.clone()
    actual = _run_recurrent(q, k, value, beta, decay, actual_state)

    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(actual_state, expected_state, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_gated_delta_packed_recurrent_continuation_matches_full_sequence():
    q, k, value, beta, decay, initial_state = _make_inputs(19)
    full_state = initial_state.clone()
    full = _run_recurrent(q, k, value, beta, decay, full_state)

    split_state = initial_state.clone()
    first = _run_recurrent(
        q[:7], k[:7], value[:7], beta[:7], decay[:7], split_state
    )
    second = _run_recurrent(
        q[7:], k[7:], value[7:], beta[7:], decay[7:], split_state
    )

    torch.testing.assert_close(torch.cat((first, second)), full, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(split_state, full_state, rtol=1e-4, atol=1e-4)


def _causal_conv_reference(inputs, weight, cu_seqlens, slots, state):
    outputs = []
    kernel_size = weight.shape[-1]
    for sequence, slot in enumerate(slots.tolist()):
        begin = int(cu_seqlens[sequence].item())
        end = int(cu_seqlens[sequence + 1].item())
        packed = torch.cat(
            (state[slot], inputs[begin:end].transpose(0, 1)),
            dim=-1,
        )
        convolved = F.conv1d(
            packed.unsqueeze(0),
            weight,
            groups=inputs.shape[1],
        )[0, :, -(end - begin) :]
        outputs.append(F.silu(convolved.transpose(0, 1)))
        state[slot].copy_(packed[:, -kernel_size:])
    return torch.cat(outputs, dim=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_packed_causal_conv_mixed_lengths_matches_reference():
    torch.manual_seed(41)
    lengths = [1, 7, 19]
    channels, kernel_size, capacity = 48, 4, 8
    inputs = torch.randn(sum(lengths), channels, device="cuda")
    weight = torch.randn(
        channels, 1, kernel_size, device="cuda"
    ).contiguous()
    cu_seqlens = torch.tensor(
        [0, 1, 8, 27], dtype=torch.int32, device="cuda"
    )
    slots = torch.tensor([6, 1, 4], dtype=torch.int32, device="cuda")
    initial_state = torch.randn(
        capacity, channels, kernel_size, device="cuda"
    )

    expected_state = initial_state.clone()
    expected = _causal_conv_reference(
        inputs, weight, cu_seqlens, slots, expected_state
    )
    actual_state = initial_state.clone()
    actual = packed_causal_conv1d(
        inputs,
        weight,
        cu_seqlens,
        slots,
        actual_state,
        max(lengths),
    )

    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(actual_state, expected_state, rtol=0, atol=0)

