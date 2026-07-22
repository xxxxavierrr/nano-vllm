import pytest
import torch
import torch.nn.functional as F

from nanovllm.layers.deltanet import packed_causal_conv1d
from nanovllm.layers.deltanet_chunk import gated_delta_packed


def _run_recurrent(q, k, value, beta, decay, state):
    device = q.device
    empty = torch.empty(0, device=device, dtype=torch.int32)
    branch_slots = torch.full((1, 1), -1, device=device, dtype=torch.int32)
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
        branch_slots,
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_recurrent_speculative_prefixes_write_distinct_state_slots():
    q, k, value, beta, decay, initial_state = _make_inputs(3)
    expected_states = []
    reference_state = initial_state.clone()
    for token in range(3):
        _reference(
            q[token : token + 1],
            k[token : token + 1],
            value[token : token + 1],
            beta[token : token + 1],
            decay[token : token + 1],
            reference_state,
        )
        expected_states.append(reference_state.clone())

    state_slab = torch.zeros(
        4, *initial_state.shape, device="cuda", dtype=torch.float32
    )
    state_slab[0].copy_(initial_state)
    empty = torch.empty(0, device="cuda", dtype=torch.int32)
    gated_delta_packed(
        q,
        k,
        value,
        beta,
        decay,
        torch.tensor((0, 3), device="cuda", dtype=torch.int32),
        empty.reshape(0, 2),
        torch.zeros(1, device="cuda", dtype=torch.int32),
        empty,
        torch.zeros(1, device="cuda", dtype=torch.int32),
        torch.zeros(1, device="cuda", dtype=torch.int32),
        torch.tensor([[1, 2, 3]], device="cuda", dtype=torch.int32),
        state_slab,
    )

    torch.testing.assert_close(state_slab[0], initial_state, rtol=0, atol=0)
    for slot, expected in enumerate(expected_states, start=1):
        torch.testing.assert_close(state_slab[slot], expected, rtol=1e-4, atol=1e-4)


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
        torch.full(
            (len(lengths), 1), -1, device="cuda", dtype=torch.int32
        ),
        actual_state,
        max(lengths),
    )

    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(actual_state, expected_state, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_causal_conv_speculative_prefixes_write_distinct_state_slots():
    torch.manual_seed(91)
    tokens, channels, kernel_size = 3, 32, 4
    inputs = torch.randn(tokens, channels, device="cuda")
    weight = torch.randn(
        channels, 1, kernel_size, device="cuda"
    ).contiguous()
    state = torch.randn(
        4, channels, kernel_size, device="cuda"
    )
    base = state[0].clone()

    packed_causal_conv1d(
        inputs,
        weight,
        torch.tensor((0, tokens), device="cuda", dtype=torch.int32),
        torch.zeros(1, device="cuda", dtype=torch.int32),
        torch.tensor([[1, 2, 3]], device="cuda", dtype=torch.int32),
        state,
        tokens,
    )

    torch.testing.assert_close(state[0], base, rtol=0, atol=0)
    for prefix, slot in enumerate((1, 2, 3), start=1):
        expected = torch.cat(
            (base, inputs[:prefix].transpose(0, 1)), dim=-1
        )[:, -kernel_size:]
        torch.testing.assert_close(state[slot], expected, rtol=0, atol=0)

