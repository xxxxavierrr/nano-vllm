import pytest
import torch

from nanovllm.layers.sampler import Sampler
from nanovllm.sampling_params import SamplingParams
from nanovllm.serve.api_server import ChatCompletionRequest


def test_sampling_params_accepts_greedy_temperature():
    assert SamplingParams(temperature=0).temperature == 0


def test_chat_request_accepts_greedy_temperature():
    request = ChatCompletionRequest(
        model="test",
        messages=[{"role": "user", "content": "hello"}],
        temperature=0,
    )
    assert request.temperature == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_sampler_greedy_is_deterministic():
    sampler = Sampler().cuda()
    logits = torch.tensor(
        [[1.0, 4.0, 2.0], [5.0, -1.0, 3.0]],
        device="cuda",
    )
    temperatures = torch.zeros(2, device="cuda")

    torch.manual_seed(1)
    first = sampler(logits, temperatures)
    torch.manual_seed(999)
    second = sampler(logits, temperatures)

    torch.testing.assert_close(first, torch.tensor([1, 0], device="cuda"))
    torch.testing.assert_close(second, first)
