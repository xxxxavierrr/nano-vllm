import pickle

import torch

from nanovllm.engine.sequence import Sequence
from nanovllm.engine.speculative import (
    RejectionSamplingAcceptance,
    sample_from_logits,
)
from nanovllm.sampling_params import SamplingParams


def test_identical_target_and_draft_distributions_accept_every_draft():
    draft_logits = torch.tensor([[1.0, -0.5, 0.25], [0.1, 0.2, 0.3]])
    target_logits = torch.cat(
        (draft_logits, torch.tensor([[0.3, -0.1, 0.2]])), dim=0
    )
    generator = torch.Generator().manual_seed(7)

    outputs, accepted = RejectionSamplingAcceptance.accept(
        target_logits,
        draft_logits,
        [0, 2],
        1.0,
        generator=generator,
    )

    assert accepted == 2
    assert outputs[:2] == [0, 2]
    assert len(outputs) == 3


def test_rejection_samples_from_positive_probability_difference():
    # q assigns almost all mass to token 0 while p assigns it to token 1.
    draft_logits = torch.tensor([[12.0, -12.0]])
    target_logits = torch.tensor([[-12.0, 12.0], [0.0, 0.0]])
    generator = torch.Generator().manual_seed(11)

    outputs, accepted = RejectionSamplingAcceptance.accept(
        target_logits,
        draft_logits,
        [0],
        1.0,
        generator=generator,
    )

    assert accepted == 0
    assert outputs == [1]


def test_one_step_speculation_preserves_target_distribution_statistically():
    target_logits = torch.tensor([[0.2, 1.1, -0.4], [0.0, 0.0, 0.0]])
    draft_logits = torch.tensor([[1.4, -0.2, 0.3]])
    expected = torch.softmax(target_logits[0], dim=-1)
    generator = torch.Generator().manual_seed(1234)
    counts = torch.zeros(3)

    for _ in range(6000):
        draft = sample_from_logits(
            draft_logits[0], 1.0, generator=generator
        )
        outputs, _ = RejectionSamplingAcceptance.accept(
            target_logits,
            draft_logits,
            [draft],
            1.0,
            generator=generator,
        )
        counts[outputs[0]] += 1

    actual = counts / counts.sum()
    torch.testing.assert_close(actual, expected, atol=0.025, rtol=0)


def test_sampling_seed_reproduces_draft_and_acceptance():
    target_logits = torch.tensor([[0.2, 0.7], [0.6, 0.1]])
    draft_logits = torch.tensor([[0.8, 0.1]])

    def run(seed):
        generator = torch.Generator().manual_seed(seed)
        draft = sample_from_logits(
            draft_logits[0], 0.8, generator=generator
        )
        return RejectionSamplingAcceptance.accept(
            target_logits,
            draft_logits,
            [draft],
            0.8,
            generator=generator,
        )

    assert run(99) == run(99)


def test_sequence_transport_preserves_temperature_and_seed():
    sequence = Sequence(
        [1, 2],
        SamplingParams(temperature=0.75, seed=123),
    )

    restored = pickle.loads(pickle.dumps(sequence))

    assert restored.temperature == 0.75
    assert restored.seed == 123
