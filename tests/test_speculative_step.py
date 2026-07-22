import torch

from nanovllm.engine.mtp_proposer import DraftProposal
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.speculative_step import SpeculativeStepCoordinator


class FakeTransaction:
    def __init__(self):
        self.prefixes = None

    def commit(self, prefixes):
        self.prefixes = prefixes


class FakeProposer:
    def propose(self, seqs, hidden_states, groups, counts, temperatures, generators):
        assert groups == [[2, 9]]
        assert counts == [1]
        assert temperatures == [0.0]
        assert len(generators) == 1
        return [DraftProposal([4, 5], torch.empty(0, 8))]


def _speculative_sequence() -> Sequence:
    seq = Sequence([1])
    seq.temperature = 0.0
    seq.draft_token_ids = [2, 3]
    seq.num_scheduled_tokens = 3
    return seq


def test_coordinator_verifies_commits_and_proposes_by_phase():
    seq = _speculative_sequence()
    coordinator = SpeculativeStepCoordinator(num_drafts=2, device="cpu")
    logits = torch.tensor(
        [
            [0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0],
            [5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )

    verified = coordinator.verify([seq], logits)
    transaction = FakeTransaction()
    coordinator.commit(transaction, [seq], verified)
    output, metrics = coordinator.propose(
        FakeProposer(), [seq], torch.empty(3, 4), [seq], verified
    )

    assert transaction.prefixes == {seq.seq_id: 2}
    assert output.token_ids == [[2, 9]]
    assert output.accepted_counts == [1]
    assert output.next_draft_token_ids == [[4, 5]]
    assert metrics.proposed == 2
    assert metrics.accepted == 1
    assert metrics.drafted == 2


def test_coordinator_release_owns_rng_and_draft_state():
    seq = _speculative_sequence()
    coordinator = SpeculativeStepCoordinator(num_drafts=2, device="cpu")
    coordinator.draft_logits[seq.seq_id] = torch.ones(1)
    coordinator.generator_for(seq)

    coordinator.release([seq.seq_id])

    assert seq.seq_id not in coordinator.draft_logits
    assert seq.seq_id not in coordinator.generators
