from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from nanovllm.engine.metrics import SpeculativeStepMetrics
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.speculative import GreedyAcceptance, RejectionSamplingAcceptance


@dataclass(slots=True)
class SpeculativeBatchOutput:
    token_ids: list[list[int]]
    accepted_counts: list[int]
    next_draft_token_ids: list[list[int] | None]


@dataclass(slots=True)
class VerificationResult:
    token_groups: list[list[int]]
    accepted_counts: list[int]


class SpeculativeStepCoordinator:
    """Own verification, prefix commit, proposal state, metrics, and RNG."""

    def __init__(self, *, num_drafts: int, device: str):
        self.num_drafts = num_drafts
        self.device = device
        self.greedy = GreedyAcceptance()
        self.rejection = RejectionSamplingAcceptance()
        self.draft_logits: dict[int, torch.Tensor] = {}
        self.generators: dict[int, torch.Generator] = {}

    def release(self, seq_ids) -> None:
        for seq_id in seq_ids:
            self.draft_logits.pop(seq_id, None)
            self.generators.pop(seq_id, None)

    def generator_for(self, seq: Sequence) -> torch.Generator:
        generator = self.generators.get(seq.seq_id)
        if generator is None:
            generator = torch.Generator(device=self.device)
            seed = seq.seed if seq.seed is not None else (
                torch.initial_seed() + seq.seq_id
            ) % (2**63 - 1)
            generator.manual_seed(seed)
            self.generators[seq.seq_id] = generator
        return generator

    def _verify_draft(
        self,
        seq: Sequence,
        target_logits: torch.Tensor,
    ) -> tuple[list[int], int]:
        if len(seq.draft_token_ids) != self.num_drafts:
            raise ValueError(
                "MTP draft chain length does not match num_speculative_tokens"
            )
        if seq.temperature == 0:
            self.draft_logits.pop(seq.seq_id, None)
            return self.greedy.accept(
                target_logits.argmax(dim=-1).tolist(), seq.draft_token_ids
            )
        draft_logits = self.draft_logits.pop(seq.seq_id, None)
        if draft_logits is None:
            raise RuntimeError("probabilistic MTP verification is missing draft logits")
        return self.rejection.accept(
            target_logits,
            draft_logits,
            seq.draft_token_ids,
            seq.temperature,
            generator=self.generator_for(seq),
        )

    def verify(
        self,
        sampled_seqs: list[Sequence],
        logits: torch.Tensor,
    ) -> VerificationResult:
        token_groups: list[list[int]] = []
        accepted_counts: list[int] = []
        offset = 0
        for seq in sampled_seqs:
            if seq.is_speculative:
                width = len(seq.draft_token_ids) + 1
                outputs, accepted = self._verify_draft(seq, logits[offset : offset + width])
                offset += width
            else:
                outputs = [int(logits[offset].argmax().item())]
                accepted = 0
                offset += 1
            token_groups.append(outputs)
            accepted_counts.append(accepted)
        if offset != logits.size(0):
            raise ValueError("target verification logits were not fully consumed")
        return VerificationResult(token_groups, accepted_counts)

    @staticmethod
    def commit(transaction, sampled_seqs: list[Sequence], verified: VerificationResult) -> None:
        prefixes = {
            seq.seq_id: 1 + accepted
            for seq, accepted in zip(sampled_seqs, verified.accepted_counts)
            if seq.is_speculative
        }
        if prefixes:
            transaction.commit(prefixes)

    def _remember_proposals(self, sampled_seqs, proposals) -> list[list[int] | None]:
        next_ids: list[list[int] | None] = []
        for seq, proposal in zip(sampled_seqs, proposals):
            if proposal is None:
                next_ids.append(None)
                self.draft_logits.pop(seq.seq_id, None)
                continue
            next_ids.append(proposal.token_ids)
            if seq.temperature > 0:
                self.draft_logits[seq.seq_id] = proposal.logits
            else:
                self.draft_logits.pop(seq.seq_id, None)
        return next_ids

    @staticmethod
    def _metrics(sampled_seqs, proposals, accepted_counts) -> SpeculativeStepMetrics:
        proposed = sum(len(seq.draft_token_ids) for seq in sampled_seqs)
        accepted = sum(accepted_counts)
        position = lambda index: sum(
            int(seq.is_speculative and count >= index)
            for seq, count in zip(sampled_seqs, accepted_counts)
        )
        return SpeculativeStepMetrics(
            drafted=sum(len(item.token_ids) for item in proposals if item is not None),
            proposed=proposed,
            accepted=accepted,
            rejected=proposed - accepted,
            bonus=sum(
                int(seq.is_speculative and count == len(seq.draft_token_ids))
                for seq, count in zip(sampled_seqs, accepted_counts)
            ),
            verification_rounds=sum(int(seq.is_speculative) for seq in sampled_seqs),
            accepted_position_1=position(1),
            accepted_position_2=position(2),
            accepted_position_3=position(3),
        )

    def propose(
        self,
        proposer,
        seqs: list[Sequence],
        hidden_states: torch.Tensor,
        sampled_seqs: list[Sequence],
        verified: VerificationResult,
    ) -> tuple[SpeculativeBatchOutput, SpeculativeStepMetrics]:
        proposals = proposer.propose(
            seqs,
            hidden_states,
            verified.token_groups,
            verified.accepted_counts,
            [seq.temperature for seq in sampled_seqs],
            [self.generator_for(seq) for seq in sampled_seqs],
        )
        if len(proposals) != len(sampled_seqs):
            raise ValueError("MTP proposal count does not match sampled requests")
        next_ids = self._remember_proposals(sampled_seqs, proposals)
        output = SpeculativeBatchOutput(
            verified.token_groups,
            verified.accepted_counts,
            next_ids,
        )
        return output, self._metrics(
            sampled_seqs, proposals, verified.accepted_counts
        )
