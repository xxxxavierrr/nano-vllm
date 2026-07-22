from __future__ import annotations

import torch


def greedy_accept(
    target_token_ids: list[int],
    draft_token_ids: list[int],
) -> tuple[list[int], int]:
    """Verify a greedy draft chain and return outputs plus accept count."""
    if len(target_token_ids) != len(draft_token_ids) + 1:
        raise ValueError(
            "target verification requires one more token than the draft chain"
        )
    for index, draft_token_id in enumerate(draft_token_ids):
        target_token_id = target_token_ids[index]
        if target_token_id != draft_token_id:
            return draft_token_ids[:index] + [target_token_id], index
    return draft_token_ids + [target_token_ids[-1]], len(draft_token_ids)


def greedy_accept_k1(
    target_token_ids: list[int],
    draft_token_id: int,
) -> tuple[list[int], int]:
    """Backward-compatible k=1 wrapper."""
    return greedy_accept(target_token_ids, [draft_token_id])


class GreedyAcceptance:
    """Exact-prefix acceptance policy for deterministic target sampling."""

    @staticmethod
    def accept(
        target_token_ids: list[int],
        draft_token_ids: list[int],
    ) -> tuple[list[int], int]:
        return greedy_accept(target_token_ids, draft_token_ids)


def sample_from_logits(
    logits: torch.Tensor,
    temperature: float,
    *,
    generator: torch.Generator | None = None,
) -> int:
    """Sample one token using the same transform as rejection sampling."""
    if logits.ndim != 1:
        raise ValueError("sampling expects one rank-1 logits tensor")
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    if temperature == 0:
        return int(logits.argmax().item())
    probabilities = torch.softmax(logits.float() / temperature, dim=-1)
    return int(
        torch.multinomial(probabilities, 1, generator=generator).item()
    )


class RejectionSamplingAcceptance:
    """Lossless speculative acceptance for temperature sampling.

    Draft token ``x`` is accepted with ``min(1, p(x) / q(x))``.  The first
    rejection samples from normalized ``(p - q)+``; full acceptance samples a
    bonus token from the final target distribution.

    This PyTorch implementation is the correctness path.  The GPU follow-up
    may replace its full-vocabulary temporaries with the vLLM-style blockwise
    Triton sampler without changing this interface or distribution.
    """

    @staticmethod
    @torch.inference_mode()
    def accept(
        target_logits: torch.Tensor,
        draft_logits: torch.Tensor,
        draft_token_ids: list[int],
        temperature: float,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[list[int], int]:
        num_drafts = len(draft_token_ids)
        if target_logits.ndim != 2 or draft_logits.ndim != 2:
            raise ValueError("target and draft logits must be rank-2")
        if target_logits.shape[0] != num_drafts + 1:
            raise ValueError("target logits require one bonus row")
        if draft_logits.shape[0] != num_drafts:
            raise ValueError("draft logits do not match the draft chain")
        if target_logits.shape[1] != draft_logits.shape[1]:
            raise ValueError("target and draft vocabularies differ")
        if temperature <= 0:
            raise ValueError("probabilistic rejection requires temperature > 0")

        target_log_probs = torch.log_softmax(
            target_logits.float() / temperature,
            dim=-1,
        )
        draft_log_probs = torch.log_softmax(
            draft_logits.float() / temperature,
            dim=-1,
        )

        outputs: list[int] = []
        for index, draft_token in enumerate(draft_token_ids):
            if not 0 <= draft_token < target_logits.shape[1]:
                raise ValueError("draft token is outside the vocabulary")
            log_ratio = (
                target_log_probs[index, draft_token]
                - draft_log_probs[index, draft_token]
            ).clamp_max(0)
            uniform = torch.rand(
                (),
                device=target_logits.device,
                generator=generator,
            ).clamp_min_(torch.finfo(torch.float32).tiny)
            if uniform.log() <= log_ratio:
                outputs.append(draft_token)
                continue

            target_probability = target_log_probs[index].exp()
            draft_probability = draft_log_probs[index].exp()
            residual = (target_probability - draft_probability).clamp_min_(0)
            residual_mass = residual.sum()
            # Exact arithmetic guarantees positive mass after rejection.  The
            # fallback only handles finite-precision cancellation.
            distribution = torch.where(
                residual_mass > torch.finfo(torch.float32).eps,
                residual / residual_mass.clamp_min(
                    torch.finfo(torch.float32).eps
                ),
                target_probability,
            )
            recovery = torch.multinomial(
                distribution,
                1,
                generator=generator,
            )
            outputs.append(int(recovery.item()))
            return outputs, index

        bonus = torch.multinomial(
            target_log_probs[-1].exp(),
            1,
            generator=generator,
        )
        outputs.append(int(bonus.item()))
        return outputs, num_drafts
