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
