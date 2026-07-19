def greedy_accept_k1(
    target_token_ids: list[int],
    draft_token_id: int,
) -> tuple[list[int], int]:
    """Verify one greedy draft and return emitted tokens plus accept count."""
    if len(target_token_ids) != 2:
        raise ValueError("k=1 verification requires two target token ids")
    replacement, bonus = target_token_ids
    if replacement == draft_token_id:
        return [draft_token_id, bonus], 1
    return [replacement], 0
