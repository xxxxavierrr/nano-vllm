from collections.abc import Mapping
from operator import index
from typing import Any


def normalize_token_ids(encoded: Any) -> list[int]:
    if isinstance(encoded, Mapping):
        encoded = encoded["input_ids"]
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], (list, tuple)):
        if len(encoded) != 1:
            raise ValueError("expected token IDs for exactly one prompt")
        encoded = encoded[0]
    try:
        return [index(token_id) for token_id in encoded]
    except TypeError as exc:
        raise ValueError("tokenizer returned non-integer token IDs") from exc
