import pytest

from nanovllm.serve.tokenizer import normalize_token_ids


def test_normalizes_legacy_token_list():
    assert normalize_token_ids([1, 2, 3]) == [1, 2, 3]


def test_normalizes_transformers_batch_encoding():
    assert normalize_token_ids({"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}) == [1, 2, 3]


def test_rejects_non_integer_tokens():
    with pytest.raises(ValueError, match="non-integer"):
        normalize_token_ids(["input_ids", "attention_mask"])
