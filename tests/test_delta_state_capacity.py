import pytest

from nanovllm.engine.capacity import plan_delta_state_capacity


MIB = 2**20


def make_plan(
    *,
    state_copies: int = 2,
    max_model_len: int = 256,
    max_num_seqs: int = 8,
    speculative_tokens: int = 3,
):
    return plan_delta_state_capacity(
        available_bytes=2661 * MIB,
        state_bytes=int(147.75 * MIB),
        state_copies=state_copies,
        block_bytes=17 * MIB,
        block_size=256,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        speculative_tokens=speculative_tokens,
    )


def test_short_context_supports_eight_mtp_slots():
    plan = make_plan()
    assert plan.capacity == 8
    assert plan.state_bytes_per_sequence == int(295.5 * MIB)
    assert plan.kv_blocks_per_sequence == 2
    assert plan.minimum_kv_blocks == 16


def test_long_context_reduces_capacity_instead_of_starving_kv():
    plan = make_plan(max_model_len=4096)
    assert plan.capacity == 4
    assert plan.kv_blocks_per_sequence == 17
    assert plan.minimum_kv_blocks == 68


def test_non_speculative_state_needs_one_copy_and_no_lookahead():
    plan = make_plan(state_copies=1, speculative_tokens=0)
    assert plan.capacity == 8
    assert plan.state_bytes_per_sequence == int(147.75 * MIB)
    assert plan.kv_blocks_per_sequence == 1


@pytest.mark.parametrize(
    "name,value",
    [
        ("available_bytes", 0),
        ("state_bytes", 0),
        ("state_copies", 0),
        ("block_bytes", 0),
        ("block_size", 0),
        ("max_model_len", 0),
        ("max_num_seqs", 0),
    ],
)
def test_positive_inputs_are_validated(name, value):
    kwargs = {
        "available_bytes": 1,
        "state_bytes": 1,
        "state_copies": 1,
        "block_bytes": 1,
        "block_size": 1,
        "max_model_len": 1,
        "max_num_seqs": 1,
        "speculative_tokens": 0,
    }
    kwargs[name] = value
    with pytest.raises(ValueError, match=name):
        plan_delta_state_capacity(**kwargs)


def test_negative_speculative_tokens_are_rejected():
    with pytest.raises(ValueError, match="speculative_tokens"):
        plan_delta_state_capacity(
            available_bytes=1,
            state_bytes=1,
            state_copies=1,
            block_bytes=1,
            block_size=1,
            max_model_len=1,
            max_num_seqs=1,
            speculative_tokens=-1,
        )
