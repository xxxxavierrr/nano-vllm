from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DeltaStateCapacityPlan:
    capacity: int
    state_bytes_per_sequence: int
    kv_blocks_per_sequence: int
    minimum_kv_blocks: int


def plan_delta_state_capacity(
    *,
    available_bytes: int,
    state_bytes: int,
    state_copies: int,
    block_bytes: int,
    block_size: int,
    max_model_len: int,
    max_num_seqs: int,
    speculative_tokens: int,
) -> DeltaStateCapacityPlan:
    positive = {
        "available_bytes": available_bytes,
        "state_bytes": state_bytes,
        "state_copies": state_copies,
        "block_bytes": block_bytes,
        "block_size": block_size,
        "max_model_len": max_model_len,
        "max_num_seqs": max_num_seqs,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise ValueError(f"{', '.join(invalid)} must be positive")
    if speculative_tokens < 0:
        raise ValueError("speculative_tokens cannot be negative")

    state_bytes_per_sequence = state_bytes * state_copies
    maximum_cached_tokens = max_model_len + speculative_tokens
    kv_blocks_per_sequence = (
        maximum_cached_tokens + block_size - 1
    ) // block_size
    bytes_per_sequence = (
        state_bytes_per_sequence + kv_blocks_per_sequence * block_bytes
    )
    capacity = min(max_num_seqs, available_bytes // bytes_per_sequence)
    return DeltaStateCapacityPlan(
        capacity=capacity,
        state_bytes_per_sequence=state_bytes_per_sequence,
        kv_blocks_per_sequence=kv_blocks_per_sequence,
        minimum_kv_blocks=capacity * kv_blocks_per_sequence,
    )
