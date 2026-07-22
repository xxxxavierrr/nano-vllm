import pytest
import torch
from torch import nn

from nanovllm.engine.hybrid_state import HybridStateManager


class FakeHybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    @staticmethod
    def create_delta_state(device, dtype):
        return (
            torch.zeros(2, 3, 4, device=device, dtype=dtype),
            torch.zeros(2, 1, 2, 3, device=device, dtype=torch.float32),
        )

    @staticmethod
    def create_delta_state_slab(capacity, device, dtype):
        return (
            torch.zeros(2, capacity, 3, 4, device=device, dtype=dtype),
            torch.zeros(
                2, capacity, 1, 2, 3, device=device, dtype=torch.float32
            ),
        )


def test_transient_batch_view_uses_production_slab_layout():
    manager = HybridStateManager(FakeHybridModel(), torch.float32)
    conv, recurrent, slots = manager.batch_view((10, 11))
    assert conv.shape == (2, 2, 3, 4)
    assert recurrent.shape == (2, 2, 1, 2, 3)
    assert slots.tolist() == [0, 1]
    manager.release((10, 11))
    assert manager.transient == {}


def test_persistent_slots_reset_reuse_and_enforce_request_capacity():
    manager = HybridStateManager(FakeHybridModel(), torch.float32)
    manager.allocate(2, device="cpu")
    first = manager.get(10)
    first[0].fill_(7)
    assert manager.slots[10] == 0
    manager.get(11)
    with pytest.raises(RuntimeError, match="request capacity exhausted"):
        manager.get(12)
    manager.release((10,))
    reused = manager.get(12)
    assert manager.slots[12] == 0
    assert torch.count_nonzero(reused[0]) == 0


@pytest.mark.parametrize("accepted_inputs", [1, 2, 3])
def test_speculative_transaction_commits_selected_prefix_slot(accepted_inputs):
    manager = HybridStateManager(FakeHybridModel(), torch.float32)
    manager.allocate(1, device="cpu", branch_slots_per_sequence=3)
    base = manager.get(10)
    base[0].fill_(5)
    old_slot = manager.slots[10]

    transaction = manager.transaction({10: 3}, enabled=True)
    transaction.begin()
    candidates = manager.branches[10]
    for prefix, slot in enumerate(candidates, start=1):
        manager.committed[0][:, slot].fill_(prefix)
        manager.committed[1][:, slot].fill_(prefix)

    transaction.commit({10: accepted_inputs})

    assert manager.slots[10] == candidates[accepted_inputs - 1]
    assert torch.all(manager.get(10)[0] == accepted_inputs)
    assert old_slot in manager.free_slots
    assert len(manager.free_slots) == 3
    assert manager.branches == {}


def test_branch_capacity_failure_does_not_leak_slots():
    manager = HybridStateManager(FakeHybridModel(), torch.float32)
    manager.allocate(1, device="cpu", branch_slots_per_sequence=2)
    manager.get(10)
    free_before = tuple(manager.free_slots)
    with pytest.raises(RuntimeError, match="planned capacity"):
        manager.reserve_branches({10: 3})
    assert tuple(manager.free_slots) == free_before
    assert manager.branches == {}


def test_release_discards_uncommitted_branches():
    manager = HybridStateManager(FakeHybridModel(), torch.float32)
    manager.allocate(1, device="cpu", branch_slots_per_sequence=2)
    manager.reserve_branches({10: 2})
    manager.release((10,))
    assert manager.slots == {}
    assert manager.branches == {}
    assert len(manager.free_slots) == manager.total_slot_capacity


def test_dummy_slots_do_not_allocate_request_ownership():
    manager = HybridStateManager(FakeHybridModel(), torch.float32)
    manager.allocate(3, device="cpu")
    manager.get(10)
    dummy = manager.dummy_slots(2)
    assert dummy == [2, 1]
    assert manager.slots == {10: 0}


def test_multi_request_prefix_commit_is_atomic_and_never_replays_target():
    manager = HybridStateManager(FakeHybridModel(), torch.float32)
    manager.allocate(2, device="cpu", branch_slots_per_sequence=3)
    manager.get(10)
    manager.get(11)
    transaction = manager.transaction({10: 3, 11: 3}, enabled=True)
    transaction.begin()
    candidates = dict(manager.branches)
    for seq_id, slots in candidates.items():
        for prefix, slot in enumerate(slots, start=1):
            manager.committed[0][:, slot].fill_(seq_id + prefix)
            manager.committed[1][:, slot].fill_(seq_id + prefix)

    slots_before = dict(manager.slots)
    branches_before = dict(manager.branches)
    with pytest.raises(ValueError, match="outside"):
        transaction.commit({10: 2, 11: 4})
    assert manager.slots == slots_before
    assert manager.branches == branches_before

    transaction.commit({10: 1, 11: 3})
    assert torch.all(manager.get(10)[0] == 11)
    assert torch.all(manager.get(11)[0] == 14)
    assert manager.branch_commits == 2
    assert manager.branch_discards == 4
    assert manager.rejected_prefix_target_replays == 0


def test_prefix_commit_requires_exact_request_mapping():
    manager = HybridStateManager(FakeHybridModel(), torch.float32)
    manager.allocate(2, device="cpu", branch_slots_per_sequence=2)
    manager.get(10)
    manager.get(11)
    manager.reserve_branches({10: 2, 11: 2})
    with pytest.raises(ValueError, match=r"missing=\[11\]"):
        manager.commit_branches({10: 1})
    assert set(manager.branches) == {10, 11}


def test_reset_metrics_preserves_owned_state_and_resets_counters():
    manager = HybridStateManager(FakeHybridModel(), torch.float32)
    manager.allocate(2, device="cpu", branch_slots_per_sequence=2)
    manager.get(10)
    manager.reserve_branches({10: 2})
    manager.max_active = 9
    manager.max_reserved_branches = 7
    manager.branch_commits = 3
    manager.branch_discards = 4

    manager.reset_metrics()

    assert manager.max_active == 1
    assert manager.max_reserved_branches == 2
    assert manager.branch_commits == 0
    assert manager.branch_discards == 0
    assert manager.slots == {10: 0}
