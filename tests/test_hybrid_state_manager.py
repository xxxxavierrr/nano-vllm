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
                2,
                capacity,
                1,
                2,
                3,
                device=device,
                dtype=torch.float32,
            ),
        )


def test_transient_batch_view_uses_production_slab_layout():
    manager = HybridStateManager(FakeHybridModel(), torch.float32)

    conv, recurrent, slots = manager.batch_view((10, 11))

    assert conv.shape == (2, 2, 3, 4)
    assert recurrent.shape == (2, 2, 1, 2, 3)
    assert slots.tolist() == [0, 1]
    assert manager.max_active == 2
    manager.release((10, 11))
    assert manager.transient == {}


def test_persistent_slots_reset_reuse_and_enforce_capacity():
    manager = HybridStateManager(FakeHybridModel(), torch.float32)
    manager.allocate(2, device="cpu", with_working_copy=False)

    first = manager.get(10)
    first[0].fill_(7)
    assert manager.slots[10] == 0
    manager.get(11)
    try:
        manager.get(12)
    except RuntimeError as exc:
        assert "capacity exhausted" in str(exc)
    else:
        raise AssertionError("capacity exhaustion was not reported")

    manager.release((10,))
    reused = manager.get(12)
    assert manager.slots[12] == 0
    assert torch.count_nonzero(reused[0]) == 0


def test_speculative_transaction_copies_and_commits_selected_slots():
    manager = HybridStateManager(FakeHybridModel(), torch.float32)
    manager.allocate(2, device="cpu", with_working_copy=True)
    state_10 = manager.get(10)
    state_11 = manager.get(11)
    state_10[0].fill_(1)
    state_11[0].fill_(2)

    transaction = manager.transaction((10, 11), enabled=True)
    transaction.begin()
    working_10 = manager.get(10, working=True)
    working_11 = manager.get(11, working=True)
    torch.testing.assert_close(working_10[0], state_10[0])
    torch.testing.assert_close(working_11[0], state_11[0])

    working_10[0].fill_(9)
    working_11[0].fill_(8)
    transaction.commit((10,))

    assert torch.all(manager.get(10)[0] == 9)
    assert torch.all(manager.get(11)[0] == 2)


def test_dummy_slots_do_not_allocate_request_ownership():
    manager = HybridStateManager(FakeHybridModel(), torch.float32)
    manager.allocate(3, device="cpu", with_working_copy=False)
    manager.get(10)

    dummy = manager.dummy_slots(2)

    assert dummy == [2, 1]
    assert manager.slots == {10: 0}
