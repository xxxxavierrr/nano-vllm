"""Rank-local lifecycle for Qwen hybrid recurrent state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


StateBank = tuple[torch.Tensor, torch.Tensor]


class HybridStateManager:
    """Own transient and persistent convolution/recurrent state banks."""

    def __init__(self, model, dtype: torch.dtype):
        self.model = model
        self.dtype = dtype
        self.enabled = hasattr(model, "create_delta_state")
        self.transient: dict[int, StateBank] = {}
        self.committed: StateBank | None = None
        self.working: StateBank | None = None
        self.slots: dict[int, int] = {}
        self.free_slots: list[int] = []
        self.capacity: int | None = None
        self.max_active = 0

    def allocate(
        self,
        capacity: int,
        *,
        device: torch.device | str,
        with_working_copy: bool,
    ) -> None:
        if not self.enabled:
            raise RuntimeError("model does not expose hybrid state")
        if capacity <= 0:
            raise ValueError("hybrid state capacity must be positive")
        self.capacity = capacity
        self.transient.clear()
        self.committed = self.model.create_delta_state_slab(
            capacity,
            device=device,
            dtype=self.dtype,
        )
        self.working = (
            self.model.create_delta_state_slab(
                capacity,
                device=device,
                dtype=self.dtype,
            )
            if with_working_copy
            else None
        )
        self.slots.clear()
        self.free_slots = list(range(capacity - 1, -1, -1))

    def _ensure_slot(self, seq_id: int) -> int:
        slot = self.slots.get(seq_id)
        if slot is not None:
            return slot
        if self.committed is None or self.capacity is None:
            raise RuntimeError("persistent hybrid state is not allocated")
        if not self.free_slots:
            raise RuntimeError(
                "Qwen3.5/3.6 DeltaNet state capacity exhausted: "
                f"{self.capacity} active sequences"
            )
        slot = self.free_slots.pop()
        for tensor in self.committed:
            tensor[:, slot].zero_()
        if self.working is not None:
            for tensor in self.working:
                tensor[:, slot].zero_()
        self.slots[seq_id] = slot
        self.max_active = max(self.max_active, len(self.slots))
        return slot

    def get(self, seq_id: int, *, working: bool = False) -> StateBank | None:
        if not self.enabled:
            return None
        if self.committed is None:
            state = self.transient.get(seq_id)
            if state is None:
                state = self.model.create_delta_state(
                    device=self._model_device(),
                    dtype=self.dtype,
                )
                self.transient[seq_id] = state
                self.max_active = max(self.max_active, len(self.transient))
            return state

        slot = self._ensure_slot(seq_id)
        bank = self.working if working else self.committed
        if bank is None:
            raise RuntimeError("speculative working state is not allocated")
        return bank[0][:, slot], bank[1][:, slot]

    def _model_device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def batch_view(
        self,
        seq_ids: Iterable[int],
        *,
        working: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        if not self.enabled:
            return None
        ids = tuple(seq_ids)
        states = tuple(self.get(seq_id, working=working) for seq_id in ids)
        if self.committed is None:
            conv = torch.stack([state[0] for state in states], dim=1)
            recurrent = torch.stack([state[1] for state in states], dim=1)
            slots = torch.arange(len(ids), dtype=torch.int32, device=conv.device)
            return conv, recurrent, slots

        bank = self.working if working else self.committed
        if bank is None:
            raise RuntimeError("speculative working state is not allocated")
        slots = torch.tensor(
            [self.slots[seq_id] for seq_id in ids],
            dtype=torch.int32,
            device=bank[0].device,
        )
        return bank[0], bank[1], slots

    def _copy_slots(
        self,
        seq_ids: Iterable[int],
        source: StateBank,
        destination: StateBank,
    ) -> None:
        ids = tuple(seq_ids)
        if not ids:
            return
        slots = torch.tensor(
            [self._ensure_slot(seq_id) for seq_id in ids],
            dtype=torch.long,
            device=source[0].device,
        )
        for source_tensor, destination_tensor in zip(source, destination):
            destination_tensor.index_copy_(
                1,
                slots,
                source_tensor.index_select(1, slots),
            )

    def prepare_working(self, seq_ids: Iterable[int]) -> None:
        if self.committed is None or self.working is None:
            return
        self._copy_slots(seq_ids, self.committed, self.working)

    def commit_working(self, seq_ids: Iterable[int]) -> None:
        if self.committed is None or self.working is None:
            return
        self._copy_slots(seq_ids, self.working, self.committed)

    def dummy_slots(self, count: int) -> list[int]:
        if count < 0:
            raise ValueError("dummy slot count cannot be negative")
        if len(self.free_slots) < count:
            raise RuntimeError("Full CUDA Graph padding requires free DeltaNet state slots")
        return self.free_slots[:count]

    def release(self, seq_ids: Iterable[int]) -> None:
        for seq_id in seq_ids:
            if self.committed is None:
                self.transient.pop(seq_id, None)
                continue
            slot = self.slots.pop(seq_id, None)
            if slot is not None:
                self.free_slots.append(slot)

    def close(self) -> None:
        self.transient.clear()
        self.slots.clear()
        self.free_slots.clear()
        self.committed = None
        self.working = None

    def transaction(
        self,
        seq_ids: Iterable[int],
        *,
        enabled: bool,
    ) -> "StateTransaction":
        return StateTransaction(self, tuple(seq_ids), enabled)


@dataclass(slots=True)
class StateTransaction:
    manager: HybridStateManager
    seq_ids: tuple[int, ...]
    enabled: bool

    def begin(self) -> None:
        if self.enabled:
            self.manager.prepare_working(self.seq_ids)

    def commit(self, seq_ids: Iterable[int]) -> None:
        if self.enabled:
            self.manager.commit_working(seq_ids)
