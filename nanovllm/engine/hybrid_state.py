"""Rank-local lifecycle for Qwen hybrid recurrent state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import torch


StateBank = tuple[torch.Tensor, torch.Tensor]


class HybridStateManager:
    """Own persistent GDN state slots and speculative prefix branches.

    A request normally owns one committed slot.  A speculative target pass
    reserves one candidate slot for every scheduled input and the GDN kernels
    write the state after each prefix directly to those candidates.  Commit is
    therefore an index remap, not a state copy or a target-model replay.
    """

    def __init__(self, model, dtype: torch.dtype):
        self.model = model
        self.dtype = dtype
        self.enabled = hasattr(model, "create_delta_state")
        self.transient: dict[int, StateBank] = {}
        self.committed: StateBank | None = None
        self.slots: dict[int, int] = {}
        self.branches: dict[int, tuple[int, ...]] = {}
        self.free_slots: list[int] = []
        self.capacity: int | None = None
        self.total_slot_capacity: int | None = None
        self.branch_slots_per_sequence = 0
        self.max_active = 0
        self.max_reserved_branches = 0
        self.branch_commits = 0
        self.branch_discards = 0
        self.rejected_prefix_target_replays = 0

    def allocate(
        self,
        capacity: int,
        *,
        device: torch.device | str,
        branch_slots_per_sequence: int = 0,
        with_working_copy: bool | None = None,
    ) -> None:
        if not self.enabled:
            raise RuntimeError("model does not expose hybrid state")
        if capacity <= 0:
            raise ValueError("hybrid state capacity must be positive")
        if with_working_copy is not None:
            # Compatibility for older callers/tests.  One working copy used to
            # mean one speculative branch per request.
            branch_slots_per_sequence = max(
                branch_slots_per_sequence,
                int(with_working_copy),
            )
        if branch_slots_per_sequence < 0:
            raise ValueError("branch slot count cannot be negative")
        self.capacity = capacity
        self.branch_slots_per_sequence = branch_slots_per_sequence
        self.total_slot_capacity = capacity * (1 + branch_slots_per_sequence)
        self.transient.clear()
        self.committed = self.model.create_delta_state_slab(
            self.total_slot_capacity,
            device=device,
            dtype=self.dtype,
        )
        self.slots.clear()
        self.branches.clear()
        self.free_slots = list(range(self.total_slot_capacity - 1, -1, -1))
        self.branch_commits = 0
        self.branch_discards = 0
        self.rejected_prefix_target_replays = 0

    @property
    def working(self) -> None:
        """The copy-based working slab was removed by branch-state commit."""
        return None

    def _take_slot(self, *, zero: bool) -> int:
        if self.committed is None or self.total_slot_capacity is None:
            raise RuntimeError("persistent hybrid state is not allocated")
        if not self.free_slots:
            raise RuntimeError(
                "Qwen3.5/3.6 DeltaNet state slot pool exhausted: "
                f"{self.total_slot_capacity} total slots"
            )
        slot = self.free_slots.pop()
        if zero:
            for tensor in self.committed:
                tensor[:, slot].zero_()
        return slot

    def _ensure_slot(self, seq_id: int) -> int:
        slot = self.slots.get(seq_id)
        if slot is not None:
            return slot
        if self.capacity is not None and len(self.slots) >= self.capacity:
            raise RuntimeError(
                "Qwen3.5/3.6 DeltaNet request capacity exhausted: "
                f"{self.capacity} active sequences"
            )
        slot = self._take_slot(zero=True)
        self.slots[seq_id] = slot
        self.max_active = max(self.max_active, len(self.slots))
        return slot

    def get(self, seq_id: int, *, working: bool = False) -> StateBank | None:
        if working:
            raise RuntimeError(
                "copy-based speculative working state was replaced by "
                "indexed prefix branches"
            )
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
        return self.committed[0][:, slot], self.committed[1][:, slot]

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
        if working:
            raise RuntimeError("working state views no longer exist")
        if not self.enabled:
            return None
        ids = tuple(seq_ids)
        states = tuple(self.get(seq_id) for seq_id in ids)
        if self.committed is None:
            conv = torch.stack([state[0] for state in states], dim=1)
            recurrent = torch.stack([state[1] for state in states], dim=1)
            slots = torch.arange(len(ids), dtype=torch.int32, device=conv.device)
            return conv, recurrent, slots

        slots = torch.tensor(
            [self.slots[seq_id] for seq_id in ids],
            dtype=torch.int32,
            device=self.committed[0].device,
        )
        return self.committed[0], self.committed[1], slots

    def reserve_branches(self, lengths: Mapping[int, int]) -> None:
        if not lengths or self.committed is None:
            return
        if self.branches:
            raise RuntimeError("speculative state branches are already active")
        required = sum(lengths.values())
        if any(length <= 0 for length in lengths.values()):
            raise ValueError("speculative branch lengths must be positive")
        if any(
            length > self.branch_slots_per_sequence
            for length in lengths.values()
        ):
            raise RuntimeError(
                "speculative state branch length exceeds planned capacity"
            )
        if len(self.free_slots) < required:
            raise RuntimeError(
                "DeltaNet speculative branch capacity exhausted: "
                f"requires {required}, has {len(self.free_slots)} free slots"
            )
        try:
            for seq_id, length in lengths.items():
                self._ensure_slot(seq_id)
                self.branches[seq_id] = tuple(
                    self._take_slot(zero=False) for _ in range(length)
                )
        except Exception:
            self.discard_branches()
            raise
        self.max_reserved_branches = max(
            self.max_reserved_branches,
            sum(map(len, self.branches.values())),
        )

    def branch_slots_view(
        self,
        seq_ids: Iterable[int],
        width: int,
    ) -> torch.Tensor | None:
        if not self.enabled:
            return None
        if width <= 0:
            raise ValueError("branch slot width must be positive")
        ids = tuple(seq_ids)
        device = (
            self.committed[0].device
            if self.committed is not None
            else self._model_device()
        )
        rows = []
        for seq_id in ids:
            branch = self.branches.get(seq_id, ())
            if len(branch) > width:
                raise ValueError("branch metadata width is too small")
            rows.append((*branch, *([-1] * (width - len(branch)))))
        return torch.tensor(rows, dtype=torch.int32, device=device)

    def commit_branches(self, accepted_inputs: Mapping[int, int]) -> None:
        """Commit candidate state after the selected scheduled-input prefix."""
        expected = set(self.branches)
        provided = set(accepted_inputs)
        if expected != provided:
            raise ValueError(
                "accepted input mapping differs from active branches: "
                f"missing={sorted(expected - provided)}, "
                f"unknown={sorted(provided - expected)}"
            )
        for seq_id, branch in self.branches.items():
            selected_inputs = accepted_inputs[seq_id]
            if not 1 <= selected_inputs <= len(branch):
                raise ValueError(
                    f"accepted input count {selected_inputs} is outside "
                    f"[1, {len(branch)}] for sequence {seq_id}"
                )
        for seq_id, branch in tuple(self.branches.items()):
            selected_inputs = accepted_inputs[seq_id]
            selected = branch[selected_inputs - 1]
            old = self.slots[seq_id]
            self.slots[seq_id] = selected
            self.free_slots.append(old)
            self.free_slots.extend(
                slot for slot in branch if slot != selected
            )
            self.branch_discards += len(branch) - 1
            del self.branches[seq_id]
            self.branch_commits += 1

    def discard_branches(self, seq_ids: Iterable[int] | None = None) -> None:
        ids = tuple(self.branches) if seq_ids is None else tuple(seq_ids)
        for seq_id in ids:
            branch = self.branches.pop(seq_id, ())
            self.free_slots.extend(branch)
            self.branch_discards += len(branch)

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
            self.discard_branches((seq_id,))
            slot = self.slots.pop(seq_id, None)
            if slot is not None:
                self.free_slots.append(slot)

    def close(self) -> None:
        self.transient.clear()
        self.slots.clear()
        self.branches.clear()
        self.free_slots.clear()
        self.committed = None

    def transaction(
        self,
        branch_lengths: Mapping[int, int],
        *,
        enabled: bool,
    ) -> "StateTransaction":
        return StateTransaction(self, dict(branch_lengths), enabled)


@dataclass(slots=True)
class StateTransaction:
    manager: HybridStateManager
    branch_lengths: dict[int, int]
    enabled: bool

    def begin(self) -> None:
        if self.enabled:
            self.manager.reserve_branches(self.branch_lengths)

    def commit(self, accepted_inputs: Mapping[int, int]) -> None:
        if self.enabled:
            self.manager.commit_branches(accepted_inputs)

    def discard(self) -> None:
        if self.enabled:
            self.manager.discard_branches(self.branch_lengths)
