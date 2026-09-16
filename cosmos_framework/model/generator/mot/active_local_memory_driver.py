# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Active Local-Memory window driver for the functional production route.

The canonical segment ABI and the trainer's active arm surface both exist, but
nothing owned the seam between them: freezing one ``GAWindowPlan`` per optimizer
window and producing each member's ``SegmentBatch`` at the exact batch the
trainer arms it.  This module is that owner.

One window is one optimizer update.  Its member count is exactly
``trainer.config.trainer.grad_accum_iter`` -- the trainer refuses any other
boundary -- so one member is one ``training_step`` invocation and one member's
gathered consumers are one packed native forward.  Slot rotation follows the
scheduler's own deficit rule: this driver decides which admissible candidate to
offer, while ``RankLocalSegmentScheduler.admit`` still validates every admission
and ``admit_next`` re-checks the frozen plan order.

The driver owns no model, optimizer, or trainer state beyond the window it
froze; ``attach`` only binds the registry and registers the batch-start hook.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .canonical_segment_runtime import RuntimePhase
from .local_memory_segment import GAWindowPlan, SegmentIdentity
from .production_active_wiring import ProductionActiveWiringRegistry

if TYPE_CHECKING:
    from .local_memory_segment import SegmentBatch


@dataclass(frozen=True)
class ActiveWindowMember:
    """One frozen window member: its stream, block cursor, and rebind duty."""

    stream: Any
    cursor: int
    training_stream_end: bool
    rebind_before_admit: bool


@dataclass(frozen=True)
class ActiveWindowFreeze:
    """One immutable optimizer-window plan and its per-member production duty."""

    plan: GAWindowPlan
    identities: tuple[SegmentIdentity, ...]
    members: tuple[ActiveWindowMember, ...]

    def __post_init__(self) -> None:
        if (
            len(self.members) != len(self.identities)
            or len(self.members) != self.plan.ga_effective
            or tuple((item.slot_id, item.episode_id, item.cursor) for item in self.identities) != self.plan.members
        ):
            raise ValueError("active window freeze is not aligned with its plan")


class ActiveLocalMemoryWindowDriver:
    """Freeze and arm exactly one canonical Local-Memory window per update.

    ``streams`` is the slot catalog: every entry is one episode's segment stream,
    and entries sharing a ``slot_id`` are consumed in the given order, so a
    finished episode is rebound to its successor without the driver knowing
    anything about the dataset beyond ``producer.block_count``.
    """

    def __init__(
        self,
        *,
        registry: ProductionActiveWiringRegistry,
        producer: Any,
        streams: tuple[Any, ...],
        window_members: int,
        plan_chain_id: str = "active-local-window",
    ) -> None:
        if window_members <= 0:
            raise ValueError("active Local window requires a positive member count")
        if not streams:
            raise ValueError("active Local window requires at least one segment stream")
        if not plan_chain_id:
            raise ValueError("active Local window requires a plan chain id")
        if not isinstance(registry, ProductionActiveWiringRegistry):
            raise TypeError("active Local window requires a production active wiring registry")
        if not callable(getattr(producer, "block_count", None)) or not callable(getattr(producer, "produce", None)):
            raise TypeError("active Local window requires a canonical segment producer")
        by_slot: dict[int, list[Any]] = {}
        for stream in streams:
            by_slot.setdefault(int(stream.slot_id), []).append(stream)
        self.registry = registry
        self.producer = producer
        self.window_members = window_members
        self.plan_chain_id = plan_chain_id
        self._by_slot = {slot: tuple(items) for slot, items in sorted(by_slot.items())}
        self._stream_index = {slot: 0 for slot in self._by_slot}
        self._active_stream: dict[int, Any] = {}
        self._active_cursor: dict[int, int] = {}
        self._window: ActiveWindowFreeze | None = None
        self._window_index = 0
        self._trainer: Any = None

    # ---- attachment --------------------------------------------------------

    def attach(self, trainer: Any, model: Any) -> None:
        """Bind the registry to trainer+model and register the batch-start hook.

        ``CallBackGroup`` exposes no public registration API, so the driver
        appends itself to its callback list; the group's dynamic dispatch then
        invokes :meth:`on_training_step_batch_start` for every batch, which is
        exactly one member of an active window.
        """
        trainer.bind_active_local_memory_registry(model, self.registry)
        existing = getattr(trainer, "_psm_active_window_driver", None)
        if existing is not None and existing is not self:
            raise RuntimeError("trainer already has another active Local window driver")
        trainer._psm_active_window_driver = self
        self._trainer = trainer
        callbacks = getattr(getattr(trainer, "callbacks", None), "_callbacks", None)
        if callbacks is None:
            raise RuntimeError("active Local window driver requires a trainer callback group")
        if self not in callbacks:
            callbacks.append(self)

    def on_training_step_batch_start(self, model: Any, data: Any, iteration: int = 0) -> None:
        """Callback seam: arm one member before the trainer's training step."""
        del data, iteration
        self.arm_next_member(self._trainer, model)

    # ---- arming ------------------------------------------------------------

    def arm_next_member(self, trainer: Any, model: Any) -> None:
        """Arm the exact member the trainer's accumulation counter expects next."""
        if trainer is None:
            raise RuntimeError("active Local window driver is not attached to a trainer")
        if getattr(trainer, "_psm_active_armed_prepared", None) is not None:
            raise RuntimeError("active Local member is already armed before the window driver ran")
        owner = self.registry.owner
        if owner.phase is RuntimePhase.IDLE:
            self._arm_initial(trainer, model)
        elif owner.phase is RuntimePhase.MEMBER_COMMITTED:
            self._arm_continuation(trainer, model)
        else:
            raise RuntimeError(f"active Local window driver found an unarmable owner phase: {owner.phase.name}")

    def _arm_initial(self, trainer: Any, model: Any) -> None:
        configured = int(trainer.config.trainer.grad_accum_iter)
        if self.window_members != configured:
            raise RuntimeError("active Local window member count differs from the accumulation boundary")
        freeze = self.freeze_window()
        self._window = freeze
        segment = self._produce(freeze, 0)
        trainer.arm_active_local_memory_initial(
            model, freeze.identities[0], segment, freeze.plan, grad_accum_iter=0
        )

    def _arm_continuation(self, trainer: Any, model: Any) -> None:
        freeze, transaction = self._window, self.registry.owner.transaction
        if freeze is None or transaction is None:
            raise RuntimeError("active Local window driver lost its frozen window")
        index = len(transaction.completed_members)
        if index >= freeze.plan.ga_effective:
            raise RuntimeError("active Local window is complete but the owner remains open")
        segment = self._produce(freeze, index)
        trainer.arm_active_local_memory_continuation(
            model, freeze.identities[index], segment, transaction, grad_accum_iter=index
        )

    def _produce(self, freeze: ActiveWindowFreeze, index: int) -> "SegmentBatch":
        member = freeze.members[index]
        if member.rebind_before_admit:
            self._rebind_terminal(int(member.stream.slot_id))
        segment = self.producer.produce(member.stream, cursor=member.cursor)
        planned = int(freeze.plan.planned_n_valid[index])
        if int(segment.consumer_valid.sum()) != planned:
            raise RuntimeError("active Local segment valid count differs from the frozen plan")
        return segment

    def _rebind_terminal(self, slot_id: int) -> None:
        """Release a finished episode's terminal slot before its successor is admitted."""
        scheduler = self.registry.owner.scheduler
        terminal = scheduler.terminal_slots.get(slot_id)
        if terminal is not None:
            scheduler.terminal_rebind(terminal)

    # ---- window freezing ---------------------------------------------------

    def freeze_window(self) -> ActiveWindowFreeze:
        """Derive one immutable window plan from the live slot catalog.

        The freeze commits the driver's own slot frontier, because a frozen plan's
        members are already irreversible: ``admit``/``admit_next`` accept exactly
        the identities returned here, and any later failure is fail-closed.
        """
        planned = int(self.producer.ttt_tbptt_steps)
        if planned <= 0:
            raise RuntimeError("active Local producer has an invalid TBPTT width")
        scheduler = self.registry.owner.scheduler
        target = scheduler.target_distribution
        exposure = dict(scheduler.cumulative_valid_consumer_exposure)
        members: list[ActiveWindowMember] = []
        identities: list[SegmentIdentity] = []
        for _ in range(self.window_members):
            choices: list[tuple[float, str, int, tuple]] = []
            for slot_id in self._by_slot:
                view = self._peek_block(slot_id)
                if view is None:
                    continue
                stream, _cursor, _terminal, _rebind, _position = view
                observed = exposure.get(stream.category, 0) / max(sum(exposure.values()), 1)
                choices.append((target[stream.category] - observed, stream.category, slot_id, view))
            if not choices:
                raise RuntimeError("active Local window exhausted every segment stream")
            _, category, slot_id, view = max(choices, key=lambda item: item[:3])
            stream, cursor, terminal, rebind, position = view
            self._commit_block(slot_id, stream, cursor, position)
            exposure[category] = exposure.get(category, 0) + planned
            members.append(ActiveWindowMember(stream, cursor, terminal, rebind))
            identities.append(
                SegmentIdentity(
                    slot_id=slot_id,
                    episode_id=str(stream.episode_index),
                    category=category,
                    cursor=cursor,
                    segment_id=cursor,
                    source_digest=self.producer.source_digest,
                    training_stream_end=terminal,
                )
            )
        self._window_index += 1
        plan = GAWindowPlan(
            members=tuple((item.slot_id, item.episode_id, item.cursor) for item in identities),
            planned_n_valid=tuple(planned for _ in identities),
            attempt=0,
            plan_chain_id=f"{self.plan_chain_id}-{self._window_index}",
        )
        return ActiveWindowFreeze(plan, tuple(identities), tuple(members))

    def _peek_block(self, slot_id: int) -> tuple[Any, int, bool, bool, int] | None:
        """Pure view of a slot's next whole block; commits no driver state.

        Returns ``(stream, cursor, training_stream_end, rebind_before_admit,
        catalog_position)``, or ``None`` when the slot has no whole block left.
        """
        stream = self._active_stream.get(slot_id)
        position = self._stream_index[slot_id]
        rebind = stream is not None
        if stream is not None:
            blocks = int(self.producer.block_count(stream))
            cursor = self._active_cursor[slot_id] + 1
            if cursor < blocks:
                return stream, cursor, cursor == blocks - 1, False, position
            position += 1
        while position < len(self._by_slot[slot_id]):
            candidate = self._by_slot[slot_id][position]
            blocks = int(self.producer.block_count(candidate))
            if blocks > 0:
                return candidate, 0, blocks == 1, rebind, position
            position += 1
            rebind = True
        return None

    def _commit_block(self, slot_id: int, stream: Any, cursor: int, position: int) -> None:
        self._stream_index[slot_id] = position
        self._active_stream[slot_id] = stream
        self._active_cursor[slot_id] = cursor
