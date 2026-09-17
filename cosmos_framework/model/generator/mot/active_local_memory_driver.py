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

from cosmos_framework.utils.callback import Callback

from .canonical_segment_runtime import RuntimePhase
from .local_memory_segment import GAWindowPlan, RankLocalSegmentScheduler, SegmentIdentity
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


class ActiveLocalMemoryWindowDriver(Callback):
    """Freeze and arm exactly one canonical Local-Memory window per update.

    ``streams`` is the slot catalog: every entry is one episode's segment stream,
    and entries sharing a ``slot_id`` are consumed in the given order, so a
    finished episode is rebound to its successor without the driver knowing
    anything about the dataset beyond ``producer.block_count``.

    ``Callback`` is the base class rather than a bare object because
    ``CallBackGroup.__getattr__`` asserts every member of its callback list
    implements every ``on_*`` hook; inheriting supplies the no-op hook surface
    that makes this driver a legal group member.
    """

    def __init__(
        self,
        *,
        registry: ProductionActiveWiringRegistry,
        producer: Any,
        streams: tuple[Any, ...],
        window_members: int,
        plan_chain_id: str = "active-local-window",
        catalog_digest: str | None = None,
    ) -> None:
        super().__init__()
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
        # catalog_digest must be supplied explicitly (the launch callback derives it
        # from manifest|config|source); fall back to source_digest only for legacy
        # callers that never persist catalog identity, and keep the distinction
        # explicit so a missing catalog digest is never silently masked.
        self.catalog_digest = catalog_digest if catalog_digest is not None else getattr(producer, "source_digest", None)
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
        # Selections made so far in this window, per slot. The deficit is measured
        # per category, so two slots of one category tie every round; without this
        # counter the tie-break degenerates to "largest slot_id wins" and the
        # smaller slot of every category is starved for the whole run.
        used: dict[int, int] = {}
        for _ in range(self.window_members):
            choices: list[tuple[float, str, int, int, tuple]] = []
            for slot_id in self._by_slot:
                view = self._peek_block(slot_id)
                if view is None:
                    continue
                stream, _cursor, _terminal, _rebind, _position = view
                observed = exposure.get(stream.category, 0) / max(sum(exposure.values()), 1)
                choices.append(
                    (target[stream.category] - observed, stream.category, -used.get(slot_id, 0), slot_id, view)
                )
            if not choices:
                raise RuntimeError("active Local window exhausted every segment stream")
            _, category, _used, slot_id, view = max(choices, key=lambda item: item[:3])
            stream, cursor, terminal, rebind, position = view
            self._commit_block(slot_id, stream, cursor, position)
            used[slot_id] = used.get(slot_id, 0) + 1
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

    # ---- checkpoint surface (resume wiring design v0.4) ---------------------

    def state_dict(self) -> dict[str, Any]:
        """Persist the driver's data-progress state plus the runtime snapshot.

        ``active_stream`` is serialized by value (the four scalar stream fields)
        so the checkpoint never references a live producer object.
        """
        return {
            "source_digest": getattr(self.producer, "source_digest", None),
            "catalog_digest": self.catalog_digest,
            "plan_chain_id": self.plan_chain_id,
            "window_index": self._window_index,
            "stream_index": dict(self._stream_index),
            "active_stream": {
                slot: (s.slot_id, s.episode_index, s.episode_position, s.category)
                for slot, s in self._active_stream.items()
            },
            "active_cursor": dict(self._active_cursor),
            "runtime": self.registry.owner.snapshot(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore driver/runtime state with the §4.3 fail-closed checks.

        Two-phase restore (project atomicity rule): every fallible check runs on
        staged/off-to-the-side state first; only after all checks pass does a
        single atomic apply mutate the live owner/sidecar/driver.  Any rejection
        therefore leaves the live runtime untouched.
        """
        # 4. 版本/身份一致性：catalog/config 变更后不得静默误 load。
        if state_dict.get("source_digest") != getattr(self.producer, "source_digest", None):
            raise RuntimeError("active Local-Memory cannot resume: source_digest differs")
        if state_dict.get("catalog_digest") != self.catalog_digest:
            raise RuntimeError("active Local-Memory cannot resume: catalog_digest differs")
        if state_dict.get("plan_chain_id") != self.plan_chain_id:
            raise RuntimeError("active Local-Memory cannot resume: plan_chain_id differs")
        # 2. 恢复点必须是窗口边界。
        if self._window is not None or self.registry.owner.phase is not RuntimePhase.IDLE:
            raise RuntimeError("active Local-Memory cannot resume: not at a window boundary")
        # 3. window_index 单调且合法（≥0 且 < max_iter）。
        window_index = state_dict.get("window_index")
        if not isinstance(window_index, int) or window_index < 0:
            raise RuntimeError("active Local-Memory cannot resume: invalid window_index")
        max_iter = getattr(getattr(getattr(self._trainer, "config", None), "trainer", None), "max_iter", None)
        if max_iter is not None and window_index >= max_iter:
            raise RuntimeError("active Local-Memory cannot resume: window_index exceeds max_iter")
        # ---- 阶段 1：staging/validation（不 mutate live）----
        # 1. _by_slot 重建确定性：active_stream 按值匹配，且恰好命中一个对象。
        active_stream = self._stage_active_stream(state_dict["active_stream"])
        # 6. frontier 跨字段一致性（ChatGPT MEDIUM-1）。
        stream_index, active_cursor = self._stage_frontier(state_dict, active_stream)
        # 5. runtime：旁路 rebuild scheduler + stage sidecar records + 全量校验。
        scheduler, sidecar_records = self._stage_runtime(state_dict["runtime"])
        # ---- 阶段 2：atomic apply（此后无 fallible 检查）----
        self._apply_runtime(scheduler, sidecar_records)
        self._window_index = window_index
        self._stream_index = stream_index
        self._active_stream = active_stream
        self._active_cursor = active_cursor

    def _stage_active_stream(self, serialized: dict[Any, Any]) -> dict[int, Any]:
        active_stream: dict[int, Any] = {}
        for slot, fields in serialized.items():
            if slot not in self._by_slot:
                raise RuntimeError("active Local-Memory cannot resume: frontier keys are not valid for the catalog geometry")
            slot_id, episode_index, episode_position, category = fields
            matches = [
                s
                for s in self._by_slot[slot]
                if (s.slot_id, s.episode_index, s.episode_position, s.category)
                == (slot_id, episode_index, episode_position, category)
            ]
            if len(matches) != 1:
                raise RuntimeError("active Local-Memory cannot resume: slot-to-episode binding is ambiguous")
            active_stream[slot] = matches[0]
        return active_stream

    def _stage_frontier(self, state_dict: dict[str, Any], active_stream: dict[int, Any]) -> tuple[dict[int, Any], dict[int, int]]:
        stream_index = dict(state_dict["stream_index"])
        active_cursor = dict(state_dict["active_cursor"])
        if set(active_cursor) != set(active_stream):
            raise RuntimeError("active Local-Memory cannot resume: active_stream/active_cursor key sets disagree")
        if not set(active_stream) <= set(stream_index) or not set(stream_index) <= set(self._by_slot):
            raise RuntimeError("active Local-Memory cannot resume: frontier keys are not valid for the catalog geometry")
        for slot in stream_index:
            if slot not in self._by_slot:
                raise RuntimeError("active Local-Memory cannot resume: slot missing from catalog")
            if slot in active_stream:
                positions = [i for i, s in enumerate(self._by_slot[slot]) if s is active_stream[slot]]
                if len(positions) != 1 or positions[0] != stream_index[slot]:
                    raise RuntimeError("active Local-Memory cannot resume: stream_index not aligned with active_stream position")
                blocks = int(self.producer.block_count(active_stream[slot]))
                cursor = active_cursor[slot]
                if not isinstance(cursor, int) or not (0 <= cursor < blocks):
                    raise RuntimeError("active Local-Memory cannot resume: cursor out of block range")
            elif not (isinstance(stream_index[slot], int) and 0 <= stream_index[slot] <= len(self._by_slot[slot])):
                raise RuntimeError("active Local-Memory cannot resume: invalid frontier for idle slot")
        return stream_index, active_cursor

    def _stage_runtime(self, runtime: Any) -> tuple[Any, dict[int, tuple[Any, Any]]]:
        """Rebuild scheduler and stage sidecar records off to the side (no live mutation).

        Mirrors ``owner.snapshot()``'s consistency checks (``canonical_segment_runtime.py:181-188``)
        against the *candidate* scheduler, so a malformed runtime fails before any
        live owner/sidecar state is touched.  The sidecar frontier must reuse the
        same ``SegmentIdentity`` objects that ``scheduler.rebuild`` materializes,
        otherwise the ``is``-checks fail on the next snapshot.
        """
        scheduler = RankLocalSegmentScheduler.rebuild(dict(runtime.scheduler))
        if any(i not in scheduler.committed_identities for i in scheduler.admission_order):
            raise RuntimeError("active Local-Memory cannot resume: snapshot has admitted-but-uncommitted authority")
        by_slot = {identity.slot_id: identity for identity in scheduler.committed_identities}
        records: dict[int, tuple[Any, Any]] = {}
        for identity, fast_state in runtime.committed:
            canonical = by_slot.get(identity.slot_id)
            if canonical is None:
                raise RuntimeError(
                    "active Local-Memory cannot resume: committed sidecar identity has no "
                    f"scheduler counterpart for slot {identity.slot_id}"
                )
            if scheduler.stable_slots.get(identity.slot_id) is not canonical:
                raise RuntimeError("active Local-Memory cannot resume: snapshot committed frontier mismatch")
            records[identity.slot_id] = (canonical, fast_state)
        if any(slot in by_slot for slot in scheduler.terminal_slots):
            raise RuntimeError("active Local-Memory cannot resume: snapshot terminal slot retains sidecar state")
        return scheduler, records

    def _apply_runtime(self, scheduler: Any, sidecar_records: dict[int, tuple[Any, Any]]) -> None:
        owner = self.registry.owner
        owner.scheduler = scheduler
        scheduler._canonical_runtime_owner = owner
        sidecar = owner.adapter.sidecar
        sidecar._records.clear()
        sidecar._records.update(sidecar_records)
