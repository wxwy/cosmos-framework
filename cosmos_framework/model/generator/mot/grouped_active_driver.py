"""A2 window driver: preserve scalar scheduling, group source rows for native I/O."""

from __future__ import annotations

from .active_local_memory_driver import (
    ActiveLocalMemoryWindowDriver,
    ActiveWindowFreeze,
    ActiveWindowMember,
)
from .canonical_segment_runtime import RuntimePhase
from .grouped_active_contract import GroupedGAWindowPlan, GroupedPlanMember, stack_segments
from .local_memory_segment import GAWindowPlan, SegmentIdentity


class GroupedActiveLocalMemoryWindowDriver(ActiveLocalMemoryWindowDriver):
    member_layout = "active_a2_v1"

    def __init__(
        self, *, group_size: int, window_members: int, manifest_digest: str, config_digest: str, **kwargs
    ) -> None:
        if isinstance(group_size, bool) or not isinstance(group_size, int) or group_size <= 0:
            raise ValueError("A2 group_size must be a positive integer")
        super().__init__(window_members=window_members * group_size, **kwargs)
        self.group_size = group_size
        if len(self._by_slot) != group_size:
            raise ValueError("A2 group_size must equal the configured stable slot count")
        self.native_members = window_members
        self.manifest_digest, self.config_digest = manifest_digest, config_digest
        self._group_plan = None
        import hashlib
        import json

        catalog = [
            (
                stream.slot_id,
                stream.episode_index,
                stream.episode_position,
                stream.category,
                int(self.producer.block_count(stream)),
            )
            for streams in self._by_slot.values()
            for stream in streams
        ]
        self._source_catalog_sha256 = hashlib.sha256(json.dumps(catalog, separators=(",", ":")).encode()).hexdigest()

    def _rebind_terminal(self, slot_id: int) -> None:
        # Source production is speculative. Only the grouped owner's post-backward
        # atomic commit may change scheduler terminal/rebind authority.
        pass

    def _maybe_rollover(self) -> None:
        """Reuse only fully exhausted terminal slots at an optimizer boundary.

        Synchronized A2 requires every native member to contain one segment from
        every stable slot.  The scalar driver's aggregate remaining-block test is
        therefore not the right boundary condition: one depleted slot cannot be
        compensated by extra blocks from another slot.
        """
        for slot_id in sorted(self._by_slot):
            if self._remaining_blocks(slot_id) == 0:
                self._rollover_slot(slot_id)

    def _stage_slot_reuse_for_plan(self, slot_id: int) -> None:
        """Advance one exhausted slot to its next catalog epoch while freezing.

        This mutates only the driver's already-irreversible frozen frontier.  The
        live scheduler/sidecar are not touched here; when execution reaches the
        first cursor-0 row, the grouped runtime performs terminal rebind and, for
        an epoch-reused identity, prunes that slot's old audit history atomically.
        """
        stream = self._active_stream.get(slot_id)
        if stream is not None:
            blocks = int(self.producer.block_count(stream))
            cursor = self._active_cursor.get(slot_id)
            if cursor is None or cursor < blocks - 1:
                raise RuntimeError("A2 cannot reuse a slot before its episode is terminal")
        self._slot_epoch[slot_id] += 1
        self._by_slot[slot_id] = self._reordered_by_slot(slot_id, self._slot_epoch[slot_id])
        self._stream_index[slot_id] = 0
        self._active_stream.pop(slot_id, None)
        self._active_cursor.pop(slot_id, None)

    def freeze_window(self) -> ActiveWindowFreeze:
        """Freeze GA synchronized microbatches: exactly one next segment per slot.

        Scalar selection order is deliberately not reused.  For each native
        microbatch, rows are the stable slot ids in sorted order; within a slot,
        cursor chronology remains strict and episode rebind happens only between
        microbatches after a terminal segment.
        """
        planned = int(self.producer.ttt_tbptt_steps)
        slots = tuple(sorted(self._by_slot))
        if len(slots) != self.group_size:
            raise RuntimeError("A2 group_size must equal the stable slot count")
        members: list[ActiveWindowMember] = []
        identities: list[SegmentIdentity] = []
        for _group_index in range(self.native_members):
            group_slots: list[int] = []
            for slot_id in slots:
                view = self._peek_block(slot_id)
                if view is None:
                    self._stage_slot_reuse_for_plan(slot_id)
                    view = self._peek_block(slot_id)
                if view is None:
                    raise RuntimeError("A2 synchronized slot has no segment after catalog reuse")
                stream, cursor, terminal, rebind, position = view
                self._commit_block(slot_id, stream, cursor, position)
                group_slots.append(slot_id)
                members.append(ActiveWindowMember(stream, cursor, terminal, rebind))
                identities.append(
                    SegmentIdentity(
                        slot_id=slot_id,
                        episode_id=str(stream.episode_index),
                        category=stream.category,
                        cursor=cursor,
                        segment_id=cursor,
                        source_digest=self.producer.source_digest,
                        training_stream_end=terminal,
                    )
                )
            if tuple(group_slots) != slots:
                raise AssertionError("A2 synchronized group lost its stable slot order")
        self._window_index += 1
        plan = GAWindowPlan(
            members=tuple((item.slot_id, item.episode_id, item.cursor) for item in identities),
            planned_n_valid=tuple(planned for _ in identities),
            attempt=0,
            plan_chain_id=f"{self.plan_chain_id}-{self._window_index}",
        )
        return ActiveWindowFreeze(plan, tuple(identities), tuple(members))

    def _plan_groups(self, freeze):
        counts = (int(self.producer.ttt_tbptt_steps),) * self.group_size
        groups = tuple(
            GroupedPlanMember(
                freeze.identities[start : start + self.group_size],
                counts,
                self.producer.ttt_tbptt_steps,
                self.manifest_digest,
                self.config_digest,
                self.producer.source_digest,
            )
            for start in range(0, len(freeze.members), self.group_size)
        )
        expected_slots = tuple(sorted(self._by_slot))
        if any(tuple(identity.slot_id for identity in group.row_identities) != expected_slots for group in groups):
            raise RuntimeError("A2 plan is not stable-slot synchronized")
        return GroupedGAWindowPlan(
            members=groups,
            planned_n_valid=tuple(group.planned_n_valid for group in groups),
            plan_chain_id=freeze.plan.plan_chain_id,
        )

    def _produce_group(self, index):
        try:
            start = index * self.group_size
            rows = tuple(self._produce(self._window, i) for i in range(start, start + self.group_size))
            segment = stack_segments(rows, segment_id=index)
            self._group_plan.members[index].validate_batch(segment)
            return segment
        except Exception:
            self._abort_source()
            raise

    def _arm_initial(self, trainer, model) -> None:
        if int(trainer.config.trainer.grad_accum_iter) != self.native_members:
            raise RuntimeError("A2 grouped GA differs from trainer accumulation boundary")
        try:
            self._maybe_rollover()
            self._window = self.freeze_window()
            self._group_plan = self._plan_groups(self._window)
            self._start_prefetch(self._window)
        except Exception:
            self._abort_source()
            raise
        segment = self._produce_group(0)
        trainer.arm_active_local_memory_initial(
            model,
            self._group_plan.members[0],
            segment,
            self._group_plan,
            grad_accum_iter=0,
        )

    def _arm_continuation(self, trainer, model) -> None:
        transaction = self.registry.owner.transaction
        if self._group_plan is None or self._window is None or transaction is None:
            raise RuntimeError("A2 continuation has no frozen window")
        index = len(transaction.completed_members)
        if index >= self.native_members:
            raise RuntimeError("A2 window completed but owner remains open")
        trainer.arm_active_local_memory_continuation(
            model,
            transaction.plan.members[index],
            self._produce_group(index),
            transaction,
            grad_accum_iter=index,
        )

    def _geometry(self):
        return {
            "member_layout": self.member_layout,
            "source_catalog_sha256": self._source_catalog_sha256,
            "group_size": self.group_size,
            "native_members": self.native_members,
            "logical_members": self.window_members,
            "tbptt_steps": self.producer.ttt_tbptt_steps,
        }

    def state_dict(self):
        state = super().state_dict()
        state.update(self._geometry())
        return state

    def load_state_dict(self, state_dict) -> None:
        if any(state_dict.get(key) != value for key, value in self._geometry().items()):
            raise RuntimeError("A2 resume geometry/layout differs; use an explicit fresh run")
        super().load_state_dict(state_dict)

    def _abort_source(self):
        owner = self.registry.owner
        owner.wiring.clear_local_slow_grads()
        if owner.transaction is not None:
            owner.transaction.terminal_failure("LOCAL_MEM_GROUP_SOURCE_FAILURE")
        owner.phase = RuntimePhase.ABORTED
        self.close(wait=False)

    def close(self, *, wait=True):
        for future in self._prefetch_futures.values():
            future.cancel()
        self._prefetch_futures.clear()
        if self._prefetch_executor is not None:
            self._prefetch_executor.shutdown(wait=wait, cancel_futures=True)
            self._prefetch_executor = None

    def on_train_end(self, model, iteration=0):
        self.close()

    def on_app_end(self):
        self.close()
