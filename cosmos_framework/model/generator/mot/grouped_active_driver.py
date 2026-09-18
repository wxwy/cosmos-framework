"""A2 window driver: preserve scalar scheduling, group source rows for native I/O."""

from __future__ import annotations

from .active_local_memory_driver import ActiveLocalMemoryWindowDriver
from .canonical_segment_runtime import RuntimePhase
from .grouped_active_contract import GroupedGAWindowPlan, GroupedPlanMember, stack_segments


class GroupedActiveLocalMemoryWindowDriver(ActiveLocalMemoryWindowDriver):
    member_layout = "active_a2_v1"

    def __init__(
        self, *, group_size: int, window_members: int, manifest_digest: str, config_digest: str, **kwargs
    ) -> None:
        if isinstance(group_size, bool) or not isinstance(group_size, int) or group_size <= 0:
            raise ValueError("A2 group_size must be a positive integer")
        super().__init__(window_members=window_members * group_size, **kwargs)
        self.group_size = group_size
        self.native_members = window_members
        self.manifest_digest, self.config_digest = manifest_digest, config_digest
        self._group_plan = None

    def _rebind_terminal(self, slot_id: int) -> None:
        # Source production is speculative. Only the grouped owner's post-backward
        # atomic commit may change scheduler terminal/rebind authority.
        pass

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
