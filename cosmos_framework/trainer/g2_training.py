# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""G2 policy for the existing grouped trainer/optimizer, not another train loop.

Pre-register all potential GEN/Global groups using the existing optimizer
factory. Preserve its optimizer type, decay exclusions, betas, eps and state.
Only policy (trainability, per-group clocks/LR and transaction boundary) lives
here. FSDP registration/leaf changes are delegated to B's explicit hook.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.tensor import DTensor

from cosmos_framework.model.generator.mot.g2_contracts import ParameterGroup, TrainingStage

PEAK_LR = {ParameterGroup.GLOBAL: 5e-5, ParameterGroup.GEN: 1e-5, ParameterGroup.ACTION: 5e-5}
STAGE_SCHEMA = "psm_g2_training_stage_v1"
G2_OPTIMIZER_SELECTION = "g2_registered_inventory_v1"


def build_g2_optimizer(model: nn.Module, optimizer_type: str, **kwargs):
    """Resolve B's exact names, then use the unchanged native optimizer factory."""
    from cosmos_framework.utils.generator.optimizer import build_optimizer

    if list(kwargs.pop("keys_to_select", ())) != [G2_OPTIMIZER_SELECTION]:
        raise ValueError("G2 optimizer requires the registered inventory resolver policy")
    names = getattr(model, "_g2_optimizer_parameter_names", None)
    if not isinstance(names, frozenset) or not names:
        raise ValueError("install G2 training and resolve B's inventory before building the optimizer")
    return build_optimizer(
        model,
        optimizer_type=optimizer_type,
        keys_to_select=[name.removeprefix("net.") for name in sorted(names)],
        **kwargs,
    )


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


@dataclass(frozen=True)
class GroupSchedule:
    warmup_steps: int
    decay: str
    min_factor: float
    horizon_steps: int

    def __post_init__(self):
        if type(self.warmup_steps) is not int or self.warmup_steps < 0:
            raise ValueError("warmup must be explicitly set to a nonnegative integer")
        if self.decay not in {"constant", "linear", "cosine"} or not math.isfinite(self.min_factor):
            raise ValueError("explicit supported decay/min_factor is required")
        if not 0 <= self.min_factor <= 1:
            raise ValueError("min_factor must be in [0,1]")
        if type(self.horizon_steps) is not int or self.horizon_steps < max(1, self.warmup_steps):
            raise ValueError("explicit schedule horizon must be positive and cover warmup")

    def factor(self, completed: int, total: int) -> float:
        if self.warmup_steps > total:
            raise ValueError("group warmup exceeds its own active update budget")
        if completed < self.warmup_steps:
            return (completed + 1) / self.warmup_steps
        # R1/R2 have the same LR at the same GEN-update count, although their
        # active budgets differ. Shorter runs truncate this explicit horizon;
        # longer runs hold the configured minimum. Do not infer it from S_A/B.
        progress = min(
            1.0, max(0.0, (completed - self.warmup_steps) / max(1, self.horizon_steps - self.warmup_steps - 1))
        )
        remaining = {"constant": 1.0, "linear": 1 - progress, "cosine": (1 + math.cos(math.pi * progress)) / 2}[
            self.decay
        ]
        return self.min_factor + (1 - self.min_factor) * remaining


@dataclass(frozen=True)
class G2TrainingPlan:
    # Every run choice below is REQUIRED; no guessed formal checkpoint/budget.
    profile: str
    init_checkpoint_identity: str
    data_identity: str
    split_identity: str
    text_recipe_identity: str
    state15_recipe_identity: str
    stage_a_steps: int
    stage_b_steps: int
    target_effective_consumers: int
    b_stream: int
    grad_accum_steps: int
    world_size: int
    global_schedule: GroupSchedule | None
    gen_schedule: GroupSchedule
    action_schedule: GroupSchedule

    def __post_init__(self):
        if self.profile not in {"R1", "R2"}:
            raise ValueError("only R1/R2 are new training configurations; R0 is historical")
        if any(
            not isinstance(value, str) or not value
            for value in (
                self.init_checkpoint_identity,
                self.data_identity,
                self.split_identity,
                self.text_recipe_identity,
                self.state15_recipe_identity,
            )
        ):
            raise ValueError("formal initialization/data/split/text/state recipe identities are required")
        if any(
            type(value) is not int or value < 1
            for value in (
                self.stage_a_steps,
                self.stage_b_steps,
                self.target_effective_consumers,
                self.b_stream,
                self.grad_accum_steps,
                self.world_size,
            )
        ):
            raise ValueError("S_A/S_B, effective consumers and batch/GA/world must be explicitly positive")
        if (self.global_schedule is None) != (self.profile == "R1"):
            raise ValueError("R1 must have no Global schedule; R2 requires one")
        for role in self.roles:
            self.schedule(role).factor(0, self.active_steps(role))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> G2TrainingPlan:
        body = dict(value)
        try:
            for key in ("global_schedule", "gen_schedule", "action_schedule"):
                if isinstance(body.get(key), dict):
                    body[key] = GroupSchedule(**body[key])
            return cls(**body)
        except TypeError as error:
            raise ValueError(f"incomplete/unknown G2 training plan fields: {error}") from error

    @property
    def total_steps(self) -> int:
        return self.stage_a_steps + self.stage_b_steps

    @property
    def roles(self) -> tuple[ParameterGroup, ...]:
        return (
            (ParameterGroup.GLOBAL, ParameterGroup.GEN, ParameterGroup.ACTION)
            if self.profile == "R2"
            else (
                ParameterGroup.GEN,
                ParameterGroup.ACTION,
            )
        )

    @property
    def identity(self) -> str:
        return _hash(asdict(self))

    @property
    def required_checkpoint_iterations(self) -> tuple[int, ...]:
        # R1 retains equal GEN-update and equal-total-budget comparison points.
        return (
            (self.stage_b_steps, self.total_steps)
            if self.profile == "R1"
            else (
                self.stage_a_steps,
                self.total_steps,
            )
        )

    def schedule(self, role: ParameterGroup) -> GroupSchedule:
        schedule = {
            ParameterGroup.GLOBAL: self.global_schedule,
            ParameterGroup.GEN: self.gen_schedule,
            ParameterGroup.ACTION: self.action_schedule,
        }[role]
        if not isinstance(schedule, GroupSchedule):
            raise ValueError(f"explicit {role} warmup/decay is required")
        return schedule

    def active_steps(self, role: ParameterGroup) -> int:
        return self.stage_b_steps if self.profile == "R2" and role != ParameterGroup.GLOBAL else self.total_steps


def parameter_roles(
    model: nn.Module,
    *,
    global_names: frozenset[str],
    gen_names: frozenset[str],
    action_names: frozenset[str],
    profile: str,
) -> dict[str, ParameterGroup]:
    named = dict(model.named_parameters())
    if global_names & gen_names or not action_names <= gen_names or not action_names or not gen_names - action_names:
        raise ValueError("Global/GEN/action ownership is missing or overlapping")
    if (bool(global_names) != (profile == "R2")) or not (global_names | gen_names) <= named.keys():
        raise ValueError("Global enabled/inventory or named parameter ownership mismatch")
    roles = {name: ParameterGroup.GLOBAL for name in global_names}
    roles.update({name: ParameterGroup.ACTION if name in action_names else ParameterGroup.GEN for name in gen_names})
    aliases: dict[int, list[str]] = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        aliases.setdefault(id(parameter), []).append(name)
    for names in aliases.values():
        owners = {roles.get(name) for name in names}
        if len(owners) > 1:
            raise ValueError(f"parameter aliases have conflicting ownership: {names}")
    return roles


def _inners(optimizer):
    return optimizer.optimizers if hasattr(optimizer, "optimizers") else (optimizer,)


def validate_comparison_plans(r1: G2TrainingPlan, r2: G2TrainingPlan) -> dict[str, Any]:
    if (r1.profile, r2.profile) != ("R1", "R2"):
        raise ValueError("comparison requires the R1 and R2 configurations")
    common = (
        "init_checkpoint_identity",
        "data_identity",
        "split_identity",
        "text_recipe_identity",
        "state15_recipe_identity",
        "stage_a_steps",
        "stage_b_steps",
        "target_effective_consumers",
        "gen_schedule",
        "action_schedule",
    )
    different = [field for field in common if getattr(r1, field) != getattr(r2, field)]
    if different:
        raise ValueError(f"R1/R2 shared initialization/input/GEN schedule/budget mismatch: {different}")
    return {
        "r1_plan": r1.identity,
        "r2_plan": r2.identity,
        "target_effective_consumers": r1.target_effective_consumers,
        "budget_equality": "requires actual committed consumer counters, not nominal B*T*GA",
        "r1_checkpoints": r1.required_checkpoint_iterations,
        "r2_checkpoints": r2.required_checkpoint_iterations,
    }


def validate_actual_comparison_budget(r1_state, r2_state):
    if (
        r1_state.get("schema") != STAGE_SCHEMA
        or r2_state.get("schema") != STAGE_SCHEMA
        or r1_state.get("stage") != TrainingStage.R1.value
        or r2_state.get("stage") != TrainingStage.TRAIN_B.value
        or type(r1_state.get("effective_consumers")) is not int
        or r1_state["effective_consumers"] <= 0
        or r1_state["effective_consumers"] != r2_state.get("effective_consumers")
    ):
        raise ValueError("R1/R2 actual effective consumer budgets are not equal or unproven")
    return {
        "effective_consumers": r1_state["effective_consumers"],
        "r1_gen_updates": r1_state["group_updates"]["gen"],
        "r2_gen_updates": r2_state["group_updates"]["gen"],
        "compute_cost": "report separately; equal consumers does not imply equal compute",
    }


class G2StageController:
    """Stateful scheduler + policy, committed by GroupedLocalMemoryTrainer.

    scheduler.step() marks the optimizer transaction; commit_boundary() runs
    only AFTER Local/frontier commit and zero_grad. No stage change can happen
    in forward/backward or midway through GA. Global optimizer objects/groups
    are never replaced and their moments/step are never cleared.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Any,
        plan: G2TrainingPlan,
        roles: Mapping[str, ParameterGroup],
        *,
        set_trainability: Callable[[frozenset[str]], None] | None = None,
    ):
        self.model, self.optimizer, self.plan = model, optimizer, plan
        self.named = dict(model.named_parameters())
        self.roles = dict(roles)
        if set(self.roles.values()) != set(plan.roles) or not self.roles.keys() <= self.named.keys():
            raise ValueError("parameter groups must exactly cover the configured roles")
        self._set_trainability = set_trainability
        if any(isinstance(parameter, DTensor) for parameter in self.named.values()) and set_trainability is None:
            raise ValueError("FSDP requires B's explicit registered trainability hook")
        selected = [p for inner in _inners(optimizer) for group in inner.param_groups for p in group["params"]]
        expected = {id(self.named[name]) for name in self.roles}
        if len(selected) != len({id(p) for p in selected}) or {id(p) for p in selected} != expected:
            raise ValueError("optimizer parameter inventory has omissions/duplicates/frozen parameters")
        by_id = {id(parameter): name for name, parameter in self.named.items()}
        # Split the existing factory's decay/mesh groups, before any state is
        # created. Preserve all factory metadata and each actual Parameter.
        for inner in _inners(optimizer):
            if inner.state:
                raise ValueError("attach G2 groups before loading/creating optimizer state")
            groups = []
            for old in inner.param_groups:
                for role in plan.roles:
                    params = [p for p in old["params"] if self.roles[by_id[id(p)]] == role]
                    if params:
                        groups.append(
                            {
                                **old,
                                "params": params,
                                "g2_role": role.value,
                                "g2_weight_decay": float(old.get("weight_decay", 0.0)),
                            }
                        )
            inner.param_groups[:] = groups
        self.completed_steps = 0
        self.effective_consumers = 0
        self.group_updates = {role.value: 0 for role in plan.roles}
        self.group_consumers = {role.value: 0 for role in plan.roles}
        self._pending_count: int | None = None
        self._optimizer_stepped = False
        self._active_trainable_names: frozenset[str] | None = None
        self._apply_policy()

    @property
    def stage(self) -> TrainingStage:
        if self.plan.profile == "R1":
            return TrainingStage.R1
        return TrainingStage.TRAIN_A if self.completed_steps < self.plan.stage_a_steps else TrainingStage.TRAIN_B

    @property
    def active_roles(self) -> frozenset[ParameterGroup]:
        return (
            frozenset((ParameterGroup.GLOBAL,)) if self.stage == TrainingStage.TRAIN_A else frozenset(self.plan.roles)
        )

    def _apply_policy(self):
        active = frozenset(name for name, role in self.roles.items() if role in self.active_roles)
        if active != self._active_trainable_names:
            if self._set_trainability is None:
                for name, parameter in self.named.items():
                    parameter.requires_grad_(name in active)
            else:
                self._set_trainability(active)
            self._active_trainable_names = active
        for name, parameter in self.named.items():
            if parameter.requires_grad != (name in active):
                raise ValueError(f"backend failed to apply G2 trainability: {name}")
            if name not in active:
                parameter.grad = None
        for inner in _inners(self.optimizer):
            for group in inner.param_groups:
                role = ParameterGroup(group["g2_role"])
                enabled = role in self.active_roles
                factor = self.plan.schedule(role).factor(self.group_updates[role.value], self.plan.active_steps(role))
                group["lr"] = PEAK_LR[role] * factor if enabled else 0.0
                group["weight_decay"] = group["g2_weight_decay"] if enabled else 0.0

    def _synchronize(self, actual_consumers: int) -> int:
        signature = (
            self.plan.identity,
            self.stage.value,
            self.completed_steps,
            self.group_updates,
            self.group_consumers,
            self.effective_consumers,
        )
        if dist.is_available() and dist.is_initialized():
            signatures = [None] * dist.get_world_size()
            dist.all_gather_object(signatures, signature)
            if any(other != signature for other in signatures):
                raise ValueError("ranks disagree on G2 stage/optimizer clocks")
            if dist.get_world_size() != self.plan.world_size:
                raise ValueError("G2 plan world size differs from actual process group")
            count = torch.tensor(actual_consumers, dtype=torch.int64, device=next(iter(self.named.values())).device)
            dist.all_reduce(count, op=dist.ReduceOp.SUM)
            return int(count.item())
        if self.plan.world_size != 1:
            raise ValueError("multi-rank plan requires an initialized process group")
        return actual_consumers

    def prepare_commit(self, *, actual_consumers: int, iteration: int):
        """All ranks call after last GA backward, BEFORE optimizer mutation."""
        if (
            type(actual_consumers) is not int
            or actual_consumers <= 0
            or iteration != self.completed_steps
            or self._pending_count is not None
            or self._optimizer_stepped
            or self.completed_steps >= self.plan.total_steps
        ):
            raise RuntimeError("invalid or partial G2 optimizer transaction")
        count = self._synchronize(actual_consumers)
        projected = self.effective_consumers + count
        if projected > self.plan.target_effective_consumers or (
            self.completed_steps + 1 == self.plan.total_steps and projected != self.plan.target_effective_consumers
        ):
            raise ValueError("actual effective consumer budget differs from the frozen plan")
        active = self.active_roles
        for name, parameter in self.named.items():
            if self.roles.get(name) not in active and parameter.grad is not None:
                raise ValueError(f"frozen parameter has a stale gradient: {name}")
        self._pending_count = count

    def step(self):
        """Called by the existing optimizer success path; no premature switch."""
        if self._pending_count is None or self._optimizer_stepped:
            raise RuntimeError("G2 scheduler step requires one prepared optimizer transaction")
        self._optimizer_stepped = True

    def commit_boundary(self, *, iteration: int, ga_complete: bool, local_transaction_complete: bool):
        if (
            not ga_complete
            or not local_transaction_complete
            or not self._optimizer_stepped
            or self._pending_count is None
            or iteration != self.completed_steps + 1
            or any(parameter.grad is not None for parameter in self.named.values())
        ):
            raise RuntimeError("G2 stage switch requires committed optimizer/GA/Local and cleared gradients")
        old_stage = self.stage
        count = self._pending_count
        for role in self.active_roles:
            self.group_updates[role.value] += 1
            self.group_consumers[role.value] += count
        self.completed_steps += 1
        self.effective_consumers += count
        self._pending_count = None
        self._optimizer_stepped = False
        if old_stage != self.stage and dist.is_available() and dist.is_initialized():
            # Every rank has now committed Local and cleared its gradients.
            dist.barrier()
        self._apply_policy()

    def state_dict(self):
        if self._pending_count is not None or self._optimizer_stepped:
            raise RuntimeError("cannot checkpoint a partial G2 optimizer/Local transaction")
        return {
            "schema": STAGE_SCHEMA,
            "plan_identity": self.plan.identity,
            "stage": self.stage.value,
            "completed_steps": self.completed_steps,
            "effective_consumers": self.effective_consumers,
            "group_updates": self.group_updates.copy(),
            "group_consumers": self.group_consumers.copy(),
            "parameter_roles": {name: role.value for name, role in sorted(self.roles.items())},
        }

    def load_state_dict(self, state):
        expected_keys = set(self.state_dict())
        if (
            not isinstance(state, dict)
            or set(state) != expected_keys
            or state["schema"] != STAGE_SCHEMA
            or state["plan_identity"] != self.plan.identity
            or state["parameter_roles"] != {name: role.value for name, role in sorted(self.roles.items())}
        ):
            raise ValueError("G2 resume stage/config/parameter identity mismatch")
        steps, consumers = state["completed_steps"], state["effective_consumers"]
        if (
            type(steps) is not int
            or not 0 <= steps <= self.plan.total_steps
            or type(consumers) is not int
            or not 0 <= consumers <= self.plan.target_effective_consumers
        ):
            raise ValueError("invalid G2 resume counters")
        expected_updates = {
            role.value: max(0, steps - self.plan.stage_a_steps)
            if self.plan.profile == "R2" and role != ParameterGroup.GLOBAL
            else steps
            for role in self.plan.roles
        }
        expected_stage = (
            TrainingStage.R1
            if self.plan.profile == "R1"
            else (TrainingStage.TRAIN_A if steps < self.plan.stage_a_steps else TrainingStage.TRAIN_B)
        )
        group_consumers = state["group_consumers"]
        updates = state["group_updates"]
        counts_valid = (
            isinstance(group_consumers, dict)
            and set(group_consumers) == set(expected_updates)
            and all(
                type(value) is int and expected_updates[role] <= value <= consumers
                for role, value in group_consumers.items()
            )
            and all(value == 0 for role, value in group_consumers.items() if expected_updates[role] == 0)
        )
        if counts_valid:
            if self.plan.profile == "R1":
                counts_valid = all(value == consumers for value in group_consumers.values())
            else:
                counts_valid = (
                    group_consumers["global"] == consumers
                    and group_consumers["gen"] == group_consumers["action_adapter"]
                )
        if (
            updates != expected_updates
            or not isinstance(updates, dict)
            or any(type(value) is not int for value in updates.values())
            or state["stage"] != expected_stage.value
            or not counts_valid
            or (steps == 0) != (consumers == 0)
            or (steps == self.plan.total_steps and consumers != self.plan.target_effective_consumers)
        ):
            raise ValueError("G2 resume group clocks/stage/actual consumer counts disagree")
        self.completed_steps, self.effective_consumers = steps, consumers
        self.group_updates, self.group_consumers = expected_updates, group_consumers.copy()
        self._active_trainable_names = None
        self._apply_policy()

    def inventory(self) -> list[dict[str, Any]]:
        by_id = {
            id(parameter): group
            for inner in _inners(self.optimizer)
            for group in inner.param_groups
            for parameter in group["params"]
        }
        return [
            {
                "name": name,
                "shape": list(parameter.shape),
                "numel": parameter.numel(),
                "group": self.roles[name].value if name in self.roles else "frozen",
                "requires_grad": parameter.requires_grad,
                "lr": by_id.get(id(parameter), {}).get("lr", 0.0),
                "peak_lr": PEAK_LR[self.roles[name]] if name in self.roles else 0.0,
            }
            for name, parameter in sorted(self.named.items())
        ]
