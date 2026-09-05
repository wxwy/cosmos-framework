# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Trainer-seam acceptance tests for R09-B TTT v0.3.2 active wiring.

Drives the real ``ImaginaireTrainer.training_step`` / ``_optimizer_step`` with a
stub model owning a genuine ``TTTLifecycle`` and a real ``CallBackGroup``
holding ``TTTLifecycleCallback``: exactly one backward per micro-batch, closing
loss reaches the witness graph, mark+commit happen exactly once at
``on_after_backward``, backward errors abort and re-raise, and the
GradScaler-skip gate keeps the fast commit while holding the slow side.
CPU-only, no distributed, no real data.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from cosmos_framework.model.generator.mot.local_evidence import ContinualTTTLocalMemoryCore, LocalEvidenceEncoder
from cosmos_framework.model.generator.mot.production_runtime_adapter import ProductionLocalMemoryRuntime
from cosmos_framework.model.generator.mot.runtime_authority import AdmissionAuthority
from cosmos_framework.model.generator.mot.ttt_lifecycle import TTTLifecycle, TTTLifecycleCallback
from cosmos_framework.trainer import ImaginaireTrainer
from cosmos_framework.utils.callback import CallBackGroup

OWNER = "libero_spatial#0"


def _source(value: float) -> dict[str, torch.Tensor]:
    return {
        "history_visual_summary": torch.full((1, 2, 4), value),
        "local_history_action": torch.full((1, 2, 3), value),
        "history_age_steps": torch.zeros(1, 2, dtype=torch.long),
        "history_dt_s": torch.ones(1, 2, 1),
        "history_mask": torch.ones(1, 2, dtype=torch.bool),
    }


class _FakeGradScaler:
    """Minimal GradScaler stand-in; ``step`` records per-optimizer found-inf state."""

    def __init__(self, *, enabled: bool = False, found_inf: bool = False) -> None:
        self._enabled = enabled
        self._found_inf = found_inf
        self.step_calls = 0
        self.update_calls = 0

    def scale(self, value: torch.Tensor) -> torch.Tensor:
        return value

    def is_enabled(self) -> bool:
        return self._enabled

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        self.step_calls += 1
        if self._enabled:
            self._per_optimizer_states = {
                id(optimizer): {
                    "found_inf_per_device": {torch.device("cpu"): torch.tensor(1.0 if self._found_inf else 0.0)}
                }
            }

    def update(self) -> None:
        self.update_calls += 1


class _TTTStubModel(nn.Module):
    """One native window per training_step call, driven by ``plan``."""

    def __init__(self, lifecycle: TTTLifecycle, projector: nn.Linear, modality: nn.Parameter, plan: list[dict]) -> None:
        super().__init__()
        self._ttt_lifecycle = lifecycle
        self._projector = projector
        self._modality = modality
        self._plan = plan
        self._cursor = 0
        self.backward_events: list[str] = []
        self.after_backward_calls = 0
        self.last_loss: torch.Tensor | None = None

    def training_step(self, data: dict, iteration: int) -> tuple[dict, torch.Tensor]:
        del data, iteration
        spec = self._plan[self._cursor]
        self._cursor += 1
        token = self._ttt_lifecycle.process_sample(
            owner_key=spec.get("owner", OWNER),
            terminal=spec.get("terminal", False),
            valid=True,
            source=_source(spec.get("value", 0.1)),
        )
        loss = self._projector(token).square().mean() + self._modality.square().mean()
        hook = spec.get("hook")
        if hook is not None:
            loss.register_hook(hook)
        else:
            loss.register_hook(lambda _grad: self.backward_events.append("backward"))
        self.last_loss = loss
        return {"loss": loss.detach()}, loss

    def on_after_backward(self) -> None:
        self.after_backward_calls += 1

    def on_before_optimizer_step(self, *args, **kwargs) -> None:
        del args, kwargs

    def on_before_zero_grad(self, *args, **kwargs) -> None:
        del args, kwargs


def _make_lifecycle(segment_steps: int = 2) -> tuple[TTTLifecycle, nn.Linear, nn.Parameter]:
    torch.manual_seed(0)
    encoder = LocalEvidenceEncoder(evidence_dim=8, visual_dim=4, action_dim=3, max_age_steps=8)
    core = ContinualTTTLocalMemoryCore(
        evidence_dim=8, local_dim=4, ttt_dim=6, fast_hidden_dim=10, ttt_tbptt_steps=segment_steps
    )
    projector = nn.Linear(4, 5)
    modality = nn.Parameter(torch.zeros(5))
    lifecycle = TTTLifecycle(
        ProductionLocalMemoryRuntime(AdmissionAuthority(), encoder, core),
        encoder=encoder,
        core=core,
        local_memory2llm=projector,
        modality_embed=modality,
    )
    return lifecycle, projector, modality


def _make_trainer() -> ImaginaireTrainer:
    trainer = object.__new__(ImaginaireTrainer)
    trainer.config = SimpleNamespace(
        trainer=SimpleNamespace(
            grad_accum_iter=1,
            distributed_parallelism="none",
            straggler_detection=SimpleNamespace(analyze_forward=False, analyze_backward=False, analyze_optimizer=False),
        )
    )
    callbacks = object.__new__(CallBackGroup)
    callbacks._callbacks = [TTTLifecycleCallback()]
    trainer.callbacks = callbacks
    trainer.training_timer = MagicMock()
    trainer.training_timer.return_value = MagicMock(
        __enter__=MagicMock(return_value=None), __exit__=MagicMock(return_value=False)
    )
    trainer.straggler_detector = MagicMock()
    trainer.straggler_detector.profile_section.return_value = MagicMock(
        __enter__=MagicMock(return_value=None), __exit__=MagicMock(return_value=False)
    )
    return trainer


def _run_step(trainer: ImaginaireTrainer, model: nn.Module, lifecycle: TTTLifecycle, scaler: _FakeGradScaler) -> tuple:
    optimizer = torch.optim.SGD(lifecycle.slow_parameters, lr=0.1)
    scheduler = MagicMock()
    return trainer.training_step(model, optimizer, scheduler, scaler, {}, iteration=0, grad_accum_iter=0), scheduler


def _spy_marks_and_commits(lifecycle: TTTLifecycle) -> dict:
    """Wrap mark/commit with counters; records witness traversal and loss identity."""
    authority = lifecycle.runtime.runtime_authority
    records = {"marks": 0, "commits": 0, "witness_grad_present": [], "observed_loss": [], "slow_grads_at_mark": []}
    original_mark = lifecycle.runtime.mark_external_backward
    original_commit = authority.commit

    def _mark(owner_key: str, *, loss: torch.Tensor) -> None:
        records["marks"] += 1
        pending = authority._pending_by_owner[owner_key]
        records["witness_grad_present"].append(
            pending.witness_leaf is not None and pending.witness_leaf.grad is not None
        )
        records["observed_loss"].append(loss)
        # Meta-gradient reach is sampled here (post-backward, pre-optimizer-step);
        # the trainer's _zero_grad(set_to_none=True) clears slow .grad afterwards.
        records["slow_grads_at_mark"].append(
            all(param.grad is not None for param in authority.encoder.parameters())
            and all(param.grad is not None for param in authority.core.parameters())
        )
        original_mark(owner_key, loss=loss)

    def _commit(owner_key: str) -> None:
        records["commits"] += 1
        original_commit(owner_key)

    lifecycle.runtime.mark_external_backward = _mark
    authority.commit = _commit
    return records


def test_training_step_commit_seam_exactly_once_per_micro_batch() -> None:
    lifecycle, projector, modality = _make_lifecycle(segment_steps=2)
    trainer = _make_trainer()
    plan = [
        {"value": 0.1},  # intermediate window: no arm, no mark, no commit
        {"value": 0.2},  # closing window: arm -> backward -> mark -> commit
        {"value": 0.3},  # next segment intermediate: proves commit preceded this forward
    ]
    model = _TTTStubModel(lifecycle, projector, modality, plan)
    records = _spy_marks_and_commits(lifecycle)
    scaler = _FakeGradScaler(enabled=False)

    _run_step(trainer, model, lifecycle, scaler)
    assert model.backward_events == ["backward"]  # exactly one backward
    assert records["marks"] == 0 and records["commits"] == 0
    authority = lifecycle.runtime.runtime_authority
    assert OWNER in lifecycle._segments and OWNER not in authority._state_by_owner

    _run_step(trainer, model, lifecycle, scaler)
    assert model.backward_events == ["backward", "backward"]  # still exactly one per micro-batch
    assert records["marks"] == 1 and records["commits"] == 1
    # The closing backward really traversed the witness graph, and the mark saw
    # the trainer's unscaled native loss (identity, not a scaled copy).
    assert records["witness_grad_present"] == [True]
    assert records["observed_loss"][0] is model.last_loss
    # Meta-gradient reached the encoder/core groups through the witness graph
    # (sampled at mark time; the trainer's _zero_grad clears slow .grad after
    # the optimizer step).
    assert records["slow_grads_at_mark"] == [True]
    assert authority._last_timestep[OWNER] == 1
    assert authority.c5_write_count == 2
    assert model.after_backward_calls == 2

    _run_step(trainer, model, lifecycle, scaler)  # would raise if the commit had not freed the pending slot
    assert records["marks"] == 1 and records["commits"] == 1
    assert OWNER in lifecycle._segments
    assert len(authority._pending_by_owner[OWNER].rows) == 1


def test_training_step_backward_error_aborts_and_reraises() -> None:
    lifecycle, projector, modality = _make_lifecycle(segment_steps=1)
    trainer = _make_trainer()
    # Pre-commit one segment so the abort must provably preserve committed state.
    model = _TTTStubModel(lifecycle, projector, modality, [{"value": 0.1}])
    _run_step(trainer, model, lifecycle, _FakeGradScaler())
    authority = lifecycle.runtime.runtime_authority
    committed_before = {key: record.value.clone() for key, record in authority._committed.items()}
    state_before = tuple(value.clone() for value in authority._state_by_owner[OWNER])

    def _boom(_grad):
        raise RuntimeError("boom")

    failing = _TTTStubModel(lifecycle, projector, modality, [{"value": 0.2, "hook": _boom}])
    with pytest.raises(RuntimeError, match="boom"):
        _run_step(trainer, failing, lifecycle, _FakeGradScaler())
    # Abort route: open segments rolled back, original exception re-raised.
    assert not lifecycle._segments and not authority._pending_by_owner
    assert failing.after_backward_calls == 0  # on_after_backward never ran
    assert set(authority._committed) == set(committed_before)
    assert all(torch.equal(authority._committed[key].value, committed_before[key]) for key in committed_before)
    assert all(
        torch.equal(left, right) for left, right in zip(authority._state_by_owner[OWNER], state_before, strict=True)
    )


def test_optimizer_step_scaler_skip_gate() -> None:
    lifecycle, projector, modality = _make_lifecycle(segment_steps=1)
    trainer = _make_trainer()
    model = _TTTStubModel(lifecycle, projector, modality, [{"value": 0.1}])
    _run_step(trainer, model, lifecycle, _FakeGradScaler())
    authority = lifecycle.runtime.runtime_authority
    state_before = tuple(value.clone() for value in authority._state_by_owner[OWNER])
    for param in lifecycle.slow_parameters:
        param.grad = torch.ones_like(param)
    optimizer = torch.optim.SGD(lifecycle.slow_parameters, lr=0.1)

    scheduler = MagicMock()
    trainer._optimizer_step(model, optimizer, scheduler, _FakeGradScaler(enabled=True, found_inf=True), iteration=1)
    # SCALER_SKIP: fast commit stays valid, slow .grad cleared, scheduler held.
    scheduler.step.assert_not_called()
    assert all(param.grad is None for param in lifecycle.slow_parameters)
    assert all(
        torch.equal(left, right) for left, right in zip(authority._state_by_owner[OWNER], state_before, strict=True)
    )

    for param in lifecycle.slow_parameters:
        param.grad = torch.ones_like(param)
    scheduler = MagicMock()
    trainer._optimizer_step(model, optimizer, scheduler, _FakeGradScaler(enabled=True, found_inf=False), iteration=2)
    scheduler.step.assert_called_once()
    assert any(param.grad is not None for param in lifecycle.slow_parameters)

    # Disabled scaler: no per-optimizer state exists; the gate must not touch it.
    scheduler = MagicMock()
    disabled_scaler = _FakeGradScaler(enabled=False)
    trainer._optimizer_step(model, optimizer, scheduler, disabled_scaler, iteration=3)
    scheduler.step.assert_called_once()
    assert not hasattr(disabled_scaler, "_per_optimizer_states")


def test_training_step_disabled_model_gate_inert() -> None:
    """No ``_ttt_lifecycle`` on the model: the try/except and skip gate are no-ops."""

    class _PlainModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(1.0))

        def training_step(self, data: dict, iteration: int) -> tuple[dict, torch.Tensor]:
            loss = self.weight.square()
            return {"loss": loss.detach()}, loss

        def on_after_backward(self) -> None:
            pass

        def on_before_optimizer_step(self, *args, **kwargs) -> None:
            del args, kwargs

        def on_before_zero_grad(self, *args, **kwargs) -> None:
            del args, kwargs

    trainer = _make_trainer()
    model = _PlainModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = MagicMock()
    # Even with a found-inf scaler, a model without a lifecycle never skips.
    scaler = _FakeGradScaler(enabled=True, found_inf=True)
    trainer.training_step(model, optimizer, scheduler, scaler, {}, iteration=0, grad_accum_iter=0)
    scheduler.step.assert_called_once()
