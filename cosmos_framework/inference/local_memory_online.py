"""Inference-only transactional Local-TTT fast state.

This module intentionally contains no trainer, optimizer, dataset or checkpoint
authority. Slow model parameters are read-only; only ephemeral fast weights are
adapted from completed causal evidence.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Callable
from dataclasses import dataclass

import torch

from cosmos_framework.model.generator.mot.local_evidence import (
    CANONICAL_EVIDENCE_FEATURE_CONFIG,
    ContinualTTTFastState,
    ContinualTTTLocalMemoryCore,
    LocalEvidenceEncoder,
)


@dataclass(frozen=True)
class OnlineMemoryRequest:
    session_id: str
    episode_id: str
    consumer_step: int
    source_steps: tuple[int, ...]
    visual_summary: torch.Tensor
    executed_action: torch.Tensor
    reset: bool = False


@dataclass(frozen=True)
class _Record:
    episode_id: str
    consumer_step: int
    state: ContinualTTTFastState | None
    token: torch.Tensor | None
    fingerprint: str


@dataclass(frozen=True)
class OnlineMemoryUpdate:
    owner: "OnlineLocalMemory"
    session_id: str
    previous: _Record | None
    replacement: _Record
    replay: bool
    telemetry: dict[str, float]

    @property
    def token(self) -> torch.Tensor | None:
        token = self.replacement.token
        return None if token is None else token.detach().clone()


class OnlineLocalMemory:
    """Per-session Local-TTT state with prepare/commit/abort semantics."""

    def __init__(
        self,
        encoder: LocalEvidenceEncoder,
        core: ContinualTTTLocalMemoryCore,
        *,
        scan_local_memory: Callable[..., tuple[torch.Tensor, ContinualTTTFastState, torch.Tensor]] | None = None,
        max_sessions: int = 64,
        max_evidence_steps: int = 256,
    ) -> None:
        if encoder.feature_config != CANONICAL_EVIDENCE_FEATURE_CONFIG:
            raise ValueError("online Local-TTT requires canonical evidence features")
        if encoder.evidence_dim != core.evidence_dim:
            raise ValueError("online Local-TTT encoder/core dimensions do not match")
        if max_sessions <= 0 or max_evidence_steps <= 0:
            raise ValueError("online Local-TTT bounds must be positive")
        action_dim = getattr(encoder.action_proj, "in_features", None)
        if not isinstance(action_dim, int) or action_dim <= 0:
            raise ValueError("online Local-TTT could not resolve evidence action width")
        self.encoder = encoder
        self.core = core
        self.scan_local_memory = scan_local_memory
        self.action_dim = action_dim
        self.max_sessions = int(max_sessions)
        self.max_evidence_steps = int(max_evidence_steps)
        self._records: dict[str, _Record] = {}
        self._pending: dict[str, OnlineMemoryUpdate] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _detach_state(state: ContinualTTTFastState) -> ContinualTTTFastState:
        return ContinualTTTFastState(*(value.detach().float().clone() for value in state))

    @staticmethod
    def _state_norm(state: ContinualTTTFastState | None) -> float:
        if state is None:
            return 0.0
        total = sum(float(value.detach().float().square().sum().cpu()) for value in state)
        return total**0.5

    def _validate(self, request: OnlineMemoryRequest) -> tuple[torch.Tensor, torch.Tensor, str]:
        if not isinstance(request, OnlineMemoryRequest):
            raise TypeError("online Local-TTT requires OnlineMemoryRequest")
        for value in (request.session_id, request.episode_id):
            if not isinstance(value, str) or not value.strip() or len(value) > 256:
                raise ValueError("session_id/episode_id must be non-empty bounded strings")
        if type(request.consumer_step) is not int or request.consumer_step < 0 or type(request.reset) is not bool:
            raise ValueError("invalid consumer_step/reset")
        n = len(request.source_steps)
        if n > self.max_evidence_steps or any(type(step) is not int or step < 0 for step in request.source_steps):
            raise ValueError("invalid completed-evidence chronology")
        visual = request.visual_summary.detach().to(device="cpu", dtype=torch.float32).clone()
        action = request.executed_action.detach().to(device="cpu", dtype=torch.float32).clone()
        if tuple(visual.shape) != (n, 96) or tuple(action.shape) != (n, self.action_dim):
            raise ValueError(
                f"online evidence must be [N,96] and [N,{self.action_dim}], got "
                f"{tuple(visual.shape)} and {tuple(action.shape)}"
            )
        if not torch.isfinite(visual).all() or not torch.isfinite(action).all():
            raise ValueError("online evidence must be finite")
        identity = (
            request.session_id,
            request.episode_id,
            request.consumer_step,
            request.source_steps,
            request.reset,
        )
        digest = hashlib.sha256(json.dumps(identity, separators=(",", ":")).encode())
        digest.update(visual.contiguous().numpy().tobytes())
        digest.update(action.contiguous().numpy().tobytes())
        return visual, action, digest.hexdigest()

    def prepare(self, request: OnlineMemoryRequest) -> OnlineMemoryUpdate:
        """Stage a fast-state update; caller commits only after generation succeeds."""
        with self._lock, torch.inference_mode(False):
            visual, action, fingerprint = self._validate(request)
            if request.session_id in self._pending:
                raise RuntimeError("session already has a pending Local-TTT prediction")
            previous = self._records.get(request.session_id)
            if previous is not None and previous.fingerprint == fingerprint:
                update = OnlineMemoryUpdate(self, request.session_id, previous, previous, True, {})
                self._pending[request.session_id] = update
                return update

            fresh = previous is None or request.reset
            device = next(self.encoder.parameters()).device
            if fresh:
                if request.consumer_step != 0 or request.source_steps:
                    raise ValueError("fresh/reset Local-TTT episode must start at step0 without evidence")
                new_pending = sum(update.previous is None for update in self._pending.values())
                if previous is None and len(self._records) + new_pending >= self.max_sessions:
                    raise RuntimeError("online Local-TTT session limit reached")
                # No evidence exists at consumer step0, so keep fast state lazy.
                # The model-owned scan will materialize W0 safely on the first update.
                state = None
                token = None
                telemetry = {
                    "adapted_steps": 0.0,
                    "fast_state_norm": 0.0,
                    "fast_update_norm": 0.0,
                }
            else:
                if previous.episode_id != request.episode_id:
                    raise ValueError("episode changed without explicit Local-TTT reset")
                if request.consumer_step <= previous.consumer_step:
                    raise ValueError("stale or changed-bytes Local-TTT request")
                expected = tuple(range(previous.consumer_step, request.consumer_step))
                if request.source_steps != expected:
                    raise ValueError("completed Local-TTT evidence must be contiguous and exactly-once")
                visual = visual.to(device)
                action = action.to(device)
                state = previous.state
                token = previous.token
                self.core.reset_telemetry()
                if self.scan_local_memory is not None:
                    valid = torch.ones(1, len(expected), dtype=torch.bool, device=device)
                    with torch.enable_grad():
                        tokens, candidate, present = self.scan_local_memory(
                            visual.unsqueeze(0),
                            action.unsqueeze(0),
                            valid,
                            state,
                            create_graph=False,
                        )
                    if not bool(present.all()):
                        raise RuntimeError("online model-owned Local scan dropped completed evidence")
                    state = self._detach_state(candidate)
                    token = tokens[0, -1].detach().float().clone()
                else:
                    if state is None:
                        state = self._detach_state(self.core.initial_state(1))
                    with torch.no_grad():
                        evidence = self.encoder.encode_segment(visual.unsqueeze(0), action.unsqueeze(0)).squeeze(0)
                    for index in range(len(expected)):
                        with torch.enable_grad():
                            tokens, candidate = self.core.step_many(
                                evidence[index : index + 1],
                                state,
                                create_graph=False,
                            )
                        state = self._detach_state(candidate)
                        token = tokens[0].detach().float().clone()
                inner = self.core.drain_telemetry()
                inner_sum = inner.get("ttt_inner_loss_sum")
                inner_count = inner.get("ttt_inner_loss_count")
                update_sum = inner.get("ttt_fast_update_norm_sum")
                update_count = inner.get("ttt_fast_update_norm_count")
                telemetry = {
                    "adapted_steps": float(len(expected)),
                    "fast_state_norm": self._state_norm(state),
                    "fast_update_norm": (
                        float((update_sum / update_count).cpu())
                        if update_sum is not None and update_count is not None and float(update_count) > 0
                        else 0.0
                    ),
                    "inner_loss_mean": (
                        float((inner_sum / inner_count).cpu())
                        if inner_sum is not None and inner_count is not None and float(inner_count) > 0
                        else 0.0
                    ),
                }
                if token is None or tuple(token.shape) != (self.core.k_local, self.core.local_dim):
                    raise ValueError("online Local-TTT produced an invalid Local prefix")
                if not torch.isfinite(token).all() or any(not torch.isfinite(value).all() for value in state):
                    raise FloatingPointError("online Local-TTT produced non-finite fast state/prefix")

            replacement = _Record(request.episode_id, request.consumer_step, state, token, fingerprint)
            update = OnlineMemoryUpdate(
                self,
                request.session_id,
                previous,
                replacement,
                False,
                telemetry,
            )
            self._pending[request.session_id] = update
            return update

    def commit(self, update: OnlineMemoryUpdate) -> None:
        with self._lock:
            if (
                update.owner is not self
                or self._pending.get(update.session_id) is not update
                or self._records.get(update.session_id) is not update.previous
            ):
                raise RuntimeError("online Local-TTT commit requires exact pending capability")
            self._records[update.session_id] = update.replacement
            del self._pending[update.session_id]

    def abort(self, update: OnlineMemoryUpdate) -> None:
        with self._lock:
            if update.owner is self and self._pending.get(update.session_id) is update:
                del self._pending[update.session_id]

    def reset_session(self, session_id: str) -> None:
        with self._lock:
            if session_id in self._pending:
                raise RuntimeError("cannot reset Local-TTT during pending generation")
            self._records.pop(session_id, None)

    def info(self) -> dict[str, object]:
        with self._lock:
            return {
                "enabled": True,
                "memory_kind": "ttt_fast_weight",
                "action_dim": self.action_dim,
                "k_local": self.core.k_local,
                "local_dim": self.core.local_dim,
                "ttt_dim": self.core.ttt_dim,
                "tbptt_steps": self.core.ttt_tbptt_steps,
                "inner_lr": self.core.inner_lr,
                "sessions": len(self._records),
                "pending": len(self._pending),
            }
