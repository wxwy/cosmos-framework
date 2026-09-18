"""Causal inference-only Local Memory; completed evidence in, detached prefix out.

No trainer, optimizer, dataset, or checkpoint authority is imported here. Each
session has an independent episode state. Adaptation uses gradients only with
respect to ephemeral fast weights; model parameters and their .grad are untouched.
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass

import torch

from cosmos_framework.model.generator.mot.local_evidence import (
    CANONICAL_EVIDENCE_FEATURE_CONFIG,
    ContinualTTTFastState,
    ContinualTTTLocalMemoryCore,
    LocalEvidenceEncoder,
)

EVIDENCE_VERSION = "causal_visual96_executed_action10_v1"


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
    state: ContinualTTTFastState
    token: torch.Tensor | None
    fingerprint: str


@dataclass(frozen=True)
class OnlineMemoryUpdate:
    owner: "OnlineLocalMemory"
    session_id: str
    previous: _Record | None
    replacement: _Record
    replay: bool

    @property
    def token(self) -> torch.Tensor | None:
        token = self.replacement.token
        return None if token is None else token.detach().clone()


class OnlineLocalMemory:
    """Transactional session memory; commit only after successful action generation."""

    def __init__(
        self,
        encoder: LocalEvidenceEncoder,
        core: ContinualTTTLocalMemoryCore,
        *,
        max_sessions: int = 64,
        max_evidence_steps: int = 256,
    ):
        if encoder.feature_config != CANONICAL_EVIDENCE_FEATURE_CONFIG:
            raise ValueError("online memory requires the canonical visual96/action10 encoder")
        if max_sessions <= 0 or max_evidence_steps <= 0:
            raise ValueError("online memory bounds must be positive")
        self.encoder, self.core = encoder, core
        self.max_sessions, self.max_evidence_steps = max_sessions, max_evidence_steps
        self._records: dict[str, _Record] = {}
        self._pending: dict[str, OnlineMemoryUpdate] = {}
        self._lock = threading.RLock()

    def _validate(self, request: OnlineMemoryRequest):
        if not isinstance(request, OnlineMemoryRequest):
            raise TypeError("online memory requires a typed request")
        for identity in (request.session_id, request.episode_id):
            if not isinstance(identity, str) or not identity.strip() or len(identity) > 256:
                raise ValueError("session and episode identifiers must be non-empty bounded strings")
        if type(request.consumer_step) is not int or request.consumer_step < 0 or type(request.reset) is not bool:
            raise ValueError("invalid consumer step/reset")
        n = len(request.source_steps)
        if n > self.max_evidence_steps or any(type(step) is not int or step < 0 for step in request.source_steps):
            raise ValueError("invalid or oversized completed-evidence chronology")
        visual = request.visual_summary.detach().to(device="cpu", dtype=torch.float32).clone()
        action = request.executed_action.detach().to(device="cpu", dtype=torch.float32).clone()
        if tuple(visual.shape) != (n, 96) or tuple(action.shape) != (n, 10):
            raise ValueError("online evidence must have shapes [N,96] and [N,10]")
        if not torch.isfinite(visual).all() or not torch.isfinite(action).all():
            raise ValueError("online evidence must be finite")
        identity = (
            EVIDENCE_VERSION,
            request.session_id,
            request.episode_id,
            request.consumer_step,
            request.source_steps,
            request.reset,
        )
        h = hashlib.sha256(json.dumps(identity, separators=(",", ":")).encode())
        h.update(visual.contiguous().numpy().tobytes())
        h.update(action.contiguous().numpy().tobytes())
        return visual, action, h.hexdigest()

    def prepare(self, request: OnlineMemoryRequest) -> OnlineMemoryUpdate:
        with self._lock, torch.inference_mode(False):
            visual, action, fingerprint = self._validate(request)
            if request.session_id in self._pending:
                raise RuntimeError("session already has a pending prediction")
            previous = self._records.get(request.session_id)
            if previous is not None and previous.fingerprint == fingerprint:
                update = OnlineMemoryUpdate(self, request.session_id, previous, previous, True)
                self._pending[request.session_id] = update
                return update
            fresh = previous is None or request.reset
            if fresh:
                if request.consumer_step != 0 or request.source_steps:
                    raise ValueError("a fresh/reset episode must begin at consumer step0 without evidence")
                new_pending = sum(u.previous is None for u in self._pending.values())
                if previous is None and len(self._records) + new_pending >= self.max_sessions:
                    raise RuntimeError("session limit reached; explicitly close an episode before admitting another")
                with torch.no_grad():
                    device = next(self.encoder.parameters()).device
                    state = self._detach(self.core.initial_state(1, device=device))
                token = None
            else:
                if previous.episode_id != request.episode_id:
                    raise ValueError("episode changed without explicit reset")
                if request.consumer_step <= previous.consumer_step:
                    raise ValueError("stale request or changed-bytes replay")
                expected = tuple(range(previous.consumer_step, request.consumer_step))
                if request.source_steps != expected:
                    raise ValueError(
                        "every completed source step must appear exactly once, without gaps or future evidence"
                    )
                state, token = previous.state, previous.token
                device = next(self.encoder.parameters()).device
                visual, action = visual.to(device), action.to(device)
                for index in range(len(expected)):
                    with torch.no_grad():
                        evidence = self.encoder.encode_segment(
                            visual[index : index + 1].unsqueeze(1), action[index : index + 1].unsqueeze(1)
                        ).squeeze(1)
                        key, query, value = self.core.project_evidence(evidence)
                    with torch.enable_grad():
                        tokens, candidate, _ = self.core.step_projected_many(
                            key_t=key,
                            query_base_t=query,
                            value_t=value,
                            state_in=state,
                            valid=torch.ones(1, dtype=torch.bool, device=device),
                            create_graph=False,
                        )
                    state, token = self._detach(candidate), tokens[0].detach().clone()
                    if not torch.isfinite(token).all() or any(not torch.isfinite(v).all() for v in state):
                        raise FloatingPointError("online fast-state adaptation produced non-finite values")
            replacement = _Record(request.episode_id, request.consumer_step, state, token, fingerprint)
            update = OnlineMemoryUpdate(self, request.session_id, previous, replacement, False)
            self._pending[request.session_id] = update
            return update

    @staticmethod
    def _detach(state):
        return ContinualTTTFastState(*(value.detach().float().clone() for value in state))

    def commit_many(self, updates: tuple[OnlineMemoryUpdate, ...]) -> None:
        with self._lock:
            sessions = [update.session_id for update in updates]
            if len(set(sessions)) != len(sessions):
                raise ValueError("a prediction batch may contain each session only once")
            for update in updates:
                if (
                    update.owner is not self
                    or self._pending.get(update.session_id) is not update
                    or self._records.get(update.session_id) is not update.previous
                ):
                    raise RuntimeError("online commit requires an exact unconsumed session capability")
            for update in updates:
                self._records[update.session_id] = update.replacement
                del self._pending[update.session_id]

    def commit(self, update: OnlineMemoryUpdate) -> None:
        self.commit_many((update,))

    def abort_many(self, updates: tuple[OnlineMemoryUpdate, ...]) -> None:
        with self._lock:
            for update in updates:
                if update.owner is self and self._pending.get(update.session_id) is update:
                    del self._pending[update.session_id]

    def reset_session(self, session_id: str) -> None:
        with self._lock:
            if session_id in self._pending:
                raise RuntimeError("cannot reset a session during pending action generation")
            self._records.pop(session_id, None)

    def metadata(self) -> dict:
        with self._lock:
            return {
                "evidence_version": EVIDENCE_VERSION,
                "sessions": len(self._records),
                "pending": len(self._pending),
                "max_sessions": self.max_sessions,
                "steps": {key: value.consumer_step for key, value in self._records.items()},
            }
