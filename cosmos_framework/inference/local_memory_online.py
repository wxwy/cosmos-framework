"""Causal inference-only Local Memory; completed evidence in, detached prefix out.

No trainer, optimizer, dataset, or checkpoint authority is imported here. Each
session has an independent episode state. Adaptation uses gradients only with
respect to ephemeral fast weights; model parameters and their .grad are untouched.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass, field

import torch

from cosmos_framework.model.generator.mot.local_evidence import (
    CANONICAL_EVIDENCE_FEATURE_CONFIG,
    ContinualTTTFastState,
    ContinualTTTLocalMemoryCore,
    LocalEvidenceEncoder,
    RecurrentLocalMemoryBackend,
)

def evidence_version_for_action_dim(action_dim: int) -> str:
    if not isinstance(action_dim, int) or action_dim <= 0:
        raise ValueError(f"action_dim must be a positive integer, got {action_dim!r}")
    return f"causal_visual96_executed_action{action_dim}_v1"


# Backward-compatible LIBERO/default evidence version. Online memory instances
# derive their actual version from the attached evidence encoder.
EVIDENCE_VERSION = evidence_version_for_action_dim(10)



def _profile_sync(enabled: bool) -> None:
    if enabled and torch.cuda.is_available():
        torch.cuda.synchronize()


def _profile_start(enabled: bool) -> float:
    _profile_sync(enabled)
    return time.perf_counter()


def _profile_elapsed_ms(start: float, enabled: bool) -> float:
    _profile_sync(enabled)
    return (time.perf_counter() - start) * 1000.0


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
    timing_ms: dict[str, float] = field(default_factory=dict)

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
            raise ValueError("online memory requires the canonical visual96/action encoder")
        if max_sessions <= 0 or max_evidence_steps <= 0:
            raise ValueError("online memory bounds must be positive")
        action_proj = getattr(encoder, "action_proj", None)
        action_dim = getattr(action_proj, "in_features", None)
        if not isinstance(action_dim, int) or action_dim <= 0:
            raise ValueError("online memory could not resolve evidence action width from encoder.action_proj")
        self.encoder, self.core = encoder, core
        self.action_dim = int(action_dim)
        self.evidence_version = evidence_version_for_action_dim(self.action_dim)
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
        if tuple(visual.shape) != (n, 96) or tuple(action.shape) != (n, self.action_dim):
            raise ValueError(
                f"online evidence must have shapes [N,96] and [N,{self.action_dim}]"
            )
        if not torch.isfinite(visual).all() or not torch.isfinite(action).all():
            raise ValueError("online evidence must be finite")
        identity = (
            self.evidence_version,
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

    def prepare(
        self,
        request: OnlineMemoryRequest,
        *,
        profile: bool = False,
        update_fast_state: bool = True,
    ) -> OnlineMemoryUpdate:
        """Prepare one transactional online-memory update.

        update_fast_state=False is the frozen-fast / init intervention: real
        evidence still flows through the canonical encoder and query projection,
        but every read uses the checkpoint-learned initial fast weights without
        applying the inner-loss update. Chronology/replay/transaction semantics
        remain identical to normal online memory.
        """
        if not isinstance(update_fast_state, bool):
            raise TypeError("update_fast_state must be bool")
        timing: dict[str, float] = {}
        total_t0 = _profile_start(profile)
        with self._lock, torch.inference_mode(False):
            stage_t0 = _profile_start(profile)
            visual, action, fingerprint = self._validate(request)
            if profile:
                timing["validate_ms"] = _profile_elapsed_ms(stage_t0, profile)
            if request.session_id in self._pending:
                raise RuntimeError("session already has a pending prediction")
            previous = self._records.get(request.session_id)
            if previous is not None and previous.fingerprint == fingerprint:
                if profile:
                    timing["prepare_total_ms"] = _profile_elapsed_ms(total_t0, profile)
                update = OnlineMemoryUpdate(self, request.session_id, previous, previous, True, timing)
                self._pending[request.session_id] = update
                return update
            fresh = previous is None or request.reset
            if fresh:
                if request.consumer_step != 0 or request.source_steps:
                    raise ValueError("a fresh/reset episode must begin at consumer step0 without evidence")
                new_pending = sum(u.previous is None for u in self._pending.values())
                if previous is None and len(self._records) + new_pending >= self.max_sessions:
                    raise RuntimeError("session limit reached; explicitly close an episode before admitting another")
                stage_t0 = _profile_start(profile)
                with torch.no_grad():
                    device = next(self.encoder.parameters()).device
                    state = self._detach(self.core.initial_state(1, device=device))
                if profile:
                    timing["state_init_ms"] = _profile_elapsed_ms(stage_t0, profile)
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
                encode_ms = project_ms = update_read_ms = detach_ms = 0.0
                for index in range(len(expected)):
                    stage_t0 = _profile_start(profile)
                    with torch.no_grad():
                        evidence = self.encoder.encode_segment(
                            visual[index : index + 1].unsqueeze(1), action[index : index + 1].unsqueeze(1)
                        ).squeeze(1)
                    if profile:
                        encode_ms += _profile_elapsed_ms(stage_t0, profile)
                    stage_t0 = _profile_start(profile)
                    with torch.no_grad():
                        key, query, value = self.core.project_evidence(evidence)
                    if profile:
                        project_ms += _profile_elapsed_ms(stage_t0, profile)
                    stage_t0 = _profile_start(profile)
                    if update_fast_state:
                        with torch.enable_grad():
                            tokens, candidate, _ = self.core.step_projected_many(
                                key_t=key,
                                query_base_t=query,
                                value_t=value,
                                state_in=state,
                                valid=torch.ones(1, dtype=torch.bool, device=device),
                                create_graph=False,
                            )
                        next_state = self._detach(candidate)
                    else:
                        # Init/frozen-fast intervention: current evidence still
                        # determines the read query, but the fast weights remain
                        # exactly the checkpoint-learned W0 for the whole episode.
                        with torch.no_grad():
                            queries = self.core.project_queries(query)
                            tokens = self.core.read_many(queries, state)
                        next_state = state
                    if profile:
                        update_read_ms += _profile_elapsed_ms(stage_t0, profile)
                    stage_t0 = _profile_start(profile)
                    state, token = next_state, tokens[0].detach().clone()
                    if profile:
                        detach_ms += _profile_elapsed_ms(stage_t0, profile)
                    if not torch.isfinite(token).all() or any(not torch.isfinite(v).all() for v in state):
                        raise FloatingPointError("online fast-state read/update produced non-finite values")
                if profile:
                    timing["evidence_encode_ms"] = encode_ms
                    timing["kqv_project_ms"] = project_ms
                    timing["inner_update_read_ms"] = update_read_ms
                    timing["state_detach_ms"] = detach_ms
                    timing["adapted_steps"] = float(len(expected) if update_fast_state else 0)
                    timing["frozen_read_steps"] = float(0 if update_fast_state else len(expected))
            replacement = _Record(request.episode_id, request.consumer_step, state, token, fingerprint)
            if profile:
                timing["prepare_total_ms"] = _profile_elapsed_ms(total_t0, profile)
            update = OnlineMemoryUpdate(self, request.session_id, previous, replacement, False, timing)
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
                "evidence_version": self.evidence_version,
                "action_dim": self.action_dim,
                "sessions": len(self._records),
                "pending": len(self._pending),
                "max_sessions": self.max_sessions,
                "memory_kind": "ttt_fast_weight",
                "steps": {key: value.consumer_step for key, value in self._records.items()},
            }


@dataclass(frozen=True)
class _RecentHistoryRecord:
    episode_id: str
    consumer_step: int
    source_steps: tuple[int, ...]
    visual_summary: torch.Tensor
    executed_action: torch.Tensor
    token: torch.Tensor | None
    fingerprint: str


@dataclass(frozen=True)
class OnlineRecentHistoryUpdate:
    owner: "OnlineRecentHistoryMemory"
    session_id: str
    previous: _RecentHistoryRecord | None
    replacement: _RecentHistoryRecord
    replay: bool
    timing_ms: dict[str, float] = field(default_factory=dict)

    @property
    def token(self) -> torch.Tensor | None:
        token = self.replacement.token
        return None if token is None else token.detach().clone()


class OnlineRecentHistoryMemory:
    """Transactional bounded-history control with no persistent recurrent state.

    Only the most recent history_horizon canonical evidence rows are retained.
    Every policy query recomputes the token by replaying those rows from a zero GRU
    state (state=None); no hidden state is stored in the session record.
    """

    def __init__(
        self,
        encoder: LocalEvidenceEncoder,
        recurrent_backend: RecurrentLocalMemoryBackend,
        *,
        history_horizon: int,
        max_sessions: int = 64,
        max_evidence_steps: int = 256,
    ):
        if encoder.feature_config != CANONICAL_EVIDENCE_FEATURE_CONFIG:
            raise ValueError("recent-history control requires canonical visual96/action evidence")
        if recurrent_backend is None:
            raise ValueError("recent-history control requires a recurrent replay backend")
        if encoder.evidence_dim != recurrent_backend.evidence_dim:
            raise ValueError("recent-history encoder/backend evidence dimensions must match")
        if history_horizon <= 0 or max_sessions <= 0 or max_evidence_steps <= 0:
            raise ValueError("recent-history bounds must be positive")
        action_proj = getattr(encoder, "action_proj", None)
        action_dim = getattr(action_proj, "in_features", None)
        if not isinstance(action_dim, int) or action_dim <= 0:
            raise ValueError("recent-history control could not resolve action width from encoder.action_proj")
        self.encoder, self.recurrent_backend = encoder, recurrent_backend
        self.action_dim = int(action_dim)
        self.evidence_version = evidence_version_for_action_dim(self.action_dim)
        self.history_horizon = int(history_horizon)
        self.max_sessions, self.max_evidence_steps = max_sessions, max_evidence_steps
        self._records: dict[str, _RecentHistoryRecord] = {}
        self._pending: dict[str, OnlineRecentHistoryUpdate] = {}
        self._lock = threading.RLock()

    def _validate(self, request: OnlineMemoryRequest):
        if not isinstance(request, OnlineMemoryRequest):
            raise TypeError("recent-history control requires a typed request")
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
        if tuple(visual.shape) != (n, 96) or tuple(action.shape) != (n, self.action_dim):
            raise ValueError(
                f"online evidence must have shapes [N,96] and [N,{self.action_dim}]"
            )
        if not torch.isfinite(visual).all() or not torch.isfinite(action).all():
            raise ValueError("online evidence must be finite")
        identity = (
            self.evidence_version,
            "recent_history",
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

    def _replay_from_zero(
        self, visual: torch.Tensor, action: torch.Tensor, *, profile: bool = False
    ) -> tuple[torch.Tensor | None, dict[str, float]]:
        timing: dict[str, float] = {}
        if visual.shape[0] == 0:
            return None, timing
        device = next(self.encoder.parameters()).device
        stage_t0 = _profile_start(profile)
        with torch.no_grad():
            evidence = self.encoder.encode_segment(visual.to(device).unsqueeze(0), action.to(device).unsqueeze(0))
        if profile:
            timing["evidence_encode_ms"] = _profile_elapsed_ms(stage_t0, profile)
        mask = torch.ones(1, evidence.shape[1], dtype=torch.bool, device=device)
        stage_t0 = _profile_start(profile)
        with torch.no_grad():
            tokens, _, present = self.recurrent_backend.replay(evidence, mask, state=None)
            token = tokens[0].detach().clone() if bool(present[0]) else None
        if profile:
            timing["gru_replay_ms"] = _profile_elapsed_ms(stage_t0, profile)
            timing["replayed_steps"] = float(evidence.shape[1])
        if token is not None and not torch.isfinite(token).all():
            raise FloatingPointError("recent-history replay produced non-finite Local token")
        return token, timing

    def prepare(self, request: OnlineMemoryRequest, *, profile: bool = False) -> OnlineRecentHistoryUpdate:
        timing: dict[str, float] = {}
        total_t0 = _profile_start(profile)
        with self._lock:
            stage_t0 = _profile_start(profile)
            visual, action, fingerprint = self._validate(request)
            if profile:
                timing["validate_ms"] = _profile_elapsed_ms(stage_t0, profile)
            if request.session_id in self._pending:
                raise RuntimeError("session already has a pending prediction")
            previous = self._records.get(request.session_id)
            if previous is not None and previous.fingerprint == fingerprint:
                if profile:
                    timing["prepare_total_ms"] = _profile_elapsed_ms(total_t0, profile)
                update = OnlineRecentHistoryUpdate(self, request.session_id, previous, previous, True, timing)
                self._pending[request.session_id] = update
                return update

            fresh = previous is None or request.reset
            if fresh:
                if request.consumer_step != 0 or request.source_steps:
                    raise ValueError("a fresh/reset episode must begin at consumer step0 without evidence")
                new_pending = sum(u.previous is None for u in self._pending.values())
                if previous is None and len(self._records) + new_pending >= self.max_sessions:
                    raise RuntimeError("session limit reached; explicitly close an episode before admitting another")
                retained_steps: tuple[int, ...] = ()
                retained_visual = torch.empty(0, 96)
                retained_action = torch.empty(0, self.action_dim)
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
                stage_t0 = _profile_start(profile)
                retained_steps = previous.source_steps + request.source_steps
                retained_visual = torch.cat((previous.visual_summary, visual), dim=0)
                retained_action = torch.cat((previous.executed_action, action), dim=0)
                if len(retained_steps) > self.history_horizon:
                    retained_steps = retained_steps[-self.history_horizon :]
                    retained_visual = retained_visual[-self.history_horizon :]
                    retained_action = retained_action[-self.history_horizon :]
                if profile:
                    timing["window_merge_ms"] = _profile_elapsed_ms(stage_t0, profile)
                token, replay_timing = self._replay_from_zero(retained_visual, retained_action, profile=profile)
                timing.update(replay_timing)

            replacement = _RecentHistoryRecord(
                request.episode_id,
                request.consumer_step,
                retained_steps,
                retained_visual.detach().cpu().clone(),
                retained_action.detach().cpu().clone(),
                token,
                fingerprint,
            )
            if profile:
                timing["prepare_total_ms"] = _profile_elapsed_ms(total_t0, profile)
            update = OnlineRecentHistoryUpdate(self, request.session_id, previous, replacement, False, timing)
            self._pending[request.session_id] = update
            return update

    def commit_many(self, updates: tuple[OnlineRecentHistoryUpdate, ...]) -> None:
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

    def commit(self, update: OnlineRecentHistoryUpdate) -> None:
        self.commit_many((update,))

    def abort_many(self, updates: tuple[OnlineRecentHistoryUpdate, ...]) -> None:
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
                "memory_kind": "bounded_recent_history",
                "history_horizon": self.history_horizon,
                "sessions": len(self._records),
                "pending": len(self._pending),
                "max_sessions": self.max_sessions,
                "steps": {key: value.consumer_step for key, value in self._records.items()},
                "retained_source_steps": {key: list(value.source_steps) for key, value in self._records.items()},
            }
