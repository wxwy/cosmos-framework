# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Corrected composite completed-evidence adapter for RoboCasa online Local-TTT."""

from __future__ import annotations

import hashlib
import struct
import threading
from contextlib import nullcontext
from dataclasses import dataclass, replace
from typing import Any, Callable, Literal

import torch
from torch.distributed.tensor import DTensor

from cosmos_framework.inference.local_memory_online import OnlineLocalMemory, OnlineMemoryRequest, OnlineMemoryUpdate
from cosmos_framework.inference.local_memory_state import CachedMemoryResponse, MemorySnapshot
from cosmos_framework.inference.robocasa_composite_visual import (
    encode_current_composite_visual96,
    prepare_robocasa_composite_frame,
)
from cosmos_framework.inference.robocasa_feature_evidence import (
    FEATURE_EVIDENCE_FORMAT,
    FEATURE_EVIDENCE_VERSION,
    SPARSE_EVIDENCE_FORMAT,
    SPARSE_EVIDENCE_VERSION,
    FeatureEvidenceOwner,
    PolicyBoundary,
    feature_evidence_digest,
)
from cosmos_framework.inference.robocasa_local_memory_contract import (
    CAMERA_HEIGHT,
    COMPOSITE_WIDTH,
    EVIDENCE_ACTION_DIM,
    EVIDENCE_FORMAT,
    EVIDENCE_VERSION,
    PREPROCESS_PROFILE,
)
from cosmos_framework.inference.robocasa_vae_protocol import canonical_json, strict_json, validate_lease

LocalMemoryMode = Literal["off", "required"]


@dataclass(frozen=True)
class _VisualRecord:
    episode_id: str
    consumer_step: int
    image_size: int
    preprocess_profile: str
    last_source_steps: tuple[int, ...] = ()
    last_visual_summary: torch.Tensor | None = None
    last_executed_action: torch.Tensor | None = None
    last_wire_digest: str | None = None
    feature_lease_json: bytes | None = None
    boundary: PolicyBoundary | None = None


@dataclass(frozen=True)
class _PreparedUpdate:
    memory: OnlineMemoryUpdate
    visual_replacement: _VisualRecord
    encoded_steps: int
    feature_profile: bool = False
    boundary_context_json: bytes | None = None


class RoboCasaLocalMemoryPolicyAdapter:
    """完成动作的 composite 经独立 Encode1 变为 Local visual96。"""

    EVIDENCE_VERSION = EVIDENCE_VERSION
    EVIDENCE_FORMAT = EVIDENCE_FORMAT

    def __init__(
        self,
        service: Any,
        *,
        mode: LocalMemoryMode,
        decode_image: Callable[[str], torch.Tensor],
        max_sessions: int = 1,
        max_evidence_steps: int = 256,
        feature_evidence_owner: FeatureEvidenceOwner | None = None,
    ) -> None:
        if mode not in ("off", "required"):
            raise ValueError("V3 Local Memory mode must be off or required")
        self.service, self.mode, self.decode_image = service, mode, decode_image
        self._visual_pending: dict[str, _PreparedUpdate] = {}
        self._boundary_pending: dict[str, PolicyBoundary] = {}
        self.feature_evidence_owner = feature_evidence_owner
        if feature_evidence_owner is not None and mode != "required":
            raise ValueError("signed Feature profile requires Local-TTT required mode")
        self._lock = threading.RLock()
        self.memory: OnlineLocalMemory | None = None
        if mode == "required":
            model = service.model
            runtime = getattr(getattr(model, "net", None), "local_memory_runtime", None)
            if runtime is None:
                raise ValueError("required Local-TTT checkpoint has no net.local_memory_runtime")
            if runtime.encoder.action_proj.in_features != EVIDENCE_ACTION_DIM:
                raise ValueError("V3 RoboCasa online Local-TTT requires raw15 evidence")
            scan = getattr(model.net, "scan_local_memory", None)
            if not callable(scan):
                raise ValueError("required V3 Local-TTT needs model-owned scan_local_memory")
            if any(isinstance(parameter, DTensor) for parameter in runtime.parameters()) and not getattr(
                model.net, "_local_memory_scan_fsdp_registered", False
            ):
                raise RuntimeError("sharded V3 Local-TTT requires FSDP-registered model-owned scan")
            if not callable(getattr(getattr(model, "tokenizer_vision_gen", None), "encode", None)):
                raise ValueError("required Corrected Local-TTT needs loaded Wan tokenizer.encode")
            self.memory = OnlineLocalMemory(
                runtime.encoder,
                runtime.core,
                scan_local_memory=scan,
                max_sessions=max_sessions,
                max_evidence_steps=max_evidence_steps,
            )
            self.memory.paired = True
            self._lock = self.memory.store.lock
            if feature_evidence_owner is not None:
                feature_evidence_owner.bind_boundary_lookup(self.committed_boundary)

    @property
    def snapshot(self) -> MemorySnapshot:
        if self.memory is None:
            raise RuntimeError("Local-TTT is off")
        return self.memory.store.snapshot

    @property
    def _visual_records(self):
        return self.snapshot.visual

    def _row(self, row: Any, expected: int) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(row, dict) or set(row) != {"source_step", "composite_image", "executed_action"}:
            raise ValueError("Corrected evidence requires one composite_image and canonical executed_action")
        if row["source_step"] != expected:
            raise ValueError("Local-TTT source steps must be contiguous and exactly-once")
        frame = self.decode_image(row["composite_image"])
        if (
            not isinstance(frame, torch.Tensor)
            or frame.dtype != torch.uint8
            or tuple(frame.shape) != (3, CAMERA_HEIGHT, COMPOSITE_WIDTH)
        ):
            raise ValueError("Corrected composite_image must be uint8 [3,256,512]")
        action = torch.as_tensor(row["executed_action"], dtype=torch.float32).flatten()
        if tuple(action.shape) != (EVIDENCE_ACTION_DIM,) or not bool(torch.isfinite(action).all()):
            raise ValueError("Local-TTT executed_action must be finite raw15")
        return frame.contiguous(), action

    @staticmethod
    def _digest(steps: tuple[int, ...], frames: list[torch.Tensor], actions: torch.Tensor) -> str:
        digest = hashlib.sha256(repr(steps).encode())
        for frame in frames:
            digest.update(frame.cpu().numpy().tobytes())
        digest.update(actions.cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def _parse_legacy(self, req: dict[str, Any]) -> tuple[OnlineMemoryRequest, _VisualRecord, int]:
        assert self.memory is not None
        payload = req.get("local_memory")
        if not isinstance(payload, dict):
            raise ValueError("required Local-TTT request is missing local_memory payload")
        if (
            payload.get("evidence_version") != self.EVIDENCE_VERSION
            or payload.get("evidence_format") != self.EVIDENCE_FORMAT
        ):
            raise ValueError("Corrected evidence version/format required; historical B1 wire rejected")
        session_id, episode_id = payload.get("session_id"), payload.get("episode_id")
        step, reset, rows = payload.get("consumer_step"), payload.get("reset", False), payload.get("evidence", [])
        image_size, profile = payload.get("image_size"), payload.get("preprocess_profile")
        if not isinstance(session_id, str) or not session_id or not isinstance(episode_id, str) or not episode_id:
            raise ValueError("Local-TTT session_id/episode_id are required")
        if type(step) is not int or step < 0 or type(reset) is not bool or not isinstance(rows, list):
            raise ValueError("invalid Local-TTT consumer_step/reset/evidence")
        if type(image_size) is not int or image_size <= 0 or image_size != req.get("image_size"):
            raise ValueError("Local-TTT image_size must match outer policy request")
        if profile != PREPROCESS_PROFILE:
            raise ValueError("Corrected preprocess_profile 不匹配")
        previous = self._visual_records.get(session_id)
        if previous is not None and previous.feature_lease_json is not None:
            raise ValueError("cannot silently replace C Feature profile with legacy server_vae")
        if reset or previous is None:
            if step != 0 or rows:
                raise ValueError("fresh Local-TTT episode must start at consumer_step=0 with no evidence")
            replacement = _VisualRecord(episode_id, 0, image_size, profile)
            return (
                OnlineMemoryRequest(session_id, episode_id, 0, (), torch.empty(0, 96), torch.empty(0, 15), True),
                replacement,
                0,
            )
        if (previous.episode_id, previous.image_size, previous.preprocess_profile) != (episode_id, image_size, profile):
            raise ValueError("Local-TTT episode/image_size/preprocess_profile changed without reset")
        replay = step == previous.consumer_step
        steps = previous.last_source_steps if replay else tuple(range(previous.consumer_step, step))
        if len(rows) != len(steps) or (replay and not steps):
            raise ValueError("Local-TTT evidence count does not match consumer frontier")
        frames, actions = [], []
        for source_step, row in zip(steps, rows, strict=True):
            frame, action = self._row(row, source_step)
            frames.append(frame)
            actions.append(action)
        action_tensor = torch.stack(actions)
        wire_digest = self._digest(steps, frames, action_tensor)
        if replay:
            if wire_digest != previous.last_wire_digest or previous.last_visual_summary is None:
                raise ValueError("Local-TTT replay bytes/action changed after commit")
            visual = previous.last_visual_summary.detach().clone()
            replacement = previous
            encoded_steps = 0
        else:
            tokenizer = self.service.model.tokenizer_vision_gen
            summaries = []
            for frame in frames:
                prepared = prepare_robocasa_composite_frame(frame, image_size)
                from cosmos_framework.inference.robocasa_feature_trace import get_trace, span

                trace = get_trace()
                with (
                    span("history_vae", producer="policy"),
                    trace.cuda_span("history_vae", producer="policy") if trace else nullcontext(),
                ):
                    _, summary = encode_current_composite_visual96(tokenizer, prepared)
                summaries.append(summary.detach().cpu())
            visual = torch.stack(summaries)
            replacement = _VisualRecord(
                episode_id, step, image_size, profile, steps, visual.clone(), action_tensor.clone(), wire_digest
            )
            encoded_steps = len(steps)
        request = OnlineMemoryRequest(session_id, episode_id, step, steps, visual, action_tensor, False)
        return request, replacement, encoded_steps

    def _parse_features(self, req, *, lease, deadline_ns, clock_domain):
        owner = self.feature_evidence_owner
        if owner is None or lease is None or deadline_ns is None or clock_domain is None:
            raise ValueError("Feature profile requires authenticated native Lease, Owner and original budget")
        payload = req.get("local_memory")
        fields = {
            "session_id",
            "episode_id",
            "consumer_step",
            "reset",
            "image_size",
            "preprocess_profile",
            "evidence_version",
            "evidence_format",
            "evidence",
        }
        if owner.history_k != 16:
            fields.add("local_ttt_history_k")
        if type(payload) is not dict or payload.keys() != fields:
            raise ValueError("Feature profile requires exact opt-in evidence fields")
        if payload["evidence_version"] != (
            FEATURE_EVIDENCE_VERSION if owner.history_k == 16 else SPARSE_EVIDENCE_VERSION
        ) or payload["evidence_format"] != (
            FEATURE_EVIDENCE_FORMAT if owner.history_k == 16 else SPARSE_EVIDENCE_FORMAT
        ):
            raise ValueError("unsupported Feature evidence version/format")
        if owner.history_k != 16 and (
            type(payload["local_ttt_history_k"]) is not int or payload["local_ttt_history_k"] != owner.history_k
        ):
            raise ValueError("Feature history K differs from frozen Policy Owner")
        if payload["image_size"] != 256 or type(payload["image_size"]) is not int or req.get("image_size") != 256:
            raise ValueError("Stage3A attested Feature preprocessing requires image_size256")
        if payload["preprocess_profile"] != PREPROCESS_PROFILE or type(payload["reset"]) is not bool:
            raise ValueError("Feature preprocess/reset mismatch")
        public = validate_lease(lease)
        owner.validate_request(public, deadline_ns, clock_domain)
        lease_json = canonical_json(public)
        session_id, episode_id = payload["session_id"], payload["episode_id"]
        if (session_id, episode_id) != (public["session_id"], public["episode_id"]):
            raise ValueError("Feature payload does not match authenticated native Lease")
        previous = self._visual_records.get(session_id)
        step, reset, rows = payload["consumer_step"], payload["reset"], payload["evidence"]
        if type(step) is not int or step < 0 or type(rows) is not list or len(rows) > 16:
            raise ValueError("invalid bounded Feature chronology")
        if previous is not None and previous.feature_lease_json != lease_json:
            raise ValueError("Feature Lease/profile changed without a new native Session")
        if previous is None or reset:
            if step != 0 or rows:
                raise ValueError("fresh Feature episode must start at0 without history")
            batch = owner.admit(
                payload, lease=public, deadline_ns=deadline_ns, clock_domain=clock_domain, previous_step=0
            )
            request = OnlineMemoryRequest(
                session_id,
                episode_id,
                0,
                (),
                torch.empty(0, 96),
                torch.empty(0, 15),
                True,
                True,
                history_k=owner.history_k,
            )
            return request, _VisualRecord(episode_id, 0, 256, PREPROCESS_PROFILE, feature_lease_json=lease_json), 0
        if previous.episode_id != episode_id or previous.image_size != 256:
            raise ValueError("Feature episode/preprocessing changed")
        if step == previous.consumer_step:
            if not previous.last_source_steps or feature_evidence_digest(rows) != previous.last_wire_digest:
                raise ValueError("Feature replay bytes changed after commit")
            visual = previous.last_visual_summary.detach().clone()
            actions = previous.last_executed_action.detach().clone()
            request = OnlineMemoryRequest(
                session_id,
                episode_id,
                step,
                previous.last_source_steps,
                visual,
                actions,
                False,
                True,
                history_k=owner.history_k,
            )
            return request, previous, 0
        batch = owner.admit(
            payload,
            lease=public,
            deadline_ns=deadline_ns,
            clock_domain=clock_domain,
            previous_step=previous.consumer_step,
        )
        visual = torch.tensor([struct.unpack("<96f", raw) for raw in batch.visual96], dtype=torch.float32).reshape(
            -1, 96
        )
        actions = torch.tensor([struct.unpack("<15f", raw) for raw in batch.raw15], dtype=torch.float32).reshape(-1, 15)
        request = OnlineMemoryRequest(
            session_id, episode_id, step, batch.source_steps, visual, actions, False, True, history_k=owner.history_k
        )
        replacement = _VisualRecord(
            episode_id,
            step,
            256,
            PREPROCESS_PROFILE,
            batch.source_steps,
            visual.clone(),
            actions.clone(),
            batch.wire_digest,
            lease_json,
        )
        return request, replacement, 0

    def _parse(self, req, *, lease=None, deadline_ns=None, clock_domain=None):
        payload = req.get("local_memory")
        if self.feature_evidence_owner is not None or (
            type(payload) is dict
            and (
                payload.get("evidence_version") == FEATURE_EVIDENCE_VERSION
                or payload.get("evidence_format") == FEATURE_EVIDENCE_FORMAT
            )
        ):
            return self._parse_features(req, lease=lease, deadline_ns=deadline_ns, clock_domain=clock_domain)
        return self._parse_legacy(req)

    def prepare(
        self, req: dict[str, Any], *, lease=None, deadline_ns=None, clock_domain=None, request_key=None
    ) -> _PreparedUpdate | None:
        if self.mode == "off":
            if req.get("local_memory") is not None:
                raise ValueError("Local Memory is off but request supplied local_memory evidence")
            return None
        assert self.memory is not None
        with self._lock:
            from cosmos_framework.inference.robocasa_feature_trace import get_trace, span

            trace = get_trace()
            with span(
                "feature_verify" if self.feature_evidence_owner else "history_payload_prepare", producer="policy"
            ):
                request, replacement, encoded_steps = self._parse(
                    req, lease=lease, deadline_ns=deadline_ns, clock_domain=clock_domain
                )
            if request.session_id in self._visual_pending:
                raise RuntimeError("Local-TTT visual session already has a pending request")
            feature_profile = replacement.feature_lease_json is not None
            context_json = None
            if feature_profile:
                if type(request_key) is not str or not request_key or len(request_key) > 256:
                    raise ValueError("C boundary capture requires authoritative native request_key")
                context_json = canonical_json(
                    {
                        "lease": validate_lease(lease),
                        "identity": self.feature_evidence_owner.identity,
                        "request_key": request_key,
                        "chunk_id": request_key,
                        "source_step": request.consumer_step,
                        "deadline_ns": int(deadline_ns),
                        "clock_domain": clock_domain,
                    }
                )
            with (
                span("fast_weight_update", producer="policy"),
                trace.cuda_span("fast_weight_update", producer="policy") if trace else nullcontext(),
            ):
                memory_update = self.memory.prepare(request)
            update = _PreparedUpdate(memory_update, replacement, encoded_steps, feature_profile, context_json)
            self._visual_pending[request.session_id] = update
            return update

    def boundary_context(self, update):
        if update is None or not update.feature_profile or update.boundary_context_json is None:
            return None
        with self._lock:
            if self._visual_pending.get(update.memory.session_id) is not update:
                raise RuntimeError("capture context requires exact pending Local transaction")
            return strict_json(update.boundary_context_json)

    def committed_boundary(self, session_id, source_step):
        """Lock-free read of ONE immutable published snapshot; safe under the Owner lock."""
        snapshot = self.snapshot
        record = snapshot.visual.get(session_id)
        boundary = None if record is None else record.boundary
        return boundary if boundary is not None and boundary.header["source_step"] == source_step else None

    def stage_boundary(self, update, boundary):
        if update is None or not update.feature_profile or self.feature_evidence_owner is None:
            raise ValueError("Policy boundary capture requires a prepared C Feature transaction")
        with self._lock:
            session_id = update.memory.session_id
            if self._visual_pending.get(session_id) is not update:
                raise RuntimeError("boundary capture requires exact pending Local transaction")
            candidate = self.feature_evidence_owner.validate_boundary_candidate(boundary)
            header = candidate.header
            context = strict_json(update.boundary_context_json)
            if (
                canonical_json(header["lease"]) != update.visual_replacement.feature_lease_json
                or header["source_step"] != update.memory.replacement.consumer_step
                or header["issued_chunk_id"] != context["chunk_id"]
                or header["policy_request_key"] != context["request_key"]
                or int(header["deadline_ns"]) != context["deadline_ns"]
                or header["clock_domain"] != context["clock_domain"]
            ):
                raise ValueError("captured current frame does not match pending Local candidate")
            previous = self._boundary_pending.get(session_id)
            if previous is not None and previous != candidate:
                raise ValueError("changed-bytes boundary capture for pending Local request")
            self._boundary_pending[session_id] = candidate

    def commit(self, update: _PreparedUpdate | None, response: CachedMemoryResponse | None = None) -> None:
        if update is None:
            return
        self.commit_many(((update, response),))

    def commit_many(self, members) -> None:
        """Prevalidate/allocate every participant, then publish ONE snapshot."""
        assert self.memory is not None
        members = tuple(members)
        if not members:
            raise ValueError("nonempty Local batch commit required")
        with self._lock:
            sessions = [update.memory.session_id for update, _ in members]
            if len(set(sessions)) != len(sessions):
                raise ValueError("duplicate Session in atomic batch")
            for update, _ in members:
                if self._visual_pending.get(update.memory.session_id) is not update:
                    raise RuntimeError("Local-TTT visual commit requires exact pending capability")
                self.memory.validate_commit(update.memory)
            snapshot = self.snapshot
            fast, visual, responses = dict(snapshot.fast), dict(snapshot.visual), dict(snapshot.responses)
            boundaries = []
            for update, response in members:
                session_id = update.memory.session_id
                record = update.visual_replacement
                if update.feature_profile:
                    boundary = self._boundary_pending.get(session_id)
                    if boundary is None and update.memory.replay:
                        boundary = record.boundary
                    if boundary is None:
                        raise RuntimeError("C Response/FastState cannot commit without its current pre-crop boundary")
                    boundaries.append(boundary)
                    record = replace(record, boundary=boundary)
                fast[session_id], visual[session_id] = update.memory.replacement, record
                responses.pop(session_id, None)
                if response is not None:
                    responses[session_id] = response
            guard = nullcontext() if not boundaries else self.feature_evidence_owner.publication_guard(boundaries)
            with guard:
                replacement = MemorySnapshot.build(fast, visual, responses)
                self.memory.store.snapshot = replacement
                for session_id in sessions:
                    del self.memory._pending[session_id]
                    del self._visual_pending[session_id]
                    self._boundary_pending.pop(session_id, None)

    def abort(self, update: _PreparedUpdate | None) -> None:
        if update is None:
            return
        assert self.memory is not None
        with self._lock:
            self.memory.abort(update.memory)
            if self._visual_pending.get(update.memory.session_id) is update:
                del self._visual_pending[update.memory.session_id]
                self._boundary_pending.pop(update.memory.session_id, None)

    def reset(self, session_id: str) -> None:
        if self.memory is None:
            return
        with self._lock:
            if session_id in self._visual_pending or session_id in self.memory._pending:
                raise RuntimeError("cannot reset Local-TTT visual session during pending generation")
            self.memory.store.snapshot = self.snapshot.without(session_id)
            self._boundary_pending.pop(session_id, None)

    def prefixes(self, update: _PreparedUpdate | None) -> tuple[torch.Tensor | None, ...] | None:
        return None if update is None else (update.memory.token,)

    def status(self, update: _PreparedUpdate | None) -> dict[str, Any] | None:
        if update is None:
            return None
        replacement = update.memory.replacement
        result = {
            "session_id": update.memory.session_id,
            "episode_id": replacement.episode_id,
            "consumer_step": replacement.consumer_step,
            "prefix_present": replacement.token is not None,
            "replay": update.memory.replay,
            "encoded_steps": update.encoded_steps,
            **update.memory.telemetry,
        }
        if update.feature_profile:
            result["evidence_profile"] = FEATURE_EVIDENCE_FORMAT
            if self.feature_evidence_owner.history_k != 16:
                selected = update.visual_replacement.last_source_steps
                start = selected[0] if selected else replacement.consumer_step
                result.update(
                    evidence_profile=SPARSE_EVIDENCE_FORMAT,
                    local_ttt_history_k=self.feature_evidence_owner.history_k,
                    selected_source_steps=list(selected),
                    skipped_source_steps=[s for s in range(start, replacement.consumer_step) if s not in selected],
                    verified_worker_features=sum(s % 16 != 0 for s in selected),
                )
            # This metadata is serialized into a PRIVATE native Response before
            # commit_many publishes that same Response/FastState/boundary. Owner
            # lookups still read only the immutable published snapshot.
            with self._lock:
                boundary = None
                if self._visual_pending.get(update.memory.session_id) is update:
                    boundary = self._boundary_pending.get(update.memory.session_id)
                if boundary is None:
                    boundary = self.committed_boundary(update.memory.session_id, replacement.consumer_step)
            result["committed_boundary"] = None if boundary is None else boundary.record()
        return result

    def info(self) -> dict[str, Any]:
        if self.memory is None:
            return {"enabled": False, "mode": "off"}
        return {
            "enabled": True,
            "mode": "required",
            "evidence_version": self.EVIDENCE_VERSION,
            "evidence_format": self.EVIDENCE_FORMAT,
            "visual_evidence": "corrected_composite_single_frame_encode1_v1",
            "feature_evidence_profile": None
            if self.feature_evidence_owner is None
            else (FEATURE_EVIDENCE_FORMAT if self.feature_evidence_owner.history_k == 16 else SPARSE_EVIDENCE_FORMAT),
            **(
                {"local_ttt_history_k": self.feature_evidence_owner.history_k}
                if self.feature_evidence_owner is not None and self.feature_evidence_owner.history_k != 16
                else {}
            ),
            **self.memory.info(),
        }
