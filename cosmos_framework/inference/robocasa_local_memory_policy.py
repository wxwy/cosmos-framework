# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Corrected composite completed-evidence adapter for RoboCasa online Local-TTT."""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass
from typing import Any, Callable, Literal

import torch
from torch.distributed.tensor import DTensor

from cosmos_framework.inference.local_memory_online import OnlineLocalMemory, OnlineMemoryRequest, OnlineMemoryUpdate
from cosmos_framework.inference.robocasa_composite_visual import (
    encode_current_composite_visual96,
    prepare_robocasa_composite_frame,
)
from cosmos_framework.inference.robocasa_local_memory_contract import (
    CAMERA_HEIGHT,
    COMPOSITE_WIDTH,
    EVIDENCE_ACTION_DIM,
    EVIDENCE_FORMAT,
    EVIDENCE_VERSION,
    PREPROCESS_PROFILE,
)

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


@dataclass(frozen=True)
class _PreparedUpdate:
    memory: OnlineMemoryUpdate
    visual_replacement: _VisualRecord
    encoded_steps: int


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
    ) -> None:
        if mode not in ("off", "required"):
            raise ValueError("V3 Local Memory mode must be off or required")
        self.service, self.mode, self.decode_image = service, mode, decode_image
        self._visual_records: dict[str, _VisualRecord] = {}
        self._visual_pending: dict[str, _PreparedUpdate] = {}
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

    def _parse(self, req: dict[str, Any]) -> tuple[OnlineMemoryRequest, _VisualRecord, int]:
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
                _, summary = encode_current_composite_visual96(tokenizer, prepared)
                summaries.append(summary.detach().cpu())
            visual = torch.stack(summaries)
            replacement = _VisualRecord(
                episode_id, step, image_size, profile, steps, visual.clone(), action_tensor.clone(), wire_digest
            )
            encoded_steps = len(steps)
        request = OnlineMemoryRequest(session_id, episode_id, step, steps, visual, action_tensor, False)
        return request, replacement, encoded_steps

    def prepare(self, req: dict[str, Any]) -> _PreparedUpdate | None:
        if self.mode == "off":
            if req.get("local_memory") is not None:
                raise ValueError("Local Memory is off but request supplied local_memory evidence")
            return None
        assert self.memory is not None
        with self._lock:
            request, replacement, encoded_steps = self._parse(req)
            if request.session_id in self._visual_pending:
                raise RuntimeError("Local-TTT visual session already has a pending request")
            memory_update = self.memory.prepare(request)
            update = _PreparedUpdate(memory_update, replacement, encoded_steps)
            self._visual_pending[request.session_id] = update
            return update

    def commit(self, update: _PreparedUpdate | None) -> None:
        if update is None:
            return
        assert self.memory is not None
        session_id = update.memory.session_id
        with self._lock:
            if self._visual_pending.get(session_id) is not update:
                raise RuntimeError("Local-TTT visual commit requires exact pending capability")
            self.memory.commit(update.memory)
            self._visual_records[session_id] = update.visual_replacement
            del self._visual_pending[session_id]

    def abort(self, update: _PreparedUpdate | None) -> None:
        if update is None:
            return
        assert self.memory is not None
        with self._lock:
            self.memory.abort(update.memory)
            if self._visual_pending.get(update.memory.session_id) is update:
                del self._visual_pending[update.memory.session_id]

    def reset(self, session_id: str) -> None:
        if self.memory is None:
            return
        with self._lock:
            if session_id in self._visual_pending:
                raise RuntimeError("cannot reset Local-TTT visual session during pending generation")
            self.memory.reset_session(session_id)
            self._visual_records.pop(session_id, None)

    def prefixes(self, update: _PreparedUpdate | None) -> tuple[torch.Tensor | None, ...] | None:
        return None if update is None else (update.memory.token,)

    def status(self, update: _PreparedUpdate | None) -> dict[str, Any] | None:
        if update is None:
            return None
        replacement = update.memory.replacement
        return {
            "session_id": update.memory.session_id,
            "episode_id": replacement.episode_id,
            "consumer_step": replacement.consumer_step,
            "prefix_present": replacement.token is not None,
            "replay": update.memory.replay,
            "encoded_steps": update.encoded_steps,
            **update.memory.telemetry,
        }

    def info(self) -> dict[str, Any]:
        if self.memory is None:
            return {"enabled": False, "mode": "off"}
        return {
            "enabled": True,
            "mode": "required",
            "evidence_version": self.EVIDENCE_VERSION,
            "evidence_format": self.EVIDENCE_FORMAT,
            "visual_evidence": "corrected_composite_single_frame_encode1_v1",
            **self.memory.info(),
        }
