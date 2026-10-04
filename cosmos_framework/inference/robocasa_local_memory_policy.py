"""RoboCasa completed-evidence adapter for V3 online Local-TTT."""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass
from typing import Any, Callable, Literal

import torch
from torch.distributed.tensor import DTensor

from cosmos_framework.inference.local_memory_online import OnlineLocalMemory, OnlineMemoryRequest, OnlineMemoryUpdate
from cosmos_framework.inference.robocasa_causal_evidence import (
    RoboCasaCausalEvidenceStream,
    RoboCasaVisualStreamState,
)

LocalMemoryMode = Literal["off", "required"]


@dataclass(frozen=True)
class _VisualRecord:
    episode_id: str
    consumer_step: int
    stream_state: RoboCasaVisualStreamState
    last_source_steps: tuple[int, ...] = ()
    last_visual_summary: torch.Tensor | None = None
    last_executed_action: torch.Tensor | None = None
    last_visual_digest: str | None = None


@dataclass(frozen=True)
class _PreparedUpdate:
    memory: OnlineMemoryUpdate
    visual_previous: _VisualRecord | None
    visual_replacement: _VisualRecord


class RoboCasaLocalMemoryPolicyAdapter:
    """Bridge completed RoboCasa evidence to the frozen V3 B1 Local-TTT ABI."""

    EVIDENCE_VERSION = "b1_causal_endpoint_visual96_executed_action15_v4"
    EVIDENCE_FORMAT = "robocasa_dual_camera_rgb_raw15_v2"

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
        self.service = service
        self.mode = mode
        self.decode_image = decode_image
        self._visual_records: dict[str, _VisualRecord] = {}
        self._visual_pending: dict[str, _PreparedUpdate] = {}
        self._lock = threading.RLock()
        self.memory: OnlineLocalMemory | None = None
        self.visual_stream: RoboCasaCausalEvidenceStream | None = None
        if mode == "required":
            runtime = getattr(getattr(service.model, "net", None), "local_memory_runtime", None)
            if runtime is None:
                raise ValueError("required Local-TTT checkpoint has no net.local_memory_runtime")
            if runtime.encoder.action_proj.in_features != 15:
                raise ValueError("V3 RoboCasa online Local-TTT requires raw15 evidence")
            if runtime.core.ttt_tbptt_steps != 16 or runtime.core.k_local != 4 or runtime.core.local_dim != 32:
                raise ValueError("V3 RoboCasa Local-TTT runtime contract drift")
            scan = getattr(service.model.net, "scan_local_memory", None)
            if not callable(scan):
                raise ValueError("required V3 Local-TTT needs model-owned scan_local_memory")
            if any(isinstance(parameter, DTensor) for parameter in runtime.parameters()) and not getattr(
                service.model.net, "_local_memory_scan_fsdp_registered", False
            ):
                raise RuntimeError("sharded V3 Local-TTT requires FSDP-registered model-owned scan")
            self.memory = OnlineLocalMemory(
                runtime.encoder,
                runtime.core,
                scan_local_memory=scan,
                max_sessions=max_sessions,
                max_evidence_steps=max_evidence_steps,
            )
            self.visual_stream = RoboCasaCausalEvidenceStream(service.model)

    @staticmethod
    def _validate_frame(frame: torch.Tensor, name: str) -> torch.Tensor:
        value = frame.detach().to("cpu").contiguous()
        if value.ndim != 3 or value.shape[0] != 3 or value.dtype != torch.uint8:
            raise ValueError(f"{name} must decode to uint8 [3,H,W]")
        if value.shape[-2:] != (256, 256):
            raise ValueError(f"{name} must match training B1 camera size 256x256, got {tuple(value.shape[-2:])}")
        return value.clone()

    @staticmethod
    def _visual_digest(
        source_steps: tuple[int, ...],
        left_frames: tuple[torch.Tensor, ...],
        wrist_frames: tuple[torch.Tensor, ...],
    ) -> str:
        if len(source_steps) != len(left_frames) or len(source_steps) != len(wrist_frames):
            raise ValueError("Local-TTT visual evidence lengths must match")
        digest = hashlib.sha256()
        for step, left, wrist in zip(source_steps, left_frames, wrist_frames, strict=True):
            digest.update(f"{step}:".encode("ascii"))
            digest.update(left.numpy().tobytes(order="C"))
            digest.update(wrist.numpy().tobytes(order="C"))
        return digest.hexdigest()

    def _parse(self, req: dict[str, Any]) -> tuple[OnlineMemoryRequest, _VisualRecord | None, _VisualRecord]:
        if self.memory is None or self.visual_stream is None:
            raise RuntimeError("Local-TTT adapter is disabled")
        payload = req.get("local_memory")
        if not isinstance(payload, dict):
            raise ValueError("required Local-TTT request is missing local_memory payload")
        if payload.get("evidence_version") != self.EVIDENCE_VERSION:
            raise ValueError(f"expected evidence_version={self.EVIDENCE_VERSION!r}")
        if payload.get("evidence_format") != self.EVIDENCE_FORMAT:
            raise ValueError(f"expected evidence_format={self.EVIDENCE_FORMAT!r}")
        session_id = payload.get("session_id")
        episode_id = payload.get("episode_id")
        consumer_step = payload.get("consumer_step")
        reset = payload.get("reset", False)
        rows = payload.get("evidence", [])
        if not isinstance(session_id, str) or not session_id or not isinstance(episode_id, str) or not episode_id:
            raise ValueError("Local-TTT session_id/episode_id are required")
        if type(consumer_step) is not int or consumer_step < 0 or type(reset) is not bool or not isinstance(rows, list):
            raise ValueError("invalid Local-TTT consumer_step/reset/evidence")

        previous = self._visual_records.get(session_id)
        if reset or previous is None:
            if consumer_step != 0 or rows:
                raise ValueError("fresh Local-TTT episode must start at consumer_step=0 with no evidence")
            replacement = _VisualRecord(episode_id, 0, self.visual_stream.fresh())
            request = OnlineMemoryRequest(
                session_id,
                episode_id,
                0,
                (),
                torch.empty(0, 96),
                torch.empty(0, 15),
                True,
            )
            return request, previous, replacement

        if previous.episode_id != episode_id:
            raise ValueError("Local-TTT episode changed without reset")

        # Lost-response replay: validate raw bytes/actions against the last committed
        # batch, then reuse its already-materialized visual96. Never advance VAE twice.
        if consumer_step == previous.consumer_step:
            expected_steps = previous.last_source_steps
            if not expected_steps or len(rows) != len(expected_steps):
                raise ValueError("stale Local-TTT request does not match the last committed evidence batch")
            replay_left, replay_wrist, replay_actions = [], [], []
            for expected, row in zip(expected_steps, rows, strict=True):
                if not isinstance(row, dict) or row.get("source_step") != expected:
                    raise ValueError("Local-TTT replay source steps changed")
                replay_left.append(self._validate_frame(self.decode_image(row.get("left_image")), "left_image"))
                replay_wrist.append(self._validate_frame(self.decode_image(row.get("wrist_image")), "wrist_image"))
                action = torch.as_tensor(row.get("executed_action"), dtype=torch.float32).flatten()
                if tuple(action.shape) != (15,) or not torch.isfinite(action).all():
                    raise ValueError("Local-TTT replay executed_action must be finite raw15")
                replay_actions.append(action)
            action_tensor = torch.stack(replay_actions)
            visual_digest = self._visual_digest(expected_steps, tuple(replay_left), tuple(replay_wrist))
            if previous.last_visual_digest is None or visual_digest != previous.last_visual_digest:
                raise ValueError("Local-TTT replay visual evidence changed after commit")
            if previous.last_executed_action is None or not torch.equal(action_tensor, previous.last_executed_action):
                raise ValueError("Local-TTT replay action evidence changed after commit")
            if previous.last_visual_summary is None:
                raise RuntimeError("Local-TTT committed replay is missing cached visual summary")
            request = OnlineMemoryRequest(
                session_id,
                episode_id,
                consumer_step,
                expected_steps,
                previous.last_visual_summary.detach().clone(),
                action_tensor,
                False,
            )
            return request, previous, previous

        expected_steps = tuple(range(previous.consumer_step, consumer_step))
        if len(rows) != len(expected_steps):
            raise ValueError("Local-TTT completed evidence count does not match consumer frontier")
        left, wrist, actions, actual_steps = [], [], [], []
        for expected, row in zip(expected_steps, rows, strict=True):
            if not isinstance(row, dict) or row.get("source_step") != expected:
                raise ValueError("Local-TTT source steps must be contiguous and exactly-once")
            left.append(self._validate_frame(self.decode_image(row.get("left_image")), "left_image"))
            wrist.append(self._validate_frame(self.decode_image(row.get("wrist_image")), "wrist_image"))
            action = torch.as_tensor(row.get("executed_action"), dtype=torch.float32).flatten()
            if tuple(action.shape) != (15,) or not torch.isfinite(action).all():
                raise ValueError("Local-TTT executed_action must be finite raw15")
            actions.append(action)
            actual_steps.append(expected)

        source_steps = tuple(actual_steps)
        left_frames, wrist_frames = tuple(left), tuple(wrist)
        visual, stream_state = self.visual_stream.advance(
            previous.stream_state,
            source_steps,
            left_frames,
            wrist_frames,
        )
        action_tensor = torch.stack(actions) if actions else torch.empty(0, 15)
        replacement = _VisualRecord(
            episode_id,
            consumer_step,
            stream_state,
            source_steps,
            visual.detach().clone(),
            action_tensor.detach().clone(),
            self._visual_digest(source_steps, left_frames, wrist_frames),
        )
        request = OnlineMemoryRequest(
            session_id,
            episode_id,
            consumer_step,
            source_steps,
            visual,
            action_tensor,
            False,
        )
        return request, previous, replacement

    def prepare(self, req: dict[str, Any]) -> _PreparedUpdate | None:
        if self.mode == "off":
            if req.get("local_memory") is not None:
                raise ValueError("Local Memory is off but request supplied local_memory evidence")
            return None
        assert self.memory is not None
        with self._lock:
            request, previous, replacement = self._parse(req)
            if request.session_id in self._visual_pending:
                raise RuntimeError("Local-TTT visual session already has a pending request")
            memory_update = self.memory.prepare(request)
            update = _PreparedUpdate(memory_update, previous, replacement)
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
        if update is None:
            return None
        return (update.memory.token,)

    def status(self, update: _PreparedUpdate | None) -> dict[str, Any] | None:
        if update is None:
            return None
        replacement = update.memory.replacement
        stream = update.visual_replacement.stream_state
        return {
            "session_id": update.memory.session_id,
            "episode_id": replacement.episode_id,
            "consumer_step": replacement.consumer_step,
            "prefix_present": replacement.token is not None,
            "replay": update.memory.replay,
            "visual_endpoint_step": stream.last_endpoint_step,
            "visual_tail_frames": len(stream.pending_left),
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
            "visual_evidence": "b1_dual_camera_causal_endpoint_streaming_v1",
            **self.memory.info(),
        }
