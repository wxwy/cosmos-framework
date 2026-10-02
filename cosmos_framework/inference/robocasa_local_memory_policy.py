"""RoboCasa completed-evidence adapter for V3 online Local-TTT."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Callable, Literal

import torch

from cosmos_framework.inference.local_memory_online import OnlineLocalMemory, OnlineMemoryRequest, OnlineMemoryUpdate
from cosmos_framework.model.generator.mot.robocasa_latent_evidence import latent_to_visual96

LocalMemoryMode = Literal["off", "required"]


@dataclass(frozen=True)
class _VisualRecord:
    episode_id: str
    consumer_step: int
    left_frames: tuple[torch.Tensor, ...]
    wrist_frames: tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class _PreparedUpdate:
    memory: OnlineMemoryUpdate
    visual_previous: _VisualRecord | None
    visual_replacement: _VisualRecord


class RoboCasaLocalMemoryPolicyAdapter:
    """Bridge HTTP completed evidence to canonical V3 Local-TTT prefixes."""

    EVIDENCE_VERSION = "causal_visual96_executed_action15_v3"
    EVIDENCE_FORMAT = "robocasa_left_wrist_raw15_v1"

    def __init__(
        self,
        service: Any,
        *,
        mode: LocalMemoryMode,
        decode_image: Callable[[str], torch.Tensor],
        max_sessions: int = 64,
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
        if mode == "required":
            runtime = getattr(getattr(service.model, "net", None), "local_memory_runtime", None)
            if runtime is None:
                raise ValueError("required Local-TTT checkpoint has no net.local_memory_runtime")
            if runtime.encoder.action_proj.in_features != 15:
                raise ValueError("V3 RoboCasa online Local-TTT requires raw15 evidence")
            if runtime.core.ttt_tbptt_steps != 16 or runtime.core.k_local != 4 or runtime.core.local_dim != 32:
                raise ValueError("V3 RoboCasa Local-TTT runtime contract drift")
            self.memory = OnlineLocalMemory(
                runtime.encoder,
                runtime.core,
                max_sessions=max_sessions,
                max_evidence_steps=max_evidence_steps,
            )

    @staticmethod
    def _validate_frame(frame: torch.Tensor, name: str) -> torch.Tensor:
        value = frame.detach().to("cpu").contiguous()
        if value.ndim != 3 or value.shape[0] != 3 or value.dtype != torch.uint8:
            raise ValueError(f"{name} must decode to uint8 [3,H,W]")
        if value.shape[-2:] != (256, 256):
            raise ValueError(f"{name} must match training cache camera size 256x256, got {tuple(value.shape[-2:])}")
        return value.clone()

    def _encode_camera_prefix(self, frames: tuple[torch.Tensor, ...], max_endpoint: int) -> torch.Tensor:
        if max_endpoint < 0 or max_endpoint >= len(frames) or max_endpoint % 4 != 0:
            raise ValueError("online causal endpoint is invalid")
        clip = torch.stack(frames[: max_endpoint + 1], dim=1).unsqueeze(0)  # [1,3,T,H,W]
        device = next(self.service.model.parameters()).device
        with torch.inference_mode():
            latent = self.service.model._encode_vision_item(clip.to(device), num_views=1)
        expected = max_endpoint // 4 + 1
        if latent.ndim != 5 or latent.shape[0] != 1 or latent.shape[1] != 48 or latent.shape[2] != expected:
            raise ValueError(
                f"online causal VAE shape mismatch: got {tuple(latent.shape)}, expected [1,48,{expected},16,16]"
            )
        if tuple(latent.shape[-2:]) != (16, 16) or not torch.isfinite(latent).all():
            raise FloatingPointError("online causal VAE produced invalid RoboCasa latent")
        # Training cache persists fp16 and B1 computes visual96 from that persisted value.
        return latent[0].detach().to(device="cpu", dtype=torch.float16).contiguous()

    def _visual96(
        self,
        left_frames: tuple[torch.Tensor, ...],
        wrist_frames: tuple[torch.Tensor, ...],
        source_steps: tuple[int, ...],
    ) -> torch.Tensor:
        if not source_steps:
            return torch.empty(0, 96, dtype=torch.float32)
        max_endpoint = (max(source_steps) // 4) * 4
        left_latent = self._encode_camera_prefix(left_frames, max_endpoint)
        wrist_latent = self._encode_camera_prefix(wrist_frames, max_endpoint)
        summaries = []
        for step in source_steps:
            endpoint = (step // 4) * 4
            index = endpoint // 4
            summaries.append(latent_to_visual96(left_latent[index], wrist_latent[index]))
        return torch.stack(summaries)

    def _parse(self, req: dict[str, Any]) -> tuple[OnlineMemoryRequest, _VisualRecord | None, _VisualRecord]:
        if self.memory is None:
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
            replacement = _VisualRecord(episode_id, 0, (), ())
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
        expected_steps = tuple(range(previous.consumer_step, consumer_step))
        if len(rows) != len(expected_steps):
            raise ValueError("Local-TTT completed evidence count does not match consumer frontier")
        left = list(previous.left_frames)
        wrist = list(previous.wrist_frames)
        actions = []
        actual_steps = []
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

        if len(left) != consumer_step or len(wrist) != consumer_step:
            raise RuntimeError("Local-TTT visual history length/frontier mismatch")
        source_steps = tuple(actual_steps)
        visual = self._visual96(tuple(left), tuple(wrist), source_steps)
        action_tensor = torch.stack(actions) if actions else torch.empty(0, 15)
        replacement = _VisualRecord(episode_id, consumer_step, tuple(left), tuple(wrist))
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
        return {
            "session_id": update.memory.session_id,
            "episode_id": replacement.episode_id,
            "consumer_step": replacement.consumer_step,
            "prefix_present": replacement.token is not None,
            "replay": update.memory.replay,
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
            **self.memory.info(),
        }
