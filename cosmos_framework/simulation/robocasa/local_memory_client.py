"""Client-side completed-evidence ledger for V3 RoboCasa online Local-TTT."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from cosmos_framework.inference.robocasa_local_memory_contract import (
    CAMERA_HEIGHT,
    COMPOSITE_WIDTH,
    EVIDENCE_ACTION_DIM,
    EVIDENCE_FORMAT,
    EVIDENCE_VERSION,
    PREPROCESS_PROFILE,
)
from cosmos_framework.simulation.robocasa.eval_utils import b64_png


@dataclass
class _Episode:
    session_id: str
    episode_id: str
    image_size: int
    preprocess_profile: str
    consumer_step: int = 0
    first_request: bool = True
    evidence: list[dict[str, Any]] = field(default_factory=list)


class RoboCasaLocalMemoryClient:
    """Track unacknowledged completed composite RGB and canonical raw15 commands."""

    def __init__(self, *, enabled: bool = True, max_evidence_steps: int = 256) -> None:
        self.enabled = bool(enabled)
        self.max_evidence_steps = int(max_evidence_steps)
        if self.max_evidence_steps <= 0:
            raise ValueError("max_evidence_steps must be positive")
        self.client_id = uuid.uuid4().hex
        self._episodes: dict[int, _Episode] = {}

    def begin(self, slot: int = 0, *, image_size: int = 256, preprocess_profile: str = PREPROCESS_PROFILE) -> None:
        if not self.enabled:
            return
        if slot in self._episodes:
            raise RuntimeError("close the preceding Local-TTT episode before reusing an environment slot")
        if type(image_size) is not int or image_size <= 0 or preprocess_profile != PREPROCESS_PROFILE:
            raise ValueError("Local-TTT image_size/preprocess_profile 不合法")
        self._episodes[slot] = _Episode(
            session_id=f"{self.client_id}:{slot}:{uuid.uuid4().hex}",
            episode_id=uuid.uuid4().hex,
            image_size=image_size,
            preprocess_profile=preprocess_profile,
        )

    def _get(self, slot: int) -> _Episode:
        episode = self._episodes.get(slot)
        if episode is None:
            raise RuntimeError("RoboCasa Local-TTT episode was not started")
        return episode

    @staticmethod
    def _image(value: np.ndarray, name: str) -> np.ndarray:
        image = np.asarray(value)
        if image.shape != (CAMERA_HEIGHT, COMPOSITE_WIDTH, 3) or image.dtype != np.uint8:
            raise ValueError(
                f"{name} must be uint8 [{CAMERA_HEIGHT},{COMPOSITE_WIDTH},3], got {image.shape} {image.dtype}"
            )
        return np.ascontiguousarray(image)

    def record_completed(
        self,
        slot: int,
        composite_image: np.ndarray,
        executed_action15: np.ndarray | list[float],
    ) -> None:
        if not self.enabled:
            return
        episode = self._get(slot)
        composite = self._image(composite_image, "composite_image")
        action = np.asarray(executed_action15, dtype=np.float32).reshape(-1)
        if action.shape != (EVIDENCE_ACTION_DIM,) or not np.isfinite(action).all():
            raise ValueError("executed Local-TTT evidence action must be finite raw15")
        if len(episode.evidence) >= self.max_evidence_steps:
            raise RuntimeError("too many unacknowledged Local-TTT evidence steps")
        episode.evidence.append(
            {
                "source_step": episode.consumer_step,
                "composite_image": b64_png(composite),
                "executed_action": action.tolist(),
            }
        )
        episode.consumer_step += 1

    def payload(self, slot: int = 0) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        episode = self._get(slot)
        return {
            "session_id": episode.session_id,
            "episode_id": episode.episode_id,
            "consumer_step": episode.consumer_step,
            "reset": episode.first_request,
            "image_size": episode.image_size,
            "preprocess_profile": episode.preprocess_profile,
            "evidence_version": EVIDENCE_VERSION,
            "evidence_format": EVIDENCE_FORMAT,
            "evidence": [dict(row) for row in episode.evidence],
        }

    def acknowledge(self, slot: int, status: dict[str, Any]) -> None:
        if not self.enabled:
            return
        episode = self._get(slot)
        if (
            not isinstance(status, dict)
            or status.get("session_id") != episode.session_id
            or status.get("episode_id") != episode.episode_id
            or status.get("consumer_step") != episode.consumer_step
        ):
            raise ValueError("Local-TTT response does not acknowledge the exact completed-evidence frontier")
        episode.evidence.clear()
        episode.first_request = False

    def end(self, slot: int = 0) -> str | None:
        episode = self._episodes.pop(slot, None)
        return None if episode is None else episode.session_id

    def session_id(self, slot: int = 0) -> str | None:
        if not self.enabled:
            return None
        return self._get(slot).session_id
