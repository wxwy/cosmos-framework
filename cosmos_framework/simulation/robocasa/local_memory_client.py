"""Client-side completed-evidence ledger for RoboCasa online Local-TTT evaluation."""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np

_EVIDENCE_VERSION = "causal_visual96_executed_action20_v1"
_EVIDENCE_FORMAT = "robocasa_rgb_action20_v1"


@dataclass
class _Episode:
    session_id: str
    episode_id: str
    consumer_step: int = 0
    first_request: bool = True
    evidence: list[dict[str, Any]] = field(default_factory=list)


class RoboCasaLocalMemoryClient:
    """Track only observations/actions that were actually executed in RoboCasa.

    The training-side canonical Local-TTT evidence uses the composed left|wrist
    pre-action image plus the canonical 20-D action produced by the RoboCasa
    dataset adapter. Predicted-but-unexecuted chunk members are never recorded.
    """

    def __init__(self, *, enabled: bool = True, max_evidence_steps: int = 256) -> None:
        self.enabled = bool(enabled)
        self.max_evidence_steps = int(max_evidence_steps)
        if self.max_evidence_steps <= 0:
            raise ValueError("max_evidence_steps must be positive")
        self.client_id = uuid.uuid4().hex
        self._episodes: dict[int, _Episode] = {}

    def begin(self, slot: int = 0) -> None:
        if not self.enabled:
            return
        if slot in self._episodes:
            raise RuntimeError("close the preceding memory episode before reusing an environment slot")
        self._episodes[slot] = _Episode(
            session_id=f"{self.client_id}:{slot}",
            episode_id=uuid.uuid4().hex,
        )

    def _get(self, slot: int) -> _Episode:
        episode = self._episodes.get(slot)
        if episode is None:
            raise RuntimeError("RoboCasa Local Memory episode was not started")
        return episode

    def record_executed(
        self,
        slot: int,
        image: np.ndarray,
        executed_action20: np.ndarray | list[float],
    ) -> None:
        if not self.enabled:
            return
        episode = self._get(slot)
        rgb = np.asarray(image)
        if rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError(f"completed RoboCasa evidence image must be HWC RGB, got {rgb.shape}")
        if rgb.dtype != np.uint8:
            raise ValueError(f"completed RoboCasa evidence image must be uint8, got {rgb.dtype}")
        action = np.asarray(executed_action20, dtype=np.float32).reshape(-1)
        if action.shape != (20,) or not np.isfinite(action).all():
            raise ValueError("completed RoboCasa evidence action must be finite canonical [20]")
        if len(episode.evidence) >= self.max_evidence_steps:
            raise RuntimeError("too many unconsumed RoboCasa Local Memory evidence steps")
        episode.evidence.append(
            {
                "source_step": episode.consumer_step,
                "image": np.ascontiguousarray(rgb),
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
            "evidence_version": _EVIDENCE_VERSION,
            "evidence_format": _EVIDENCE_FORMAT,
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
            raise ValueError("RoboCasa Local Memory response does not acknowledge the exact evidence frontier")
        episode.evidence.clear()
        episode.first_request = False

    def end(self, slot: int = 0) -> str | None:
        episode = self._episodes.pop(slot, None)
        return None if episode is None else episode.session_id

    def session_id(self, slot: int = 0) -> str | None:
        if not self.enabled:
            return None
        return self._get(slot).session_id
