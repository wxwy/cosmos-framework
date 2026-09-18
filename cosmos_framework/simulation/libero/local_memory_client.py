"""Client-side completed observation/action ledger for online Local Memory."""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field


@dataclass
class _Episode:
    session_id: str
    episode_id: str
    consumer_step: int = 0
    first_request: bool = True
    evidence: list[dict] = field(default_factory=list)


class ClientLocalMemory:
    """Record executed environment steps, never unexecuted predicted actions."""

    def __init__(self, *, enabled=False, max_evidence_steps=256):
        self.enabled = bool(enabled)
        self.max_evidence_steps = max_evidence_steps
        self.client_id = uuid.uuid4().hex
        self._episodes: dict[int, _Episode] = {}

    def begin(self, slot=0):
        if not self.enabled:
            return
        if slot in self._episodes:
            raise RuntimeError("close the preceding memory episode before reusing an environment slot")
        self._episodes[slot] = _Episode(f"{self.client_id}:{slot}", uuid.uuid4().hex)

    def _get(self, slot):
        episode = self._episodes.get(slot)
        if episode is None:
            raise RuntimeError("memory episode was not started")
        return episode

    def record_executed(self, slot, image, executed_action, *, gripper_mode):
        if not self.enabled:
            return
        episode = self._get(slot)
        values = [float(value) for value in executed_action]
        if len(values) != 7 or not all(math.isfinite(value) for value in values):
            raise ValueError("completed LIBERO action must contain seven finite values")
        if not isinstance(image, str) or not image:
            raise ValueError("completed evidence requires the pre-action image")
        if len(episode.evidence) >= self.max_evidence_steps:
            raise RuntimeError("too many unconsumed evidence steps; request a prediction before continuing")
        if gripper_mode not in {"zero_one", "pm_one", "pm_one_flip"}:
            raise ValueError("completed evidence requires the exact LIBERO gripper_mode")
        episode.evidence.append(
            {
                "source_step": episode.consumer_step,
                "image": image,
                "executed_action": values,
                "gripper_mode": gripper_mode,
            }
        )
        episode.consumer_step += 1

    def payload(self, slot=0):
        if not self.enabled:
            return None
        episode = self._get(slot)
        return {
            "session_id": episode.session_id,
            "episode_id": episode.episode_id,
            "consumer_step": episode.consumer_step,
            "reset": episode.first_request,
            "evidence_version": "causal_visual96_executed_action10_v1",
            "evidence_format": "libero_rgb_action7_v1",
            "evidence": [dict(row) for row in episode.evidence],
        }

    def acknowledge(self, slot, status):
        if not self.enabled:
            return
        episode = self._get(slot)
        if (
            not isinstance(status, dict)
            or status.get("session_id") != episode.session_id
            or status.get("episode_id") != episode.episode_id
            or status.get("consumer_step") != episode.consumer_step
        ):
            raise ValueError("prediction response does not acknowledge this exact memory frontier")
        episode.evidence.clear()
        episode.first_request = False

    def end(self, slot=0):
        episode = self._episodes.pop(slot, None)
        return None if episode is None else episode.session_id
