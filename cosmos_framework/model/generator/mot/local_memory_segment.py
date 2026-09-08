# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Synthetic CPU contracts for canonical Local-Memory segments.

This module deliberately has no dataset, trainer, checkpoint, or model-forward
dependencies. Production wiring is a separate Gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch


@dataclass(frozen=True)
class SegmentProvenance:
    manifest_digest: str
    config_digest: str
    source_digest: str
    segment_id: int


@dataclass(frozen=True)
class SegmentBatch:
    consumer_visual_summary: torch.Tensor
    consumer_payload: tuple[tuple[Any | None, ...], ...]
    consumer_valid: torch.Tensor
    consumer_step: torch.Tensor
    evidence_visual_summary_prev: torch.Tensor
    evidence_executed_action_prev: torch.Tensor
    evidence_valid: torch.Tensor
    evidence_source_step: torch.Tensor
    slot_id: torch.Tensor
    episode_id: tuple[str, ...]
    category: tuple[str, ...]
    segment_provenance: SegmentProvenance

    def validate(self, ttt_tbptt_steps: int) -> None:
        if self.consumer_visual_summary.ndim != 3 or self.consumer_visual_summary.shape[-1] != 96:
            raise ValueError("consumer_visual_summary must have shape [B,T,96].")
        batch, steps, _ = self.consumer_visual_summary.shape
        if not 1 <= steps <= ttt_tbptt_steps:
            raise ValueError("segment length must be in [1, ttt_tbptt_steps].")
        expected = (batch, steps)
        for name, value in (
            ("consumer_valid", self.consumer_valid),
            ("evidence_valid", self.evidence_valid),
            ("consumer_step", self.consumer_step),
            ("evidence_source_step", self.evidence_source_step),
        ):
            if tuple(value.shape) != expected:
                raise ValueError(f"{name} must have shape [B,T].")
        if tuple(self.evidence_visual_summary_prev.shape) != (batch, steps, 96):
            raise ValueError("evidence_visual_summary_prev must have shape [B,T,96].")
        if tuple(self.evidence_executed_action_prev.shape) != (batch, steps, 10):
            raise ValueError("evidence_executed_action_prev must have shape [B,T,10].")
        if tuple(self.slot_id.shape) != (batch,) or len(self.episode_id) != batch or len(self.category) != batch:
            raise ValueError("slot/episode/category batch identities are incompatible.")
        if len(self.consumer_payload) != batch or any(len(row) != steps for row in self.consumer_payload):
            raise ValueError("consumer_payload must have shape [B,T].")
        valid = self.consumer_valid.bool()
        evidence_valid = self.evidence_valid.bool()
        if not torch.equal(self.evidence_source_step.eq(-1), ~evidence_valid):
            raise ValueError("evidence_source_step must be -1 iff evidence is invalid.")
        if torch.any(evidence_valid & ~valid):
            raise ValueError("evidence cannot be valid for PAD consumers.")
        if torch.any(valid & self.consumer_step.eq(0) & evidence_valid):
            raise ValueError("consumer step0 must have absent evidence.")
        if torch.any(evidence_valid & self.evidence_source_step.ne(self.consumer_step - 1)):
            raise ValueError("evidence_source_step must equal consumer_step - 1.")

    def gather_consumers(self, local_tokens: torch.Tensor, local_present: torch.Tensor) -> tuple[list[Any], list[torch.Tensor | None], list[tuple[int, str, int]]]:
        batch, steps = self.consumer_valid.shape
        if local_tokens.ndim != 4 or tuple(local_tokens.shape[:2]) != (batch, steps):
            raise ValueError("Local tokens must have shape [B,T,K,D].")
        if tuple(local_present.shape) != (batch, steps):
            raise ValueError("Local token/presence shapes are incompatible with SegmentBatch.")
        payloads: list[Any] = []
        local: list[torch.Tensor | None] = []
        identities: list[tuple[int, str, int]] = []
        for row in range(batch):
            for index in range(steps):
                if not bool(self.consumer_valid[row, index]):
                    continue
                payloads.append(self.consumer_payload[row][index])
                local.append(local_tokens[row, index] if bool(local_present[row, index]) else None)
                identities.append((int(self.slot_id[row]), self.episode_id[row], int(self.consumer_step[row, index])))
        return payloads, local, identities


@dataclass(frozen=True)
class GAWindowPlan:
    members: tuple[tuple[int, str, int], ...]
    planned_n_valid: tuple[int, ...]
    attempt: int = 0

    def __post_init__(self) -> None:
        if not self.members or len(self.members) != len(self.planned_n_valid) or any(value <= 0 for value in self.planned_n_valid):
            raise ValueError("GAWindowPlan requires non-empty positive planned counts.")

    @property
    def n_window(self) -> int:
        return sum(self.planned_n_valid)

    @property
    def ga_effective(self) -> int:
        return len(self.members)

    def objective(self, index: int, consumer_loss: torch.Tensor, auxiliary_loss: torch.Tensor, actual_n_valid: int) -> torch.Tensor:
        if actual_n_valid != self.planned_n_valid[index]:
            raise ValueError("actual gathered count must equal planned count.")
        return (self.planned_n_valid[index] / self.n_window) * consumer_loss + auxiliary_loss / self.ga_effective


@dataclass(frozen=True)
class SegmentIdentity:
    slot_id: int
    episode_id: str
    category: str
    cursor: int
    segment_id: int
    source_digest: str
    training_stream_end: bool = False


class RankLocalSegmentScheduler:
    """Deterministic, main-process-only synthetic metadata scheduler."""

    def __init__(self, *, rank: int, target_distribution: dict[str, float], num_workers: int = 0) -> None:
        if num_workers != 0:
            raise ValueError("RankLocalSegmentScheduler requires num_workers=0.")
        if rank < 0 or not target_distribution or any(value <= 0 for value in target_distribution.values()):
            raise ValueError("rank and target_distribution are invalid.")
        total = sum(target_distribution.values())
        self.rank = rank
        self.num_workers = num_workers
        self.target_distribution = {key: value / total for key, value in target_distribution.items()}
        self.cumulative_valid_consumer_exposure = {key: 0 for key in self.target_distribution}
        self.admission_order: list[SegmentIdentity] = []

    def admit(self, candidates: Iterable[SegmentIdentity]) -> SegmentIdentity:
        eligible = [item for item in candidates if item.category in self.target_distribution]
        if not eligible:
            raise ValueError("no candidate has a configured category.")
        total = sum(self.cumulative_valid_consumer_exposure.values())
        def deficit(item: SegmentIdentity) -> tuple[float, str, int, str, int]:
            observed = self.cumulative_valid_consumer_exposure[item.category] / max(total, 1)
            return (self.target_distribution[item.category] - observed, item.category, item.slot_id, item.episode_id, item.cursor)
        chosen = max(eligible, key=deficit)
        self.admission_order.append(chosen)
        return chosen

    def commit(self, identity: SegmentIdentity, valid_consumers: int) -> None:
        if valid_consumers <= 0 or not self.admission_order or self.admission_order[-1] != identity:
            raise ValueError("only the latest admitted identity may commit a positive valid count.")
        self.cumulative_valid_consumer_exposure[identity.category] += valid_consumers

    def snapshot(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "num_workers": self.num_workers,
            "target_distribution": dict(self.target_distribution),
            "cumulative_valid_consumer_exposure": dict(self.cumulative_valid_consumer_exposure),
            "admission_order": tuple(self.admission_order),
        }
