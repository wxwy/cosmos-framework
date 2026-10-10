# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Debug-only Phase4A exact-window Local evidence and stable-slot planning."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch.nn import functional as F

from cosmos_framework.data.generator.action.datasets.action_sft_dataset import ActionSFTDataset
from cosmos_framework.data.generator.action.datasets.robocasa_exact_window_cache import ExactWindowEpisodeKey
from cosmos_framework.data.generator.action.datasets.robocasa_exact_window_cached_sft import (
    RoboCasaExactWindowCachedDataset,
)
from cosmos_framework.model.generator.mot.local_memory_segment import SegmentBatch, SegmentIdentity, SegmentProvenance


def robocasa_current_latent_to_visual96(z0: torch.Tensor) -> torch.Tensor:
    """Use the V2 current-latent visual summary without detaching its graph."""
    if not isinstance(z0, torch.Tensor) or z0.dtype != torch.float32 or z0.ndim != 3 or z0.shape[0] != 48:
        raise ValueError("Local current z0 必须是 fp32 [48,H,W]")
    if min(z0.shape[1:]) <= 0 or not bool(torch.isfinite(z0).all()):
        raise ValueError("Local current z0 必须为有限且非空的 latent")
    return F.adaptive_avg_pool2d(z0.unsqueeze(0), output_size=(1, 2)).flatten()


def _digest(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _positive(name: str, value: int) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} 必须是正整数")


@dataclass(frozen=True)
class ExactWindowLocalEpisode:
    key: ExactWindowEpisodeKey
    uid: str
    task_class: str
    flat_start: int
    window_count: int
    segment_count: int
    source_digest: str


class ExactWindowLocalCatalog:
    """Build planning metadata from the Phase3 raw dataset, without reading raw items."""

    def __init__(self, wrapped_sft: ActionSFTDataset, *, ttt_tbptt_steps: int) -> None:
        if not isinstance(wrapped_sft, ActionSFTDataset) or not isinstance(
            wrapped_sft._dataset, RoboCasaExactWindowCachedDataset
        ):
            raise TypeError("Local catalog 只接受 Phase3 ActionSFTDataset/raw dataset")
        _positive("ttt_tbptt_steps", ttt_tbptt_steps)
        self.wrapped_sft = wrapped_sft
        self.raw = wrapped_sft._dataset
        self.ttt_tbptt_steps = ttt_tbptt_steps
        self.cache_corpus_digest = self.raw.catalog.corpus_digest
        self.source_binding_digest = self.raw.source_reader.source_binding_digest
        self.manifest_digest = self.raw.catalog.manifest_sha256
        blocks = self.raw.get_shuffle_blocks()
        if len(blocks) != len(self.raw.catalog.episodes):
            raise ValueError("Phase3 cache block 与 episode 数量不一致")
        episodes = []
        expected_flat = 0
        for record, (flat_start, count) in zip(self.raw.catalog.episodes, blocks, strict=True):
            if flat_start != expected_flat or count != record.window_count or count <= 0:
                raise ValueError("Phase3 cache block 的 flat layout 不一致")
            if record.source_video_frames is not None and record.source_video_frames - 16 != count:
                raise ValueError("exact-window episode 的 F-16 几何不一致")
            bound = self.raw.source_reader._bound.get(record.key)
            if bound is None or len(bound.first_rows) != 17 or len(bound.terminal_rows) != 17:
                raise ValueError("Local episode 缺少首尾 global-row witness")
            source_digest = _digest(
                {
                    "cache_corpus_digest": self.cache_corpus_digest,
                    "source_binding_digest": self.source_binding_digest,
                    "task_class": record.key.task_class,
                    "episode_index": record.key.episode_index,
                    "window_count": count,
                    "first_global_row_indices": bound.first_rows,
                    "terminal_global_row_indices": bound.terminal_rows,
                }
            )
            uid = f"{record.key.task_slug}/episode_{record.key.episode_index:06d}"
            episodes.append(
                ExactWindowLocalEpisode(
                    record.key,
                    uid,
                    record.key.task_class,
                    flat_start,
                    count,
                    (count + ttt_tbptt_steps - 1) // ttt_tbptt_steps,
                    source_digest,
                )
            )
            expected_flat += count
        if expected_flat != len(self.raw) or len({episode.uid for episode in episodes}) != len(episodes):
            raise ValueError("Local episode flat layout 或 uid 重复")
        self.episodes = tuple(episodes)
        self.by_uid = {episode.uid: episode for episode in self.episodes}
        self.task_classes = tuple(sorted({episode.task_class for episode in self.episodes}))


@dataclass(frozen=True)
class ExactWindowSlotFrontier:
    uid: str | None = None
    binding_epoch: int | None = None
    cursor: int = 0
    next_segment_id: int = 0


@dataclass(frozen=True)
class ExactWindowCatalogFrontier:
    epoch: int
    assigned_in_epoch: tuple[str, ...]
    slots: tuple[ExactWindowSlotFrontier, ...]


@dataclass(frozen=True)
class ExactWindowSegmentRequest:
    episode: ExactWindowLocalEpisode
    identity: SegmentIdentity
    binding_epoch: int
    valid_count: int


@dataclass(frozen=True)
class ExactWindowGroupedPlan:
    members: tuple[tuple[ExactWindowSegmentRequest, ...], ...]
    candidate_frontier: ExactWindowCatalogFrontier
    member_counts: tuple[int, ...]

    @property
    def n_window(self) -> int:
        return sum(self.member_counts)


class ExactWindowRankPlanner:
    """Plan a candidate GA window without publishing the live frontier."""

    def __init__(
        self,
        catalog: ExactWindowLocalCatalog,
        *,
        rank: int,
        world_size: int,
        b_stream: int = 8,
        active_ga: int = 2,
        seed: int = 0,
    ) -> None:
        _positive("world_size", world_size)
        _positive("b_stream", b_stream)
        _positive("active_ga", active_ga)
        if type(rank) is not int or not 0 <= rank < world_size or type(seed) is not int:
            raise ValueError("rank/seed 不合法")
        self.catalog, self.rank, self.world_size = catalog, rank, world_size
        self.b_stream, self.active_ga, self.seed = b_stream, active_ga, seed
        self.episodes = tuple(
            episode
            for episode in catalog.episodes
            if int(hashlib.sha256(episode.uid.encode()).hexdigest(), 16) % world_size == rank
        )
        if len(self.episodes) < b_stream:
            raise ValueError("rank episode 少于 b_stream，不能重复 episode 凑 batch")
        self.by_uid = {episode.uid: episode for episode in self.episodes}

    def initial_frontier(self) -> ExactWindowCatalogFrontier:
        return ExactWindowCatalogFrontier(0, (), tuple(ExactWindowSlotFrontier() for _ in range(self.b_stream)))

    def _queue(self, epoch: int) -> tuple[str, ...]:
        per_task = {}
        for task in self.catalog.task_classes:
            entries = [episode.uid for episode in self.episodes if episode.task_class == task]
            random.Random(int(_digest([self.seed, epoch, task, self.rank]), 16)).shuffle(entries)
            per_task[task] = entries
        order = []
        while any(per_task.values()):
            for task in self.catalog.task_classes:
                if per_task[task]:
                    order.append(per_task[task].pop())
        return tuple(order)

    def _validate_frontier(self, frontier: ExactWindowCatalogFrontier) -> None:
        if type(frontier.epoch) is not int or frontier.epoch < 0 or len(frontier.slots) != self.b_stream:
            raise ValueError("Local frontier 几何不匹配")
        assigned = frontier.assigned_in_epoch
        bound = [slot.uid for slot in frontier.slots if slot.uid is not None]
        if len(set(assigned)) != len(assigned) or len(set(bound)) != len(bound):
            raise ValueError("Local frontier episode 重复")
        if any(uid not in self.by_uid for uid in (*assigned, *bound)):
            raise ValueError("Local frontier 包含非本 rank episode")
        for slot in frontier.slots:
            if (
                slot.cursor < 0
                or slot.next_segment_id < 0
                or (slot.uid is None) != (slot.binding_epoch is None)
                or (slot.uid is None and slot.cursor != 0)
                or (slot.binding_epoch is not None and not 0 <= slot.binding_epoch <= frontier.epoch)
            ):
                raise ValueError("Local slot frontier 非法")

    def validate_frontier(self, frontier: ExactWindowCatalogFrontier) -> None:
        """只读校验恢复 frontier，不生成后续窗口。"""
        if type(frontier) is not ExactWindowCatalogFrontier or type(frontier.slots) is not tuple:
            raise ValueError("Local frontier 类型不合法")
        if type(frontier.assigned_in_epoch) is not tuple:
            raise ValueError("Local frontier assigned 类型不合法")
        if any(type(slot) is not ExactWindowSlotFrontier for slot in frontier.slots):
            raise ValueError("Local slot frontier 类型不合法")
        if any(type(uid) is not str for uid in frontier.assigned_in_epoch) or any(
            (slot.uid is not None and type(slot.uid) is not str)
            or type(slot.cursor) is not int
            or type(slot.next_segment_id) is not int
            for slot in frontier.slots
        ):
            raise ValueError("Local frontier uid/cursor/segment_id 类型不合法")
        self._validate_frontier(frontier)
        if frontier.assigned_in_epoch != tuple(sorted(frontier.assigned_in_epoch)):
            raise ValueError("Local frontier assigned 顺序不合法")
        if not set(frontier.assigned_in_epoch).issubset(self._queue(frontier.epoch)):
            raise ValueError("Local frontier assigned 不属于当前确定性队列")
        for slot in frontier.slots:
            if slot.uid is None:
                continue
            episode = self.by_uid[slot.uid]
            if (
                type(slot.binding_epoch) is not int
                or slot.cursor >= episode.segment_count
                or (slot.binding_epoch == frontier.epoch and slot.uid not in frontier.assigned_in_epoch)
            ):
                raise ValueError("Local slot cursor/binding epoch 不合法")

    def plan_window(self, frontier: ExactWindowCatalogFrontier) -> ExactWindowGroupedPlan:
        self._validate_frontier(frontier)
        slots = list(frontier.slots)
        epoch, assigned = frontier.epoch, set(frontier.assigned_in_epoch)
        used: set[str] = set()
        members = []
        counts = []
        for _ in range(self.active_ga):
            requests = []
            for local_slot, slot in enumerate(slots):
                if slot.uid is None:
                    queue = self._queue(epoch)
                    if all(uid in assigned for uid in queue):
                        epoch += 1
                        assigned = set()
                        queue = self._queue(epoch)
                    forbidden = {bound.uid for bound in slots if bound.uid is not None} | used
                    uid = next((item for item in queue if item not in assigned and item not in forbidden), None)
                    if uid is None:
                        raise RuntimeError("没有足够的不重复 episode 绑定 stable slots")
                    assigned.add(uid)
                    episode = self.by_uid[uid]
                    slot = ExactWindowSlotFrontier(uid, epoch, 0, slot.next_segment_id)
                else:
                    episode = self.by_uid[slot.uid]
                if not 0 <= slot.cursor < episode.segment_count:
                    raise ValueError("Local slot cursor 超出 episode segment 范围")
                terminal = slot.cursor == episode.segment_count - 1
                identity = SegmentIdentity(
                    self.rank * self.b_stream + local_slot,
                    episode.uid,
                    episode.task_class,
                    slot.cursor,
                    slot.next_segment_id,
                    episode.source_digest,
                    terminal,
                )
                count = min(
                    self.catalog.ttt_tbptt_steps, episode.window_count - slot.cursor * self.catalog.ttt_tbptt_steps
                )
                requests.append(ExactWindowSegmentRequest(episode, identity, slot.binding_epoch, count))
                used.add(episode.uid)
                slots[local_slot] = ExactWindowSlotFrontier(
                    None if terminal else episode.uid,
                    None if terminal else slot.binding_epoch,
                    0 if terminal else slot.cursor + 1,
                    slot.next_segment_id + 1,
                )
            members.append(tuple(requests))
            counts.append(sum(request.valid_count for request in requests))
        candidate = ExactWindowCatalogFrontier(epoch, tuple(sorted(assigned)), tuple(slots))
        return ExactWindowGroupedPlan(tuple(members), candidate, tuple(counts))


@dataclass(frozen=True)
class PreparedExactWindowSegment:
    """Immutable request binding; raw CPU items only, no prompt transform or model state."""

    request: ExactWindowSegmentRequest
    items: tuple[tuple[int, dict[str, Any]], ...]


class ExactWindowSegmentProducer:
    """Reuse a single Phase3 raw item for current payload and next-step evidence."""

    def __init__(self, catalog: ExactWindowLocalCatalog, *, config_digest: str) -> None:
        if not config_digest:
            raise ValueError("Local producer 需要 config_digest")
        self.catalog, self.config_digest = catalog, config_digest

    def prepare(
        self,
        request: ExactWindowSegmentRequest,
        *,
        raw_getter: Callable[[int], dict[str, Any]] | None = None,
    ) -> PreparedExactWindowSegment:
        """Read deterministic raw windows; never consume transform/model RNG or Local state."""
        episode, identity = request.episode, request.identity
        if self.catalog.by_uid.get(episode.uid) is not episode:
            raise ValueError("Local request episode 不是 catalog capability")
        t = self.catalog.ttt_tbptt_steps
        start = identity.cursor * t
        expected = min(t, episode.window_count - start)
        if (
            identity.episode_id != episode.uid
            or identity.category != episode.task_class
            or identity.source_digest != episode.source_digest
            or identity.training_stream_end != (start + expected == episode.window_count)
            or request.valid_count != expected
            or expected <= 0
        ):
            raise ValueError("Local request identity/geometry 不匹配")
        raw = self.catalog.raw
        if (
            raw.catalog.corpus_digest != self.catalog.cache_corpus_digest
            or raw.catalog.manifest_sha256 != self.catalog.manifest_digest
            or raw.source_reader.source_binding_digest != self.catalog.source_binding_digest
        ):
            raise ValueError("Local catalog/source identity 已漂移")
        getter = raw.__getitem__ if raw_getter is None else raw_getter
        items = tuple((step, getter(episode.flat_start + step)) for step in range(max(0, start - 1), start + expected))
        return PreparedExactWindowSegment(request, items)

    def materialize(self, prepared: PreparedExactWindowSegment) -> SegmentBatch:
        """Apply the original ordered transform and construct the exact SegmentBatch on trainer thread."""
        if not isinstance(prepared, PreparedExactWindowSegment):
            raise TypeError("raw prefetch prepared segment 类型无效")
        request = prepared.request
        episode, identity = request.episode, request.identity
        t = self.catalog.ttt_tbptt_steps
        start = identity.cursor * t
        expected = min(t, episode.window_count - start)
        if tuple(step for step, _ in prepared.items) != tuple(range(max(0, start - 1), start + expected)):
            raise ValueError("raw prefetch Segment frame range 不匹配")
        raw = self.catalog.raw
        items = dict(prepared.items)
        visual = torch.zeros((1, t, 96), dtype=torch.float32)
        evidence_visual = torch.zeros_like(visual)
        evidence_action = torch.zeros((1, t, 15), dtype=torch.float32)
        consumer_valid = torch.zeros((1, t), dtype=torch.bool)
        evidence_valid = torch.zeros_like(consumer_valid)
        consumer_step = torch.full((1, t), -1, dtype=torch.long)
        evidence_source_step = torch.full((1, t), -1, dtype=torch.long)
        payloads: list[dict[str, Any] | None] = [None] * t
        for index in range(expected):
            step = start + index
            item = items[step]
            if (
                item["task_class"] != episode.task_class
                or item["episode_index"] != episode.key.episode_index
                or item["start_frame"] != step
                or item["cache_corpus_digest"] != self.catalog.cache_corpus_digest
                or item["source_binding_digest"] != self.catalog.source_binding_digest
                or item["cached_latent_required"] is not True
                or tuple(item["video_latent"].shape) != raw.catalog.latent_shape
                or item["action"].shape != (17, 15)
            ):
                raise ValueError("Local raw item identity/ABI 不匹配")
            visual[0, index] = robocasa_current_latent_to_visual96(item["video_latent"][0])
            payloads[index] = self.catalog.wrapped_sft._transform(dict(item), self.catalog.wrapped_sft._resolution)
            consumer_valid[0, index] = True
            consumer_step[0, index] = step
            if step > 0:
                previous = items[step - 1]
                if previous["start_frame"] != step - 1 or previous["action"].shape != (17, 15):
                    raise ValueError("Local previous raw item chronology 不匹配")
                evidence_visual[0, index] = robocasa_current_latent_to_visual96(previous["video_latent"][0])
                evidence_action[0, index] = previous["action"][1]
                evidence_valid[0, index] = True
                evidence_source_step[0, index] = step - 1
        segment = SegmentBatch(
            consumer_visual_summary=visual,
            consumer_payload=(tuple(payloads),),
            consumer_valid=consumer_valid,
            consumer_step=consumer_step,
            evidence_visual_summary_prev=evidence_visual,
            evidence_executed_action_prev=evidence_action,
            evidence_valid=evidence_valid,
            evidence_source_step=evidence_source_step,
            slot_id=torch.tensor([identity.slot_id], dtype=torch.long),
            episode_id=(episode.uid,),
            category=(episode.task_class,),
            segment_provenance=SegmentProvenance(
                self.catalog.manifest_digest,
                self.config_digest,
                episode.source_digest,
                identity.segment_id,
            ),
        )
        segment.validate(t)
        return segment

    def produce(self, request: ExactWindowSegmentRequest) -> SegmentBatch:
        return self.materialize(self.prepare(request))


@dataclass(frozen=True)
class ExactWindowNativeIndexBatch:
    index: int
    slot_ids: tuple[int, ...]
    payloads: tuple[Any, ...]
    consumer_steps: tuple[int, ...]


def gather_exact_window_same_index(segments: tuple[SegmentBatch, ...], index: int) -> ExactWindowNativeIndexBatch:
    if not segments or type(index) is not int:
        raise ValueError("same-index 需要非空 segments 和整数 index")
    width = segments[0].consumer_valid.shape[1]
    if not 0 <= index < width or any(segment.consumer_valid.shape != (1, width) for segment in segments):
        raise ValueError("same-index segment width 不一致")
    ids = tuple(int(segment.slot_id[0]) for segment in segments)
    if ids != tuple(sorted(set(ids))):
        raise ValueError("same-index slot 必须互异且排序")
    selected = [segment for segment in segments if bool(segment.consumer_valid[0, index])]
    return ExactWindowNativeIndexBatch(
        index,
        tuple(int(segment.slot_id[0]) for segment in selected),
        tuple(segment.consumer_payload[0][index] for segment in selected),
        tuple(int(segment.consumer_step[0, index]) for segment in selected),
    )
