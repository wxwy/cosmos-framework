# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""H3-B RoboCasa episode catalog 与纯 candidate grouped segment 规划。"""

from __future__ import annotations

import hashlib
import json
import random
from collections import OrderedDict
from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import pyarrow.dataset as arrow_dataset
import torch

from cosmos_framework.data.generator.action.datasets.robocasa_lerobot_dataset import DEFAULT_ALL_ATOMIC_TASKS
from cosmos_framework.data.generator.sequence_packing import SequencePlan
from cosmos_framework.model.generator.mot.local_memory_segment import SegmentBatch, SegmentIdentity
from cosmos_framework.model.generator.mot.robocasa_latent_evidence import RoboCasaLatentReader
from cosmos_framework.model.generator.mot.robocasa_segment_producer import RoboCasaSegmentProducer


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True, order=True)
class EpisodeKey:
    task: str
    dated_shard: str
    source_episode_index: int

    def __post_init__(self) -> None:
        if not self.task or not self.dated_shard or type(self.source_episode_index) is not int:
            raise ValueError("episode key 必须有 task/date/source index")
        if self.source_episode_index < 0:
            raise ValueError("source episode index 必须非负")

    @property
    def uid(self) -> str:
        return f"{self.task}/{self.dated_shard}/ep_{self.source_episode_index:06d}"

    @property
    def episode_id(self) -> str:
        return f"ep_{self.source_episode_index:06d}"


@dataclass(frozen=True)
class CatalogEpisode:
    key: EpisodeKey
    source_relative: str
    cache_relative: str
    source_frames: int
    valid_consumer_count: int
    source_index: int
    row_start: int
    flat_start: int
    source_digest: str

    @property
    def uid(self) -> str:
        return self.key.uid

    @property
    def segment_count(self) -> int:
        return (self.valid_consumer_count + 15) // 16


class RoboCasaEpisodeCatalog:
    def __init__(self, episodes: Sequence[CatalogEpisode], task_order: Sequence[str]) -> None:
        self.task_order = tuple(task_order)
        if not self.task_order or len(set(self.task_order)) != len(self.task_order):
            raise ValueError("task catalog 必须非空且互异")
        self.episodes = tuple(sorted(episodes, key=lambda episode: episode.key))
        if not self.episodes or len({episode.uid for episode in self.episodes}) != len(self.episodes):
            raise ValueError("episode catalog 为空或存在重复 UID")
        if {episode.key.task for episode in self.episodes} != set(self.task_order):
            raise ValueError("catalog 与冻结 task 集合不一致")
        for episode in self.episodes:
            if (
                episode.source_frames <= 32
                or episode.valid_consumer_count != episode.source_frames - 32
                or episode.segment_count <= 0
                or not episode.source_digest
            ):
                raise ValueError(f"episode {episode.uid} 的 raw15/chunk32 span 不合法")
        self.by_uid = {episode.uid: episode for episode in self.episodes}
        manifest = [
            (
                episode.uid,
                episode.source_relative,
                episode.cache_relative,
                episode.source_frames,
                episode.valid_consumer_count,
                episode.source_index,
                episode.row_start,
                episode.flat_start,
                episode.source_digest,
            )
            for episode in self.episodes
        ]
        self.manifest_digest = _digest(json.dumps(manifest, ensure_ascii=False, separators=(",", ":")))

    @classmethod
    def from_stage_a_dataset(
        cls,
        dataset: Any,
        *,
        source_root: Path,
        cache_root: Path,
    ) -> RoboCasaEpisodeCatalog:
        """复用 Stage-A loader 的 train span；只读 cache 元数据，完整 latent 在绑定时校验。"""
        if (
            dataset.fps != 20
            or dataset.chunk_length != 32
            or dataset.split != "train"
            or dataset.action_dim != 15
            or dataset._split_seed != 42
            or dataset._split_val_ratio != 0.01
            or dataset._camera_set != "left_wrist"
            or not dataset._use_state
            or not dataset._use_base_action
            or dataset._base_encoding != "raw"
            or dataset._action_normalizer is not None
            or dataset._sample_stride != 1
        ):
            raise ValueError("catalog 要求冻结 Stage-A RoboCasa train/raw15/left_wrist 合同")
        source_root, cache_root = Path(source_root).resolve(), Path(cache_root).resolve()
        shards: dict[int, tuple[Path, str, str, dict[int, tuple[int, int]]]] = {}
        for source_index, root in enumerate(dataset._all_shard_roots):
            shard = Path(root).resolve()
            try:
                relative = shard.relative_to(source_root)
            except ValueError as exc:
                raise ValueError("RoboCasa shard 不在 source_root 下") from exc
            if len(relative.parts) != 3 or relative.parts[-1] != "lerobot":
                raise ValueError("RoboCasa shard 必须为 task/date/lerobot")
            task, date, _ = relative.parts
            columns = ("episode_index", "length", "dataset_from_index", "dataset_to_index")
            rows = (
                arrow_dataset.dataset(shard / "meta/episodes", format="parquet")
                .to_table(columns=list(columns))
                .to_pydict()
            )
            lengths: dict[int, tuple[int, int]] = {}
            for episode_index, frames, first, last in zip(*(rows[column] for column in columns), strict=True):
                if (
                    any(type(value) is not int for value in (episode_index, frames, first, last))
                    or episode_index in lengths
                    or frames != last - first
                ):
                    raise ValueError(f"{shard} episode metadata 重复、类型或来源索引不合法")
                lengths[episode_index] = frames, first
            shards[source_index] = shard, task, date, lengths
        if {entry[1] for entry in shards.values()} != set(DEFAULT_ALL_ATOMIC_TASKS):
            raise ValueError("Stage-A train source 必须包含精确的18类 target atomic tasks")
        if len(dataset._episode_records) != len(dataset._episode_cum_ends):
            raise ValueError("loader episode record/cumulative index 不匹配")
        episodes = []
        for record_index, (source_index, row_start, valid_count, episode_index) in enumerate(dataset._episode_records):
            shard, task, date, lengths = shards[source_index]
            metadata = lengths.get(episode_index)
            flat_start = 0 if record_index == 0 else dataset._episode_cum_ends[record_index - 1]
            if metadata is None:
                raise ValueError("loader span 的 episode 不在源 metadata")
            frames, expected_row_start = metadata
            if (
                row_start != expected_row_start
                or valid_count != frames - 32
                or dataset._episode_cum_ends[record_index] != flat_start + valid_count
            ):
                raise ValueError("loader span 与 source episode frame_count 不匹配")
            if dataset._resolve_index(flat_start) != (source_index, row_start, episode_index, 0):
                raise ValueError("loader source timestep0 锚点不匹配")
            key = EpisodeKey(task, date, episode_index)
            relative = shard.relative_to(source_root)
            cache_relative = relative / f"{key.episode_id}.h5"
            cache = cache_root / cache_relative
            if not cache.is_file():
                raise FileNotFoundError(f"缺少 Local latent cache: {cache}")
            with h5py.File(cache, "r") as handle:
                stored_id, stored_frames = handle.attrs.get("episode_id"), handle.attrs.get("frame_count")
                if isinstance(stored_id, bytes):
                    stored_id = stored_id.decode("utf-8")
                if stored_id != key.episode_id or stored_frames != frames:
                    raise ValueError(f"cache episode/frame_count 与 source 不匹配: {cache}")
            source_relative = relative.as_posix()
            source_digest = _digest(f"{key.uid}|{source_relative}|{frames}|{valid_count}")
            episodes.append(
                CatalogEpisode(
                    key,
                    source_relative,
                    cache_relative.as_posix(),
                    frames,
                    valid_count,
                    source_index,
                    row_start,
                    flat_start,
                    source_digest,
                )
            )
        return cls(episodes, DEFAULT_ALL_ATOMIC_TASKS)

    def rank_episodes(self, rank: int, world_size: int = 8) -> tuple[CatalogEpisode, ...]:
        if type(rank) is not int or type(world_size) is not int or world_size <= 0 or not 0 <= rank < world_size:
            raise ValueError("rank/world_size 不合法")
        return tuple(episode for episode in self.episodes if int(_digest(episode.uid), 16) % world_size == rank)


@dataclass(frozen=True)
class SlotFrontier:
    uid: str | None = None
    binding_epoch: int | None = None
    cursor: int = 0
    next_segment_id: int = 0


@dataclass(frozen=True)
class CatalogFrontier:
    epoch: int
    assigned_in_epoch: tuple[str, ...]
    slots: tuple[SlotFrontier, ...]


@dataclass(frozen=True)
class SegmentRequest:
    episode: CatalogEpisode
    identity: SegmentIdentity
    binding_epoch: int
    valid_count: int


@dataclass(frozen=True)
class GroupedWindowPlan:
    members: tuple[tuple[SegmentRequest, ...], tuple[SegmentRequest, ...]]
    candidate_frontier: CatalogFrontier
    member_counts: tuple[int, int]

    @property
    def n_window(self) -> int:
        return sum(self.member_counts)


class RankLocalGroupedPlanner:
    """只计算候选 catalog/frontier；不持有或发布 live 训练状态。"""

    def __init__(self, catalog: RoboCasaEpisodeCatalog, *, rank: int, world_size: int = 8, seed: int = 0) -> None:
        if world_size != 8 or type(seed) is not int:
            raise ValueError("H3-B 固定8 rank与整数 seed")
        self.catalog, self.rank, self.world_size, self.seed = catalog, rank, world_size, seed
        self.episodes = catalog.rank_episodes(rank, world_size)
        if len(self.episodes) < 8:
            raise ValueError("rank episode 少于8，无法保持 B_stream=8")
        self.by_uid = {episode.uid: episode for episode in self.episodes}

    def initial_frontier(self) -> CatalogFrontier:
        return CatalogFrontier(0, (), (SlotFrontier(),) * 8)

    def _queue(self, epoch: int) -> tuple[str, ...]:
        per_task = {}
        for task in self.catalog.task_order:
            entries = [episode.uid for episode in self.episodes if episode.key.task == task]
            task_seed = int(_digest(f"{self.seed}|{epoch}|{task}|{self.rank}"), 16)
            random.Random(task_seed).shuffle(entries)
            per_task[task] = entries
        order = []
        while any(per_task.values()):
            for task in self.catalog.task_order:
                if per_task[task]:
                    order.append(per_task[task].pop())
        return tuple(order)

    def _bind(
        self,
        epoch: int,
        assigned: set[str],
        slots: list[SlotFrontier],
        window_used: set[str],
    ) -> tuple[CatalogEpisode, int, set[str]]:
        forbidden = {slot.uid for slot in slots if slot.uid is not None} | window_used
        queue = self._queue(epoch)
        if all(uid in assigned for uid in queue):
            epoch += 1
            assigned = set()
            queue = self._queue(epoch)
        uid = next((candidate for candidate in queue if candidate not in assigned and candidate not in forbidden), None)
        if uid is None:
            raise RuntimeError("没有足够的不重复 episode 绑定8个 stable slots")
        assigned.add(uid)
        return self.by_uid[uid], epoch, assigned

    def plan_window(self, frontier: CatalogFrontier) -> GroupedWindowPlan:
        if (
            frontier.epoch < 0
            or len(frontier.slots) != 8
            or len(set(frontier.assigned_in_epoch)) != len(frontier.assigned_in_epoch)
        ):
            raise ValueError("catalog frontier 形状或 assigned 集合不合法")
        if any(slot.uid is not None and slot.uid not in self.by_uid for slot in frontier.slots):
            raise ValueError("frontier 包含非本 rank 的 episode")
        bound = [slot.uid for slot in frontier.slots if slot.uid is not None]
        if len(set(bound)) != len(bound) or any(uid not in self.by_uid for uid in frontier.assigned_in_epoch):
            raise ValueError("frontier 重复绑定 episode 或 assigned 集合非法")
        for slot in frontier.slots:
            if (
                slot.cursor < 0
                or slot.next_segment_id < 0
                or (slot.uid is None) != (slot.binding_epoch is None)
                or (slot.uid is None and slot.cursor != 0)
                or (slot.binding_epoch is not None and not 0 <= slot.binding_epoch <= frontier.epoch)
            ):
                raise ValueError("slot frontier 身份、cursor 或 epoch 非法")
        slots = list(frontier.slots)
        epoch, assigned = frontier.epoch, set(frontier.assigned_in_epoch)
        window_used: set[str] = set()
        members: list[tuple[SegmentRequest, ...]] = []
        counts = []
        for _ in range(2):
            requests = []
            for local_slot in range(8):
                slot = slots[local_slot]
                if slot.uid is None:
                    episode, epoch, assigned = self._bind(epoch, assigned, slots, window_used)
                    slot = SlotFrontier(episode.uid, epoch, 0, slot.next_segment_id)
                else:
                    episode = self.by_uid[slot.uid]
                if not 0 <= slot.cursor < episode.segment_count:
                    raise ValueError("slot cursor 超出 episode segment 范围")
                terminal = slot.cursor == episode.segment_count - 1
                identity = SegmentIdentity(
                    self.rank * 8 + local_slot,
                    episode.key.episode_id,
                    episode.key.task,
                    slot.cursor,
                    slot.next_segment_id,
                    episode.source_digest,
                    terminal,
                )
                count = min(16, episode.valid_consumer_count - slot.cursor * 16)
                requests.append(SegmentRequest(episode, identity, slot.binding_epoch, count))
                window_used.add(episode.uid)
                slots[local_slot] = SlotFrontier(
                    None if terminal else episode.uid,
                    None if terminal else slot.binding_epoch,
                    0 if terminal else slot.cursor + 1,
                    slot.next_segment_id + 1,
                )
            members.append(tuple(requests))
            counts.append(sum(request.valid_count for request in requests))
        candidate = CatalogFrontier(epoch, tuple(sorted(assigned)), tuple(slots))
        return GroupedWindowPlan((members[0], members[1]), candidate, (counts[0], counts[1]))


class StageARoboCasaEpisodeBinder:
    """按 catalog UID 绑定官方 raw15/RGB payload 与双相机 cached Local evidence。"""

    def __init__(
        self,
        dataset: Any,
        catalog: RoboCasaEpisodeCatalog,
        *,
        source_root: Path,
        cache_root: Path,
        transform: Callable[[dict[str, Any], Any], dict[str, Any]],
        resolution: Any,
        config_digest: str,
        max_bound_episodes: int = 64,
    ) -> None:
        if (
            not callable(transform)
            or not config_digest
            or type(max_bound_episodes) is not int
            or max_bound_episodes < 16
        ):
            raise ValueError("Stage-A binder 需要 transform/config_digest 与至少16个缓存位")
        self.dataset, self.catalog = dataset, catalog
        self.source_root, self.cache_root = Path(source_root).resolve(), Path(cache_root).resolve()
        self.transform, self.resolution, self.config_digest = transform, resolution, config_digest
        self.max_bound_episodes = max_bound_episodes
        self._bound: OrderedDict[str, RoboCasaSegmentProducer] = OrderedDict()

    def producer_for(self, episode: CatalogEpisode) -> RoboCasaSegmentProducer:
        if self.catalog.by_uid.get(episode.uid) is not episode:
            raise ValueError("binder 要求 catalog 持有的 exact episode capability")
        if episode.uid in self._bound:
            self._bound.move_to_end(episode.uid)
            return self._bound[episode.uid]
        shard = Path(self.dataset._all_shard_roots[episode.source_index]).resolve()
        if shard != self.source_root / episode.source_relative:
            raise ValueError("catalog 与 Stage-A loader 的 shard 绑定发生变化")
        lerobot = self.dataset._get_dataset(episode.source_index)
        rows = lerobot.hf_dataset[episode.row_start : episode.row_start + episode.source_frames]
        if list(rows["episode_index"]) != [episode.key.source_episode_index] * episode.source_frames or [
            int(frame) for frame in rows["frame_index"]
        ] != list(range(episode.source_frames)):
            raise ValueError(f"episode {episode.uid} 的源行/frame_index 不连续")
        raw12 = torch.stack(rows["action"]).float()
        if raw12.shape != (episode.source_frames, 12) or not torch.isfinite(raw12).all():
            raise ValueError(f"episode {episode.uid} 的原始 action 不是有限12D")
        raw15 = torch.cat((raw12[:, :5], self.dataset._build_frame_wise_action(raw12)), dim=-1)
        if raw15.shape != (episode.source_frames, 15) or not torch.isfinite(raw15).all():
            raise ValueError(f"episode {episode.uid} 的官方 raw15 转换失败")
        reader = RoboCasaLatentReader(
            self.cache_root / episode.cache_relative,
            expected_episode_id=episode.key.episode_id,
            expected_source_frames=episode.source_frames,
        )

        def payload_at(step: int) -> dict[str, Any]:
            flat = episode.flat_start + step
            if self.dataset._resolve_index(flat) != (
                episode.source_index,
                episode.row_start + step,
                episode.key.source_episode_index,
                step,
            ):
                raise ValueError(f"episode {episode.uid} consumer {step} 的 loader anchor 不匹配")
            raw_payload = self.dataset[flat]
            action, video = raw_payload.get("action"), raw_payload.get("video")
            if (
                not isinstance(action, torch.Tensor)
                or action.shape != (33, 15)
                or not torch.equal(action[1:], raw15[step : step + 32])
                or not isinstance(video, torch.Tensor)
                or video.ndim != 4
                or video.shape[:2] != (3, 33)
                or video.shape[3] != 2 * video.shape[2]
                or "video_latent" in raw_payload
            ):
                raise ValueError(f"episode {episode.uid} consumer {step} 的官方 RGB/raw15 合同不匹配")
            transformed = self.transform(deepcopy(raw_payload), self.resolution)
            action_padded, action_raw = transformed.get("action"), transformed.get("action_raw")
            if (
                not isinstance(action_raw, torch.Tensor)
                or not torch.equal(action_raw, action)
                or not isinstance(action_padded, torch.Tensor)
                or action_padded.shape != (33, 64)
                or not torch.equal(action_padded[:, :15], action)
                or bool(torch.count_nonzero(action_padded[:, 15:]))
                or transformed.get("raw_action_dim") != 15
                or not isinstance(transformed.get("text_token_ids"), torch.Tensor)
                or transformed["text_token_ids"].dtype != torch.long
                or transformed["text_token_ids"].ndim != 1
                or not transformed["text_token_ids"].numel()
                or not isinstance(transformed.get("sequence_plan"), SequencePlan)
                or "video_latent" in transformed
            ):
                raise ValueError(f"episode {episode.uid} consumer {step} 的 Stage-A transform 合同不匹配")
            return transformed

        producer = RoboCasaSegmentProducer(
            reader,
            episode_id=episode.key.episode_id,
            category=episode.key.task,
            raw15=raw15,
            payload_at=payload_at,
            manifest_digest=self.catalog.manifest_digest,
            config_digest=self.config_digest,
            source_digest=episode.source_digest,
        )
        self._bound[episode.uid] = producer
        if len(self._bound) > self.max_bound_episodes:
            self._bound.popitem(last=False)
        return producer


def materialize_member(
    requests: tuple[SegmentRequest, ...],
    producer_for: Callable[[CatalogEpisode], RoboCasaSegmentProducer],
) -> tuple[SegmentBatch, ...]:
    if len(requests) != 8 or tuple(request.identity.slot_id for request in requests) != tuple(
        sorted(request.identity.slot_id for request in requests)
    ):
        raise ValueError("grouped member 必须有8个按 slot 排序的请求")
    segments = []
    for request in requests:
        producer = producer_for(request.episode)
        if (
            producer.episode_id != request.episode.key.episode_id
            or producer.category != request.episode.key.task
            or producer.reader.source_frames != request.episode.source_frames
            or producer.segment_count != request.episode.segment_count
            or producer.identity(
                slot_id=request.identity.slot_id,
                cursor=request.identity.cursor,
                segment_id=request.identity.segment_id,
            )
            != request.identity
        ):
            raise ValueError("B1 producer 与 grouped catalog/identity 不匹配")
        segment = producer.produce(request.identity)
        if int(segment.consumer_valid.sum()) != request.valid_count:
            raise ValueError("B1 segment 与 grouped valid_count 不匹配")
        segments.append(segment)
    return tuple(segments)


@dataclass(frozen=True)
class NativeIndexBatch:
    index: int
    slot_ids: tuple[int, ...]
    payloads: tuple[Any, ...]
    consumer_steps: tuple[int, ...]


def gather_same_index(segments: tuple[SegmentBatch, ...], index: int) -> NativeIndexBatch:
    if len(segments) != 8 or type(index) is not int or not 0 <= index < 16:
        raise ValueError("same-index batch 要求8 slots与 index 0..15")
    ids = tuple(int(segment.slot_id[0]) for segment in segments)
    if ids != tuple(sorted(set(ids))):
        raise ValueError("same-index segments 的 slot 必须互异且排序")
    selected = [segment for segment in segments if bool(segment.consumer_valid[0, index])]
    return NativeIndexBatch(
        index,
        tuple(int(segment.slot_id[0]) for segment in selected),
        tuple(segment.consumer_payload[0][index] for segment in selected),
        tuple(int(segment.consumer_step[0, index]) for segment in selected),
    )
