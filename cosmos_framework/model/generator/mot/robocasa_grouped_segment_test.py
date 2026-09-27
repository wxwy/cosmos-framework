# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""H3-B 18-task catalog、8-rank/8-slot/GA2 candidate 与同 index B1 生产测试。"""

from __future__ import annotations

import hashlib
from bisect import bisect_right
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import h5py
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from cosmos_framework.data.generator.action.datasets.robocasa_lerobot_dataset import DEFAULT_ALL_ATOMIC_TASKS
from cosmos_framework.data.generator.sequence_packing import SequencePlan
from cosmos_framework.model.generator.mot.robocasa_grouped_segment import (
    CatalogEpisode,
    EpisodeKey,
    RankLocalGroupedPlanner,
    RoboCasaEpisodeCatalog,
    StageARoboCasaEpisodeBinder,
    gather_same_index,
    materialize_member,
)
from cosmos_framework.model.generator.mot.robocasa_latent_evidence_test import read_cache, write_cache
from cosmos_framework.model.generator.mot.robocasa_segment_producer import RoboCasaSegmentProducer


def _catalog(*, frames: int = 64, episodes_per_task: int = 12) -> RoboCasaEpisodeCatalog:
    episodes = []
    for task in DEFAULT_ALL_ATOMIC_TASKS:
        for index in range(episodes_per_task):
            key = EpisodeKey(task, "20250816", index)
            episodes.append(
                CatalogEpisode(
                    key,
                    f"{task}/20250816/lerobot",
                    f"{task}/20250816/lerobot/{key.episode_id}.h5",
                    frames,
                    frames - 32,
                    0,
                    index * frames,
                    index * (frames - 32),
                    key.uid,
                )
            )
    return RoboCasaEpisodeCatalog(episodes, DEFAULT_ALL_ATOMIC_TASKS)


def _rank_catalog(count: int, *, frames: int) -> RoboCasaEpisodeCatalog:
    episodes = []
    index = 0
    while len(episodes) < count:
        key = EpisodeKey("Task", "date", index)
        if int.from_bytes(hashlib.sha256(key.uid.encode()).digest(), "big") % 8 == 0:
            episodes.append(
                CatalogEpisode(key, "Task/date/lerobot", f"{key.uid}.h5", frames, frames - 32, 0, 0, 0, key.uid)
            )
        index += 1
    return RoboCasaEpisodeCatalog(episodes, ("Task",))


def _fake_stage_a_dataset(source_root: Path, cache_root: Path, *, frames: int = 48):
    roots, records, ends = [], [], []
    for task in DEFAULT_ALL_ATOMIC_TASKS:
        shard = source_root / task / "20250816" / "lerobot"
        metadata = shard / "meta/episodes/chunk-000/file-000.parquet"
        metadata.parent.mkdir(parents=True)
        pq.write_table(
            pa.Table.from_pylist(
                [{"episode_index": 0, "length": frames, "dataset_from_index": 0, "dataset_to_index": frames}]
            ),
            metadata,
        )
        cache = cache_root / task / "20250816" / "lerobot" / "ep_000000.h5"
        cache.parent.mkdir(parents=True)
        with h5py.File(cache, "w") as handle:
            handle.attrs["episode_id"] = "ep_000000"
            handle.attrs["frame_count"] = frames
        roots.append(str(shard))
        records.append((len(roots) - 1, 0, frames - 32, 0))
        ends.append(len(records) * (frames - 32))

    def resolve(index: int):
        row = bisect_right(ends, index)
        source, start, _, episode = records[row]
        previous = 0 if row == 0 else ends[row - 1]
        return source, start + index - previous, episode, index - previous

    return SimpleNamespace(
        fps=20,
        chunk_length=32,
        split="train",
        action_dim=15,
        _split_seed=42,
        _split_val_ratio=0.01,
        _camera_set="left_wrist",
        _use_state=True,
        _use_base_action=True,
        _base_encoding="raw",
        _action_normalizer=None,
        _sample_stride=1,
        _all_shard_roots=roots,
        _episode_records=records,
        _episode_cum_ends=ends,
        _resolve_index=resolve,
    )


def test_exact_18_task_catalog_manifest_and_cache_metadata(tmp_path: Path) -> None:
    source, cache = tmp_path / "source", tmp_path / "cache"
    dataset = _fake_stage_a_dataset(source, cache)
    first = RoboCasaEpisodeCatalog.from_stage_a_dataset(dataset, source_root=source, cache_root=cache)
    second = RoboCasaEpisodeCatalog.from_stage_a_dataset(dataset, source_root=source, cache_root=cache)
    assert len(first.episodes) == len(DEFAULT_ALL_ATOMIC_TASKS) == 18
    assert first.manifest_digest == second.manifest_digest
    assert first.task_order == DEFAULT_ALL_ATOMIC_TASKS
    assert all(episode.valid_consumer_count == 16 and episode.segment_count == 1 for episode in first.episodes)
    assert len({episode.uid for episode in first.episodes}) == 18
    assert all(episode.cache_relative.endswith("/ep_000000.h5") for episode in first.episodes)


@pytest.mark.parametrize("failure", ["missing_cache", "wrong_frames", "wrong_loader", "duplicate_meta"])
def test_catalog_fails_closed_on_missing_or_mismatched_authority(tmp_path: Path, failure: str) -> None:
    source, cache = tmp_path / "source", tmp_path / "cache"
    dataset = _fake_stage_a_dataset(source, cache)
    path = cache / DEFAULT_ALL_ATOMIC_TASKS[0] / "20250816/lerobot/ep_000000.h5"
    if failure == "missing_cache":
        path.rename(path.with_suffix(".absent"))
        with pytest.raises(FileNotFoundError):
            RoboCasaEpisodeCatalog.from_stage_a_dataset(dataset, source_root=source, cache_root=cache)
        return
    if failure == "wrong_frames":
        with h5py.File(path, "r+") as handle:
            handle.attrs["frame_count"] = 47
    elif failure == "wrong_loader":
        dataset.action_dim = 12
    else:
        meta = source / DEFAULT_ALL_ATOMIC_TASKS[0] / "20250816/lerobot/meta/episodes/chunk-000/file-000.parquet"
        table = pq.read_table(meta)
        pq.write_table(pa.concat_tables((table, table)), meta)
    with pytest.raises(ValueError):
        RoboCasaEpisodeCatalog.from_stage_a_dataset(dataset, source_root=source, cache_root=cache)


def test_rank_partition_and_full_t16_ga2_window_are_deterministic() -> None:
    catalog = _catalog()
    parts = [set(episode.uid for episode in catalog.rank_episodes(rank)) for rank in range(8)]
    assert set.union(*parts) == set(catalog.by_uid)
    assert sum(map(len, parts)) == len(catalog.episodes)
    assert all(parts[left].isdisjoint(parts[right]) for left in range(8) for right in range(left + 1, 8))
    for rank in range(8):
        planner = RankLocalGroupedPlanner(catalog, rank=rank)
        frontier = planner.initial_frontier()
        plan = planner.plan_window(frontier)
        assert planner.plan_window(frontier) == plan
        assert frontier == planner.initial_frontier()
        assert plan.member_counts == (128, 128) and plan.n_window == 256
        assert len(plan.members) == 2
        assert all(len(member) == 8 for member in plan.members)
        assert tuple(request.identity.slot_id for request in plan.members[0]) == tuple(range(rank * 8, rank * 8 + 8))
        assert all(
            request.identity.cursor == member for member, requests in enumerate(plan.members) for request in requests
        )
        assert len({request.episode.uid for request in plan.members[0]}) == 8


def test_terminal_rebind_in_second_member_is_fresh_s0_and_rollback_safe() -> None:
    catalog = _catalog(frames=48)
    planner = RankLocalGroupedPlanner(catalog, rank=0)
    frontier = planner.initial_frontier()
    plan = planner.plan_window(frontier)
    assert plan.n_window == 256
    assert all(request.identity.training_stream_end for member in plan.members for request in member)
    assert all(request.identity.cursor == 0 for member in plan.members for request in member)
    assert len({request.episode.uid for member in plan.members for request in member}) == 16
    assert all(
        request.identity.segment_id == member for member, requests in enumerate(plan.members) for request in requests
    )
    assert frontier == planner.initial_frontier()
    assert all(slot.uid is None and slot.next_segment_id == 2 for slot in plan.candidate_frontier.slots)


def test_terminal_remainder_preserves_t16_and_true_counts() -> None:
    planner = RankLocalGroupedPlanner(_catalog(frames=49), rank=0)
    plan = planner.plan_window(planner.initial_frontier())
    assert plan.member_counts == (128, 8) and plan.n_window == 136
    assert all(request.valid_count == 1 and request.identity.cursor == 1 for request in plan.members[1])
    assert all(request.identity.training_stream_end for request in plan.members[1])


def test_nonmultiple_epoch_rollover_retains_unique_window_and_frontier(tmp_path: Path) -> None:
    catalog = _rank_catalog(17, frames=48)
    planner = RankLocalGroupedPlanner(catalog, rank=0)
    first = planner.plan_window(planner.initial_frontier())
    assert first.n_window == 256 and first.candidate_frontier.epoch == 0
    second = planner.plan_window(first.candidate_frontier)
    assert second.n_window == 256 and second.candidate_frontier.epoch == 1
    assert len({request.episode.uid for member in second.members for request in member}) == 16
    assert first.candidate_frontier.epoch == 0
    assert planner.plan_window(first.candidate_frontier) == second


def test_long_epoch_chronology_has_no_duplicate_or_skipped_binding() -> None:
    catalog = _rank_catalog(17, frames=48)
    planner = RankLocalGroupedPlanner(catalog, rank=0)
    frontier = planner.initial_frontier()
    assigned: dict[int, list[str]] = {}
    for _ in range(5):
        plan = planner.plan_window(frontier)
        for member in plan.members:
            for request in member:
                assert request.identity.cursor == 0 and request.identity.training_stream_end
                assigned.setdefault(request.binding_epoch, []).append(request.episode.uid)
        frontier = plan.candidate_frontier
    assert len(assigned[0]) == 17 and set(assigned[0]) == set(catalog.by_uid)
    assert all(len(uids) == len(set(uids)) for uids in assigned.values())
    assert all(plan.member_counts == (128, 128) for plan in (planner.plan_window(frontier),))


def test_tampered_frontier_cannot_duplicate_active_episode() -> None:
    planner = RankLocalGroupedPlanner(_catalog(frames=80), rank=0)
    first = planner.plan_window(planner.initial_frontier()).candidate_frontier
    bad_slots = list(first.slots)
    bad_slots[1] = bad_slots[0]
    with pytest.raises(ValueError, match="重复绑定"):
        planner.plan_window(replace(first, slots=tuple(bad_slots)))


def test_same_index_materializes_b1_segments_without_mixing_source(tmp_path: Path) -> None:
    planner = RankLocalGroupedPlanner(_catalog(frames=48), rank=0)
    plan = planner.plan_window(planner.initial_frontier())
    producers = {}

    def producer_for(episode: CatalogEpisode) -> RoboCasaSegmentProducer:
        if episode.uid not in producers:
            path = tmp_path / f"{episode.uid.replace('/', '_')}.h5"
            reader = read_cache(
                write_cache(path, episode.source_frames, episode.key.episode_id),
                episode.source_frames,
                episode.key.episode_id,
            )
            raw15 = torch.zeros(episode.source_frames, 15)
            producers[episode.uid] = RoboCasaSegmentProducer(
                reader,
                episode_id=episode.key.episode_id,
                category=episode.key.task,
                raw15=raw15,
                payload_at=lambda step, uid=episode.uid: {"uid": uid, "step": step, "rgb_frames": 33},
                manifest_digest=planner.catalog.manifest_digest,
                config_digest="stage-a",
                source_digest=episode.source_digest,
            )
        return producers[episode.uid]

    segments = materialize_member(plan.members[0], producer_for)
    assert len(segments) == 8 and len({segment.segment_provenance.source_digest for segment in segments}) == 8
    for index in range(16):
        batch = gather_same_index(segments, index)
        assert batch.index == index and len(batch.payloads) == len(batch.slot_ids) == 8
        assert batch.slot_ids == tuple(range(8))
        assert batch.consumer_steps == (index,) * 8
        assert all(payload["step"] == index and payload["rgb_frames"] == 33 for payload in batch.payloads)
        for segment in segments:
            if index == 0:
                assert not segment.evidence_valid[0, index]
            else:
                assert segment.evidence_source_step[0, index] == index - 1

    def wrong_category(episode: CatalogEpisode) -> RoboCasaSegmentProducer:
        producer = producer_for(episode)
        producer.category = "wrong"
        return producer

    with pytest.raises(ValueError):
        materialize_member(plan.members[0], wrong_category)


def test_materialization_exception_never_advances_catalog_frontier(tmp_path: Path) -> None:
    planner = RankLocalGroupedPlanner(_catalog(frames=48), rank=0)
    frontier = planner.initial_frontier()
    plan = planner.plan_window(frontier)

    def fail(episode: CatalogEpisode):
        raise OSError(f"decoder fail: {episode.uid}")

    with pytest.raises(OSError, match="decoder fail"):
        materialize_member(plan.members[0], fail)
    assert frontier == planner.initial_frontier()
    assert planner.plan_window(frontier) == plan


def test_stage_a_binder_reuses_raw15_rgb_and_exact_cache(tmp_path: Path) -> None:
    frames = 48
    key = EpisodeKey("Task", "date", 0)
    shard = tmp_path / "source/Task/date/lerobot"
    shard.mkdir(parents=True)
    cache = tmp_path / "cache/Task/date/lerobot"
    cache.mkdir(parents=True)
    write_cache(cache / "ep_000000.h5", frames, key.episode_id)
    episode = CatalogEpisode(key, "Task/date/lerobot", "Task/date/lerobot/ep_000000.h5", frames, 16, 0, 0, 0, key.uid)
    catalog = RoboCasaEpisodeCatalog((episode,), ("Task",))

    class FakeDataset:
        _all_shard_roots = [str(shard)]

        def _get_dataset(self, source: int):
            assert source == 0

            class Rows:
                def __getitem__(self, span: slice):
                    assert span == slice(0, frames)
                    return {
                        "episode_index": [0] * frames,
                        "frame_index": list(range(frames)),
                        "action": [torch.zeros(12) for _ in range(frames)],
                    }

            return SimpleNamespace(hf_dataset=Rows())

        def _build_frame_wise_action(self, raw12: torch.Tensor) -> torch.Tensor:
            assert raw12.shape == (frames, 12)
            return torch.zeros(frames, 10)

        def _resolve_index(self, flat: int):
            return 0, flat, 0, flat

        def __getitem__(self, flat: int):
            return {"action": torch.zeros(33, 15), "video": torch.zeros(3, 33, 4, 8), "step": flat}

    def transform(payload, resolution):
        assert resolution is None
        action = payload["action"]
        payload.update(
            action_raw=action.clone(),
            action=torch.nn.functional.pad(action, (0, 49)),
            raw_action_dim=15,
            text_token_ids=torch.tensor([1, 2], dtype=torch.long),
            sequence_plan=SequencePlan(has_text=True, has_vision=True, has_action=True),
        )
        return payload

    binder = StageARoboCasaEpisodeBinder(
        FakeDataset(),
        catalog,
        source_root=tmp_path / "source",
        cache_root=tmp_path / "cache",
        transform=transform,
        resolution=None,
        config_digest="stage-a",
    )
    producer = binder.producer_for(episode)
    assert binder.producer_for(episode) is producer
    identity = producer.identity(slot_id=0, cursor=0, segment_id=0)
    segment = producer.produce(identity)
    segment.validate(16)
    assert segment.consumer_valid.sum() == 16
    assert segment.consumer_payload[0][0]["action"].shape == (33, 64)
    assert segment.consumer_payload[0][0]["video"].shape == (3, 33, 4, 8)
    assert segment.consumer_step[0].tolist() == list(range(16))
