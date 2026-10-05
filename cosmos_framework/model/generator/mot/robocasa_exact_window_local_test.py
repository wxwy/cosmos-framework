"""Phase4A debug-only CPU contracts for exact-window Local evidence and planning."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from cosmos_framework.data.generator.action.datasets.action_sft_dataset import ActionSFTDataset
from cosmos_framework.data.generator.action.datasets.robocasa_exact_window_cache import (
    ExactWindowEpisodeKey,
    ExactWindowEpisodeRecord,
    RoboCasaExactWindowCacheCatalog,
)
from cosmos_framework.data.generator.action.datasets.robocasa_exact_window_cached_sft import (
    RoboCasaExactWindowCachedDataset,
)
from cosmos_framework.data.generator.action.datasets.robocasa_exact_window_policy import CorrectedRoboCasaPolicyContract
from cosmos_framework.data.generator.action.datasets.robocasa_exact_window_source import RoboCasaExactWindowSourceReader
from cosmos_framework.data.generator.action.utils.transforms import ActionTransformPipeline
from cosmos_framework.data.generator.joint_dataloader import custom_collate_fn
from cosmos_framework.model.generator.mot.robocasa_exact_window_local import (
    ExactWindowLocalCatalog,
    ExactWindowRankPlanner,
    ExactWindowSegmentProducer,
    gather_exact_window_same_index,
    robocasa_current_latent_to_visual96,
)


class _FakeRaw(RoboCasaExactWindowCachedDataset):
    def __init__(self, counts: tuple[int, ...], tasks: tuple[str, ...] | None = None):
        tasks = tasks or ("Pick Mug",) * len(counts)
        self.records = tuple(
            ExactWindowEpisodeRecord(
                ExactWindowEpisodeKey(task, task.replace(" ", "_"), index),
                count,
                count + 16,
                Path(f"episode_{index:06d}.pt"),
            )
            for index, (task, count) in enumerate(zip(tasks, counts, strict=True))
        )
        self.catalog = SimpleNamespace(
            episodes=self.records,
            corpus_digest="corpus",
            manifest_sha256="manifest",
            latent_shape=(5, 48, 2, 2),
        )
        self.blocks = []
        flat = 0
        for count in counts:
            self.blocks.append((flat, count))
            flat += count
        self.source_reader = SimpleNamespace(
            source_binding_digest="binding",
            _bound={
                record.key: SimpleNamespace(
                    first_rows=tuple(range(self.blocks[index][0] * 100, self.blocks[index][0] * 100 + 17)),
                    terminal_rows=tuple(
                        range(
                            self.blocks[index][0] * 100 + record.window_count - 1,
                            self.blocks[index][0] * 100 + record.window_count + 16,
                        )
                    ),
                )
                for index, record in enumerate(self.records)
            },
        )
        self.calls: list[int] = []

    def __len__(self):
        return sum(record.window_count for record in self.records)

    def get_shuffle_blocks(self):
        return list(self.blocks)

    def __getitem__(self, index):
        self.calls.append(index)
        if not 0 <= index < len(self):
            raise IndexError(index)
        record = next(
            record
            for record, (flat, count) in zip(self.records, self.blocks, strict=True)
            if flat <= index < flat + count
        )
        step = index - self.blocks[record.key.episode_index][0]
        latent = torch.zeros((5, 48, 2, 2), dtype=torch.float32)
        latent[0] = float(step + 1)
        latent[1:] = 999.0
        action = torch.full((17, 15), 777.0)
        action[0] = -100.0
        action[1] = float(step + 10)
        return {
            "video_latent": latent,
            "cached_latent_required": True,
            "action": action,
            "task_class": record.key.task_class,
            "episode_index": record.key.episode_index,
            "start_frame": step,
            "cache_corpus_digest": self.catalog.corpus_digest,
            "source_binding_digest": self.source_reader.source_binding_digest,
        }


class _Transform:
    def __init__(self):
        self.calls: list[int] = []

    def __call__(self, item, resolution):
        self.calls.append(item["start_frame"])
        item["action_raw"] = item["action"]
        item["action"] = F.pad(item["action"], (0, 49))
        return item


def _setup(counts=(33,), tasks=None, *, t=16):
    raw = _FakeRaw(counts, tasks)
    transform = _Transform()
    wrapped = ActionSFTDataset(raw, transform, resolution=None)
    catalog = ExactWindowLocalCatalog(wrapped, ttt_tbptt_steps=t)
    return raw, transform, catalog, ExactWindowSegmentProducer(catalog, config_digest="config")


def _request(catalog, *, cursor=0, slot=0, episode_index=0, segment_id=None):
    from cosmos_framework.model.generator.mot.local_memory_segment import SegmentIdentity
    from cosmos_framework.model.generator.mot.robocasa_exact_window_local import ExactWindowSegmentRequest

    episode = catalog.episodes[episode_index]
    count = min(catalog.ttt_tbptt_steps, episode.window_count - cursor * catalog.ttt_tbptt_steps)
    identity = SegmentIdentity(
        slot,
        episode.uid,
        episode.task_class,
        cursor,
        cursor if segment_id is None else segment_id,
        episode.source_digest,
        cursor == episode.segment_count - 1,
    )
    return ExactWindowSegmentRequest(episode, identity, 0, count)


def test_visual96_exact_v2_formula_and_gradient():
    z0 = torch.arange(48 * 3 * 5, dtype=torch.float32).reshape(48, 3, 5).requires_grad_()
    result = robocasa_current_latent_to_visual96(z0)
    torch.testing.assert_close(result, F.adaptive_avg_pool2d(z0.unsqueeze(0), (1, 2)).flatten(), rtol=0, atol=0)
    assert result.shape == (96,)
    result.sum().backward()
    assert z0.grad is not None and bool(z0.grad.any())


@pytest.mark.parametrize(
    "bad", [torch.zeros(47, 2, 2), torch.zeros(48, 2, 2).half(), torch.full((48, 2, 2), float("nan"))]
)
def test_visual96_rejects_wrong_channel_dtype_or_nonfinite(bad):
    with pytest.raises(ValueError):
        robocasa_current_latent_to_visual96(bad)


def test_catalog_requires_phase3_raw_capability_and_path_independent_digest():
    raw, _, catalog, _ = _setup((33,))
    assert catalog.episodes[0].window_count == 33
    assert catalog.episodes[0].segment_count == 3
    assert catalog.task_classes == ("Pick Mug",)
    assert raw.calls == []
    first_digest = catalog.episodes[0].source_digest
    raw.catalog.cache_root = Path("/other/machine/cache")
    assert ExactWindowLocalCatalog(catalog.wrapped_sft, ttt_tbptt_steps=16).episodes[0].source_digest == first_digest
    raw.source_reader.source_binding_digest = "changed"
    assert ExactWindowLocalCatalog(catalog.wrapped_sft, ttt_tbptt_steps=16).episodes[0].source_digest != first_digest
    raw.source_reader.source_binding_digest = "binding"
    raw.source_reader._bound[raw.records[0].key].first_rows = tuple(range(200, 217))
    assert ExactWindowLocalCatalog(catalog.wrapped_sft, ttt_tbptt_steps=16).episodes[0].source_digest != first_digest
    with pytest.raises(TypeError):
        ExactWindowLocalCatalog(ActionSFTDataset([], _Transform(), None), ttt_tbptt_steps=16)


@pytest.mark.parametrize("t,count,expected_segments", [(16, 33, 3), (32, 33, 2), (16, 2, 1)])
def test_geometry_s0_previous_action_and_terminal_pad(t, count, expected_segments):
    raw, transform, catalog, producer = _setup((count,), t=t)
    assert catalog.episodes[0].segment_count == expected_segments
    first = producer.produce(_request(catalog))
    assert first.consumer_valid.shape == (1, t)
    assert first.evidence_valid[0, 0].item() is False
    assert first.evidence_source_step[0, 0].item() == -1
    assert first.consumer_payload[0][0]["video_latent"].shape == (5, 48, 2, 2)
    assert first.consumer_payload[0][0]["cached_latent_required"] is True
    assert first.consumer_payload[0][0]["action"].shape == (17, 64)
    assert first.consumer_payload[0][0]["action_raw"].shape == (17, 15)
    if count > 1:
        torch.testing.assert_close(first.evidence_visual_summary_prev[0, 1], torch.ones(96))
        torch.testing.assert_close(first.evidence_executed_action_prev[0, 1], torch.full((15,), 10.0))
        assert not torch.equal(first.evidence_executed_action_prev[0, 1], first.consumer_payload[0][0]["action_raw"][0])
    assert raw.calls == list(range(min(count, t)))
    assert transform.calls == list(range(min(count, t)))
    before_tail = len(raw.calls)
    tail = producer.produce(_request(catalog, cursor=expected_segments - 1))
    valid = min(t, count - (expected_segments - 1) * t)
    tail_start = (expected_segments - 1) * t
    assert raw.calls[before_tail:] == list(range(max(0, tail_start - 1), tail_start + valid))
    assert tail.consumer_valid[0].tolist() == [True] * valid + [False] * (t - valid)
    assert tail.consumer_payload[0][valid:] == (None,) * (t - valid)
    assert tail.evidence_source_step[0, valid:].tolist() == [-1] * (t - valid)


def test_produce_deduplicates_previous_and_current_raw_reads():
    raw, transform, catalog, producer = _setup((33,))
    segment = producer.produce(_request(catalog, cursor=1))
    assert raw.calls == list(range(15, 32))
    assert transform.calls == list(range(16, 32))
    assert len(raw.calls) == len(set(raw.calls))
    torch.testing.assert_close(segment.evidence_visual_summary_prev[0, 0], torch.full((96,), 16.0))
    torch.testing.assert_close(segment.evidence_executed_action_prev[0, 0], torch.full((15,), 25.0))
    assert segment.evidence_source_step[0].tolist() == list(range(15, 31))


def test_producer_rejects_identity_and_digest_drift():
    from dataclasses import replace

    raw, _, catalog, producer = _setup((33,))
    request = _request(catalog)
    with pytest.raises(ValueError, match="identity"):
        producer.produce(replace(request, identity=replace(request.identity, source_digest="wrong")))
    raw.source_reader.source_binding_digest = "changed"
    with pytest.raises(ValueError, match="identity"):
        producer.produce(request)


def test_planner_dynamic_b_ga_stable_slots_and_candidate_only():
    _, _, catalog, _ = _setup((33, 33, 33, 33), ("A", "B", "A", "B"), t=16)
    planner = ExactWindowRankPlanner(catalog, rank=0, world_size=1, b_stream=2, active_ga=3, seed=7)
    live = planner.initial_frontier()
    plan = planner.plan_window(live)
    assert len(plan.members) == 3
    assert all(len(member) == 2 for member in plan.members)
    assert all(tuple(request.identity.slot_id for request in member) == (0, 1) for member in plan.members)
    assert plan.member_counts == (32, 32, 2)
    assert plan.n_window == 66
    assert all(plan.members[0][row].episode.uid == plan.members[1][row].episode.uid for row in range(2))
    assert all(plan.members[2][row].identity.training_stream_end for row in range(2))
    assert all(slot.uid is None for slot in plan.candidate_frontier.slots)
    assert all(slot.uid is None for slot in live.slots)
    next_plan = planner.plan_window(plan.candidate_frontier)
    assert {request.episode.uid for request in next_plan.members[0]}.isdisjoint(
        {request.episode.uid for request in plan.members[0]}
    )


def test_rank_partition_deterministic_and_capacity_fail_closed():
    _, _, catalog, _ = _setup((33,) * 10)
    first = ExactWindowRankPlanner(catalog, rank=0, world_size=2, b_stream=2, active_ga=1)
    second = ExactWindowRankPlanner(catalog, rank=0, world_size=2, b_stream=2, active_ga=1)
    assert tuple(item.uid for item in first.episodes) == tuple(item.uid for item in second.episodes)
    assert all(int(__import__("hashlib").sha256(item.uid.encode()).hexdigest(), 16) % 2 == 0 for item in first.episodes)
    with pytest.raises(ValueError, match="少于"):
        ExactWindowRankPlanner(catalog, rank=0, world_size=1, b_stream=11)


def test_planner_terminal_rebind_within_one_window_never_reuses_episode():
    _, _, catalog, _ = _setup((1, 1, 1, 1), ("A", "A", "B", "B"), t=16)
    planner = ExactWindowRankPlanner(catalog, rank=0, world_size=1, b_stream=2, active_ga=2)
    live = planner.initial_frontier()
    plan = planner.plan_window(live)
    first = {request.episode.uid for request in plan.members[0]}
    second = {request.episode.uid for request in plan.members[1]}
    assert len(first) == len(second) == 2 and first.isdisjoint(second)
    assert all(request.identity.training_stream_end for member in plan.members for request in member)
    assert plan.member_counts == (2, 2)
    assert all(slot.uid is None and slot.next_segment_id == 2 for slot in plan.candidate_frontier.slots)
    assert all(slot.next_segment_id == 0 for slot in live.slots)


def test_same_index_gather_respects_pad_and_slot_order():
    from dataclasses import replace

    _, _, catalog, producer = _setup((17, 16), t=16)
    left = producer.produce(_request(catalog, cursor=1, episode_index=0, slot=0))
    right = producer.produce(_request(catalog, cursor=0, episode_index=1, slot=1))
    batch = gather_exact_window_same_index((left, right), 0)
    assert batch.slot_ids == (0, 1) and batch.consumer_steps == (16, 0)
    assert len(batch.payloads) == 2
    assert gather_exact_window_same_index((left, right), 1).slot_ids == (1,)
    with pytest.raises(ValueError, match="排序"):
        gather_exact_window_same_index((right, left), 0)
    with pytest.raises(ValueError, match="排序"):
        gather_exact_window_same_index((left, replace(right, slot_id=torch.tensor([0]))), 0)


def test_corrected_module_has_no_historical_b1_imports():
    module = Path(__file__).with_name("robocasa_exact_window_local.py").read_text()
    for banned in (
        "robocasa_latent_evidence",
        "RoboCasaLatentReader",
        "StageARoboCasaEpisodeBinder",
        "RoboCasaSegmentProducer",
        "CatalogFrontier",
        "RankLocalGroupedPlanner",
    ):
        assert f"import {banned}" not in module


@pytest.mark.skipif(
    not (os.environ.get("PSM_PHASE4A_CACHE_ROOT") and os.environ.get("PSM_PHASE4A_SOURCE_ROOT")),
    reason="real debug cache/source paths were not supplied",
)
def test_optional_real_data_debug_smoke(monkeypatch: pytest.MonkeyPatch):
    """Read a bounded real segment only when the caller supplies local asset paths."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    cache = RoboCasaExactWindowCacheCatalog(Path(os.environ["PSM_PHASE4A_CACHE_ROOT"]))
    source = RoboCasaExactWindowSourceReader(cache, Path(os.environ["PSM_PHASE4A_SOURCE_ROOT"]))
    contract = CorrectedRoboCasaPolicyContract.from_cache_catalog(cache)
    raw = RoboCasaExactWindowCachedDataset(cache, source, contract)
    transform = ActionTransformPipeline(
        tokenizer_config=None,
        cfg_dropout_rate=0.0,
        max_action_dim=contract.max_action_dim,
        append_viewpoint_info=True,
        append_duration_fps_timestamps=True,
        append_resolution_info=True,
        append_idle_frames=True,
        format_prompt_as_json=True,
    )
    catalog = ExactWindowLocalCatalog(ActionSFTDataset(raw, transform, None), ttt_tbptt_steps=16)
    planner = ExactWindowRankPlanner(catalog, rank=0, world_size=1, b_stream=2, active_ga=1)
    request = planner.plan_window(planner.initial_frontier()).members[0][0]
    segment = ExactWindowSegmentProducer(catalog, config_digest="phase4a-debug").produce(request)
    assert segment.consumer_valid.shape == (1, 16)
    assert int(segment.evidence_valid.sum()) == 15
    assert segment.consumer_payload[0][0]["video_latent"].shape == cache.latent_shape
    assert segment.consumer_payload[0][0]["cached_latent_required"] is True
    assert segment.consumer_payload[0][0]["action_raw"].shape == (17, 15)
    torch.testing.assert_close(
        segment.evidence_visual_summary_prev[0, 1],
        robocasa_current_latent_to_visual96(segment.consumer_payload[0][0]["video_latent"][0]),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        segment.evidence_executed_action_prev[0, 1],
        segment.consumer_payload[0][0]["action_raw"][1],
        rtol=0,
        atol=0,
    )
    previous_policy = raw.policy_adapter.convert(source.read_at(request.episode.flat_start))
    torch.testing.assert_close(segment.evidence_executed_action_prev[0, 1], previous_policy.action15[0], rtol=0, atol=0)
    assert custom_collate_fn([segment.consumer_payload[0][0]])["video_latent"].shape == (
        1,
        *cache.latent_shape,
    )
