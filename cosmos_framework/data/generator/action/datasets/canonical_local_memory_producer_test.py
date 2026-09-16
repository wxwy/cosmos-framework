# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CPU-only contract tests for the canonical Local-Memory segment producer."""

from __future__ import annotations

from typing import Any

import pytest
import torch
import torch.nn.functional as F

from cosmos_framework.data.generator.action.datasets.canonical_local_memory_producer import (
    CanonicalLocalMemorySegmentProducer,
    CanonicalSegmentStream,
)

_CHUNK = 16
_STEPS = 16
_EPISODE_FRAMES = 48  # -> 32 valid window anchors -> 2 whole blocks


class _FakeFrameSource:
    """Minimal stand-in for ``LIBEROLeRobotDataset``'s evidence surface."""

    def __init__(self, *, frames: int = _EPISODE_FRAMES, episodes: int = 2) -> None:
        self._latent_cache_root = "/nonexistent"
        self._chunk_length = _CHUNK
        self.action_normalization = None  # normalization itself is covered by _build_local_history's own tests
        self._ep_vals = torch.arange(episodes, dtype=torch.long).numpy()
        self._ep_starts = torch.arange(episodes, dtype=torch.long).numpy() * frames
        per_episode = frames - _CHUNK
        self._valid_cum = torch.arange(1, episodes + 1, dtype=torch.long).numpy() * per_episode
        self._row_action = torch.arange((episodes * frames) * 7, dtype=torch.float32).reshape(-1, 7)
        self.built: list[int] = []

    @property
    def action_dim(self) -> int:
        return 10

    def _load_cached_latent(self, episode_index: int, local_frame: int) -> torch.Tensor:
        # latent[0] is a deterministic [48, H, W] ramp keyed by (episode, frame).
        base = float(episode_index * 1000 + local_frame)
        return torch.full((5, 48, 2, 2), base, dtype=torch.float32)

    def _build_frame_wise_action(self, raw: torch.Tensor) -> torch.Tensor:
        return raw[:, :1].repeat(1, 10)  # [n, 10]

    def _load_norm_stats(self) -> dict[str, Any]:
        return {}

    def _build_item(self, idx: int) -> dict[str, Any]:
        self.built.append(idx)
        return {"flat_index": idx, "episode_index": torch.tensor(0, dtype=torch.long)}


class _FakeWrappedDataset:
    """Minimal stand-in for ``ActionSFTDataset``."""

    def __init__(self, frame_source: _FakeFrameSource) -> None:
        self._dataset = frame_source
        self._resolution = "256"
        self.transformed: list[int] = []

    def _transform(self, item: dict[str, Any], resolution: Any) -> dict[str, Any]:
        self.transformed.append(int(item["flat_index"]))
        return {**item, "resolution": resolution, "sequence_plan": object()}


@pytest.fixture(name="producer")
def fixture_producer() -> tuple[CanonicalLocalMemorySegmentProducer, _FakeFrameSource]:
    frame_source = _FakeFrameSource()
    producer = CanonicalLocalMemorySegmentProducer(
        _FakeWrappedDataset(frame_source),
        category="libero_10",
        ttt_tbptt_steps=_STEPS,
        manifest_digest="manifest",
        config_digest="config",
        source_digest="source",
    )
    return producer, frame_source


def _stream(slot_id: int = 0, episode_position: int = 0) -> CanonicalSegmentStream:
    return CanonicalSegmentStream(
        slot_id=slot_id, episode_index=episode_position, episode_position=episode_position, category="libero_10"
    )


def test_producer_geometry_matches_valid_window_anchors(producer) -> None:
    segment_producer, _ = producer
    stream = _stream()
    assert segment_producer.valid_start_count(stream) == _EPISODE_FRAMES - _CHUNK
    assert segment_producer.block_count(stream) == 2


def test_fresh_block_exposes_one_local_neutral_s0(producer) -> None:
    segment_producer, frame_source = producer
    stream = _stream()
    segment = segment_producer.produce(stream, cursor=0)

    assert segment.consumer_step.tolist() == [list(range(_STEPS))]
    assert segment.consumer_valid.all()
    assert segment.evidence_valid.tolist() == [[False, *([True] * (_STEPS - 1))]]
    assert segment.evidence_source_step.tolist() == [[-1, *range(_STEPS - 1)]]
    # Evidence absent exactly at global step 0, whose tensors stay zeroed.
    assert torch.equal(segment.evidence_visual_summary_prev[0, 0], torch.zeros(96))
    assert torch.equal(segment.evidence_executed_action_prev[0, 0], torch.zeros(10))
    # Consumer t is the sample anchored at episode-local frame t.
    assert frame_source.built == list(range(_STEPS))


def test_continuation_block_keeps_global_steps_and_full_evidence(producer) -> None:
    segment_producer, frame_source = producer
    stream = _stream()
    segment = segment_producer.produce(stream, cursor=1)

    assert segment.consumer_step.tolist() == [[*range(_STEPS, 2 * _STEPS)]]
    assert segment.evidence_valid.all()
    assert segment.evidence_source_step.tolist() == [[*range(_STEPS - 1, 2 * _STEPS - 1)]]
    assert frame_source.built == list(range(_STEPS, 2 * _STEPS))


def test_visual_summary_and_action_reuse_the_dataset_evidence_recipe(producer) -> None:
    segment_producer, frame_source = producer
    stream = _stream()
    segment = segment_producer.produce(stream, cursor=1)

    latent = frame_source._load_cached_latent(stream.episode_index, _STEPS)
    expected = F.adaptive_avg_pool2d(latent[0].unsqueeze(0), output_size=(1, 2)).flatten()
    torch.testing.assert_close(segment.consumer_visual_summary[0, 0], expected)
    # Evidence at index t is the frame immediately before consumer t.
    previous = frame_source._load_cached_latent(stream.episode_index, _STEPS - 1)
    torch.testing.assert_close(
        segment.evidence_visual_summary_prev[0, 0],
        F.adaptive_avg_pool2d(previous[0].unsqueeze(0), output_size=(1, 2)).flatten(),
    )
    expected_action = frame_source._build_frame_wise_action(
        frame_source._row_action[frame_source._ep_starts[0] + _STEPS - 1 :][:1]
    )[0]
    torch.testing.assert_close(segment.evidence_executed_action_prev[0, 0], expected_action)


def test_payloads_carry_the_transform_output_of_their_own_frame(producer) -> None:
    segment_producer, _ = producer
    stream = _stream()
    segment = segment_producer.produce(stream, cursor=0)
    payloads = segment.consumer_payload[0]
    assert [payload["flat_index"] for payload in payloads] == list(range(_STEPS))
    assert all(payload["resolution"] == "256" for payload in payloads)
    assert all("sequence_plan" in payload for payload in payloads)


def test_producer_rejects_blocks_past_the_episode_end(producer) -> None:
    segment_producer, _ = producer
    stream = _stream()
    with pytest.raises(ValueError, match="runs past the end of its episode"):
        segment_producer.produce(stream, cursor=2)
    # A trailing partial block cannot be widened past the frozen TBPTT bound.
    with pytest.raises(ValueError, match="width must be in"):
        segment_producer.produce(stream, cursor=1, steps=_STEPS + 1)


def test_producer_rejects_foreign_stream_identity(producer) -> None:
    segment_producer, _ = producer
    with pytest.raises(ValueError, match="category differs"):
        segment_producer.produce(
            CanonicalSegmentStream(slot_id=0, episode_index=0, episode_position=0, category="libero_object"),
            cursor=0,
        )
    with pytest.raises(ValueError, match="disagrees with its dataset position"):
        segment_producer.produce(
            CanonicalSegmentStream(slot_id=0, episode_index=1, episode_position=0, category="libero_10"), cursor=0
        )
