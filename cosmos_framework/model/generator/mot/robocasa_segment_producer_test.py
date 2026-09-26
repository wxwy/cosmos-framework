"""真实 H5 合成 fixture 到 B0 SegmentBatch；不执行 policy/encoder/trainer。"""

from dataclasses import replace

import h5py
import pytest
import torch

from cosmos_framework.model.generator.mot.robocasa_latent_evidence_test import read_cache, write_cache
from cosmos_framework.model.generator.mot.robocasa_segment_producer import RoboCasaSegmentProducer


def make_producer(tmp_path, frames=67, episode="CloseFridge/shard0/episode0", raw15=None, payload_at=None):
    reader = read_cache(write_cache(tmp_path / "episode.h5", frames, episode), frames, episode)
    action = torch.arange(frames * 15, dtype=torch.float32).reshape(frames, 15) if raw15 is None else raw15
    payloads = [
        {"rgb_frames": tuple(range(step, step + 33)), "policy_chunk_length": 32} for step in range(max(0, frames - 32))
    ]
    producer = RoboCasaSegmentProducer(
        reader,
        episode_id=episode,
        category="CloseFridge",
        raw15=action,
        payload_at=payloads.__getitem__ if payload_at is None else payload_at,
        manifest_digest="manifest",
        config_digest="config",
        source_digest="source",
    )
    return producer, action, payloads


@pytest.mark.parametrize("cursor,count,terminal", [(0, 16, False), (1, 16, False), (2, 3, True)])
def test_cursor_and_terminal_remainder_exact_causal_contract(tmp_path, cursor, count, terminal):
    producer, raw15, payloads = make_producer(tmp_path)
    identity = producer.identity(slot_id=7, cursor=cursor, segment_id=100 + cursor)
    assert identity.training_stream_end is terminal
    segment = producer.produce(identity)
    segment.validate(16)
    assert segment.consumer_valid.shape == (1, 16)
    assert segment.evidence_visual_summary_prev.shape == (1, 16, 96)
    assert segment.evidence_executed_action_prev.shape == (1, 16, 15)
    assert segment.consumer_valid.sum() == count
    assert segment.slot_id.tolist() == [7]
    assert segment.episode_id == (producer.episode_id,)
    assert segment.segment_provenance.segment_id == 100 + cursor
    assert segment.consumer_step[0, :count].tolist() == list(range(cursor * 16, cursor * 16 + count))
    for offset in range(16):
        if offset >= count:
            assert segment.consumer_payload[0][offset] is None
            assert not segment.evidence_valid[0, offset]
            assert segment.evidence_source_step[0, offset] == -1
            assert torch.count_nonzero(segment.evidence_executed_action_prev[0, offset]) == 0
            continue
        step = cursor * 16 + offset
        assert segment.consumer_payload[0][offset] is payloads[step]
        assert payloads[step]["policy_chunk_length"] == 32
        assert len(payloads[step]["rgb_frames"]) == 33
        assert payloads[step]["rgb_frames"][-1] < producer.reader.source_frames
        assert "video_latent" not in payloads[step]
        if step == 0:
            assert not segment.evidence_valid[0, offset]
            assert segment.evidence_source_step[0, offset] == -1
            assert torch.count_nonzero(segment.evidence_visual_summary_prev[0, offset]) == 0
        else:
            assert segment.evidence_source_step[0, offset] == step - 1
            torch.testing.assert_close(
                segment.evidence_executed_action_prev[0, offset], raw15[step - 1], rtol=0, atol=0
            )
            endpoint = 4 * ((step - 1) // 4)
            assert endpoint <= step - 1
            torch.testing.assert_close(
                segment.evidence_visual_summary_prev[0, offset, :48], torch.full((48,), float(endpoint))
            )


def test_action_not_snapped_and_current_visual_not_previous_authority(tmp_path):
    producer, raw15, _ = make_producer(tmp_path)
    segment = producer.produce(producer.identity(slot_id=0, cursor=0, segment_id=0))
    assert segment.consumer_visual_summary[0, 4, 0] == 4
    assert segment.evidence_visual_summary_prev[0, 4, 0] == 0
    torch.testing.assert_close(segment.evidence_executed_action_prev[0, 4], raw15[3])
    assert not torch.equal(segment.evidence_executed_action_prev[0, 4], raw15[0])


@pytest.mark.parametrize("frames,valid", [(33, 1), (48, 16), (49, 17), (64, 32)])
def test_policy_horizon_and_terminal_geometry(tmp_path, frames, valid):
    producer, _, _ = make_producer(tmp_path, frames=frames)
    assert producer.policy_chunk_length == 32
    assert producer.policy_consumer_frames == 33
    assert producer.ttt_tbptt_steps == 16
    assert producer.valid_consumer_count == valid
    cursor = producer.segment_count - 1
    identity = producer.identity(slot_id=0, cursor=cursor, segment_id=cursor)
    assert identity.training_stream_end
    segment = producer.produce(identity)
    segment.validate(16)
    assert segment.consumer_valid.sum() == (valid - cursor * 16)


@pytest.mark.parametrize(
    "change",
    [{"episode_id": "other"}, {"source_digest": "other"}, {"category": "other"}, {"training_stream_end": True}],
)
def test_foreign_identity_or_wrong_terminal_rejected(tmp_path, change):
    producer, _, _ = make_producer(tmp_path)
    identity = producer.identity(slot_id=0, cursor=0, segment_id=0)
    with pytest.raises(ValueError):
        producer.produce(replace(identity, **change))


@pytest.mark.parametrize("kind", ["length", "dim64", "dim12", "nan", "inf"])
def test_source_action_length_and_raw15_fail_closed(tmp_path, kind):
    raw15 = torch.zeros(67, 15)
    if kind == "length":
        raw15 = raw15[:-1]
    elif kind == "dim64":
        raw15 = torch.zeros(67, 64)
    elif kind == "dim12":
        path = write_cache(tmp_path / "native.h5")
        with h5py.File(path, "r") as cache:
            raw15 = torch.from_numpy(cache["robot/action"][:])
    else:
        raw15[0, 0] = float(kind)
    with pytest.raises(ValueError):
        make_producer(tmp_path, raw15=raw15)


def test_action_input_mutation_does_not_change_bound_episode(tmp_path):
    producer, actions, _ = make_producer(tmp_path)
    expected = actions[3].clone()
    actions.fill_(999)
    segment = producer.produce(producer.identity(slot_id=0, cursor=0, segment_id=0))
    assert torch.equal(segment.evidence_executed_action_prev[0, 4], expected)


def test_invalid_cursor_and_short_episode_rejected(tmp_path):
    producer, _, _ = make_producer(tmp_path)
    for cursor in (-1, 3, True):
        with pytest.raises(ValueError):
            producer.identity(slot_id=0, cursor=cursor, segment_id=0)
    short, _, _ = make_producer(tmp_path, frames=32)
    assert short.segment_count == 0
    with pytest.raises(ValueError):
        short.identity(slot_id=0, cursor=0, segment_id=0)


def test_payload_failure_propagates_without_resampling(tmp_path):
    visited = []

    def payload_at(step):
        visited.append(step)
        if step == 3:
            raise OSError("RGB consumer decode failure")
        return {"step": step}

    producer, _, _ = make_producer(tmp_path, payload_at=payload_at)
    with pytest.raises(OSError):
        producer.produce(producer.identity(slot_id=0, cursor=0, segment_id=0))
    assert visited == [0, 1, 2, 3]
