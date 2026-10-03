from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from cosmos_framework.inference.robocasa_local_memory_policy import RoboCasaLocalMemoryPolicyAdapter
from cosmos_framework.model.generator.mot.memory_prefix import LocalMemoryRuntime


class _FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0))
        self.net = nn.Module()
        self.net.local_memory_runtime = LocalMemoryRuntime(action_dim=15)
        runtime = self.net.local_memory_runtime

        def scan(visual, action, valid, state, *, create_graph=True):
            return runtime.core.scan_segment_masked_encoded_many(
                runtime.encoder,
                visual,
                action,
                valid,
                state,
                create_graph=create_graph,
            )

        self.net.scan_local_memory = scan
        self.encode_calls = 0
        self.encode_shapes: list[tuple[int, ...]] = []

    def _encode_vision_item(self, clip: torch.Tensor, *, num_views: int) -> torch.Tensor:
        assert num_views == 1
        assert clip.ndim == 5 and clip.shape[1:3] == (3, 1)
        self.encode_calls += 1
        self.encode_shapes.append(tuple(clip.shape))
        values = clip[:, 0, 0, 0, 0].float()
        return values[:, None, None, None, None].expand(-1, 48, 1, 16, 16).clone()


def _frame(value: int) -> torch.Tensor:
    return torch.full((3, 256, 256), value, dtype=torch.uint8)


def _payload(step: int, start: int, *, reset: bool = False) -> dict:
    rows = [
        {
            "source_step": index,
            "left_image": f"left-{index}",
            "wrist_image": f"wrist-{index}",
            "executed_action": [0.01 * (index + 1)] * 15,
        }
        for index in range(start, step)
    ]
    return {
        "local_memory": {
            "session_id": "s",
            "episode_id": "e",
            "consumer_step": step,
            "reset": reset,
            "evidence_version": RoboCasaLocalMemoryPolicyAdapter.EVIDENCE_VERSION,
            "evidence_format": RoboCasaLocalMemoryPolicyAdapter.EVIDENCE_FORMAT,
            "evidence": rows,
        }
    }


def _adapter(count: int) -> tuple[_FakeModel, RoboCasaLocalMemoryPolicyAdapter, dict[str, torch.Tensor]]:
    model = _FakeModel()
    images: dict[str, torch.Tensor] = {}
    for index in range(count):
        images[f"left-{index}"] = _frame(index)
        images[f"wrist-{index}"] = _frame(100 + index)
    service = SimpleNamespace(model=model)
    adapter = RoboCasaLocalMemoryPolicyAdapter(
        service,
        mode="required",
        decode_image=lambda key: images[key],
    )
    return model, adapter, images


def test_required_mode_cold_start_then_current_frame_batch_encode() -> None:
    model, adapter, _ = _adapter(16)

    cold = adapter.prepare(_payload(0, 0, reset=True))
    assert adapter.prefixes(cold) == (None,)
    adapter.commit(cold)

    update = adapter.prepare(_payload(16, 0))
    prefix = adapter.prefixes(update)[0]
    assert prefix is not None and prefix.shape == (4, 32)
    assert model.encode_calls == 1
    assert model.encode_shapes == [(32, 3, 1, 256, 256)]
    status = adapter.status(update)
    assert status["consumer_step"] == 16
    assert status["adapted_steps"] == 16
    assert status["fast_update_norm"] > 0
    adapter.commit(update)
    record = adapter._visual_records["s"]
    assert record.last_source_steps == tuple(range(16))
    assert record.last_visual_summary is not None and record.last_visual_summary.shape == (16, 96)
    assert record.last_visual_digest is not None


@pytest.mark.parametrize("count", [4, 8, 16])
def test_replan_horizon_updates_only_completed_current_frames(count: int) -> None:
    model, adapter, _ = _adapter(count)
    cold = adapter.prepare(_payload(0, 0, reset=True))
    adapter.commit(cold)

    update = adapter.prepare(_payload(count, 0))
    status = adapter.status(update)
    assert status["consumer_step"] == count
    assert status["adapted_steps"] == count
    assert model.encode_shapes == [(2 * count, 3, 1, 256, 256)]


def test_off_mode_rejects_hidden_evidence() -> None:
    service = SimpleNamespace(model=_FakeModel())
    adapter = RoboCasaLocalMemoryPolicyAdapter(service, mode="off", decode_image=lambda _: _frame(0))
    assert adapter.prepare({}) is None
    with pytest.raises(ValueError, match="off"):
        adapter.prepare(_payload(0, 0, reset=True))


def test_visual96_rows_depend_only_on_their_current_frames() -> None:
    model, adapter, _ = _adapter(4)
    left = tuple(_frame(index) for index in range(4))
    wrist = tuple(_frame(100 + index) for index in range(4))

    first = adapter._visual96_current_frames(left, wrist)
    changed_left = list(left)
    changed_left[2] = _frame(42)
    second = adapter._visual96_current_frames(tuple(changed_left), wrist)

    assert first.shape == second.shape == (4, 96)
    assert model.encode_shapes == [(8, 3, 1, 256, 256), (8, 3, 1, 256, 256)]
    torch.testing.assert_close(first[:2], second[:2])
    assert not torch.equal(first[2], second[2])
    torch.testing.assert_close(first[3:], second[3:])


def test_committed_evidence_retry_is_idempotent_and_changed_retry_fails() -> None:
    model, adapter, images = _adapter(16)

    cold = adapter.prepare(_payload(0, 0, reset=True))
    adapter.commit(cold)

    first = adapter.prepare(_payload(16, 0))
    first_prefix = adapter.prefixes(first)[0]
    assert first_prefix is not None
    adapter.commit(first)
    encode_calls_after_commit = model.encode_calls

    replay = adapter.prepare(_payload(16, 0))
    replay_prefix = adapter.prefixes(replay)[0]
    assert replay_prefix is not None
    assert adapter.status(replay)["replay"] is True
    torch.testing.assert_close(replay_prefix, first_prefix)
    assert model.encode_calls == encode_calls_after_commit
    adapter.commit(replay)

    changed_action = _payload(16, 0)
    changed_action["local_memory"]["evidence"][-1]["executed_action"][0] += 1.0
    with pytest.raises(ValueError, match="replay action evidence changed"):
        adapter.prepare(changed_action)
    assert model.encode_calls == encode_calls_after_commit

    images["left-changed"] = _frame(77)
    changed_visual = _payload(16, 0)
    changed_visual["local_memory"]["evidence"][-1]["left_image"] = "left-changed"
    with pytest.raises(ValueError, match="replay visual evidence changed"):
        adapter.prepare(changed_visual)
    assert model.encode_calls == encode_calls_after_commit
