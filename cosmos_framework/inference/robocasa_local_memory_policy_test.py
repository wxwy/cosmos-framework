from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from cosmos_framework.inference.robocasa_local_memory_policy import RoboCasaLocalMemoryPolicyAdapter
from cosmos_framework.model.generator.mot.memory_prefix import LocalMemoryRuntime
from cosmos_framework.model.generator.tokenizers.wan2pt2_vae_4x16x16 import WanEncoderStreamState


class _FakeTokenizer:
    is_causal = True

    def __init__(self) -> None:
        self.state = WanEncoderStreamState((), None)

    def new_encoder_stream_state(self) -> WanEncoderStreamState:
        return WanEncoderStreamState((), None)

    def snapshot_encoder_stream_state(self) -> WanEncoderStreamState:
        return self.state

    def restore_encoder_stream_state(self, state: WanEncoderStreamState) -> None:
        self.state = state


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
        self.tokenizer_vision_gen = _FakeTokenizer()
        self.encode_shapes: list[tuple[int, ...]] = []

    def _encode_vision_item_streaming(self, clip: torch.Tensor, *, num_views: int) -> torch.Tensor:
        assert num_views == 1
        assert clip.ndim == 5 and clip.shape[:2] == (2, 3)
        self.encode_shapes.append(tuple(clip.shape))
        state = self.tokenizer_vision_gen.state
        if clip.shape[2] == 1:
            assert state.stream_shape is None
            endpoint = 0
        else:
            assert clip.shape[2] == 4 and state.stream_shape is not None
            endpoint = int(state.stream_shape[0]) + 4
        self.tokenizer_vision_gen.state = WanEncoderStreamState(
            (),
            (endpoint, 256, 256, torch.device("cpu"), torch.float32),
        )
        values = torch.tensor([float(endpoint + 1), float(endpoint + 101)], device=clip.device)
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
    adapter = RoboCasaLocalMemoryPolicyAdapter(
        SimpleNamespace(model=model),
        mode="required",
        decode_image=lambda key: images[key],
    )
    return model, adapter, images


def _cold(adapter: RoboCasaLocalMemoryPolicyAdapter) -> None:
    update = adapter.prepare(_payload(0, 0, reset=True))
    assert adapter.prefixes(update) == (None,)
    adapter.commit(update)


@pytest.mark.parametrize(
    ("count", "temporal_shapes"),
    [
        (4, [1]),
        (8, [1, 4]),
        (16, [1, 4, 4, 4]),
    ],
)
def test_replan_horizon_reproduces_b1_causal_endpoints(count: int, temporal_shapes: list[int]) -> None:
    model, adapter, _ = _adapter(count)
    _cold(adapter)
    update = adapter.prepare(_payload(count, 0))
    status = adapter.status(update)
    assert status["consumer_step"] == count
    assert status["adapted_steps"] == count
    assert [shape[2] for shape in model.encode_shapes] == temporal_shapes

    record = update.visual_replacement
    assert len(record.stream_state.pending_left) == (count - 1) % 4
    assert len(record.stream_state.pending_wrist) == (count - 1) % 4
    visual = record.last_visual_summary
    assert visual is not None and visual.shape == (count, 96)
    for start in range(0, count, 4):
        stop = min(start + 4, count)
        torch.testing.assert_close(visual[start:stop], visual[start : start + 1].expand(stop - start, -1))
        if start:
            assert not torch.equal(visual[start], visual[start - 1])


def test_replan_boundary_continues_stream_without_episode_history() -> None:
    model, adapter, _ = _adapter(8)
    _cold(adapter)

    first = adapter.prepare(_payload(4, 0))
    assert adapter.status(first)["visual_endpoint_step"] == 0
    assert adapter.status(first)["visual_tail_frames"] == 3
    adapter.commit(first)

    second = adapter.prepare(_payload(8, 4))
    assert adapter.status(second)["visual_endpoint_step"] == 4
    assert adapter.status(second)["visual_tail_frames"] == 3
    assert [shape[2] for shape in model.encode_shapes] == [1, 4]
    assert not hasattr(second.visual_replacement, "left_frames")
    assert not hasattr(second.visual_replacement, "wrist_frames")


def test_prepare_abort_restores_visual_stream_transaction() -> None:
    model, adapter, _ = _adapter(8)
    _cold(adapter)
    live_before = model.tokenizer_vision_gen.snapshot_encoder_stream_state()

    first = adapter.prepare(_payload(8, 0))
    prefix_first = adapter.prefixes(first)[0].detach().clone()
    assert model.tokenizer_vision_gen.snapshot_encoder_stream_state() == live_before
    adapter.abort(first)

    second = adapter.prepare(_payload(8, 0))
    torch.testing.assert_close(adapter.prefixes(second)[0], prefix_first)
    assert [shape[2] for shape in model.encode_shapes] == [1, 4, 1, 4]
    assert model.tokenizer_vision_gen.snapshot_encoder_stream_state() == live_before


def test_committed_evidence_retry_is_idempotent_and_changed_retry_fails() -> None:
    model, adapter, images = _adapter(16)
    _cold(adapter)

    first = adapter.prepare(_payload(16, 0))
    first_prefix = adapter.prefixes(first)[0]
    adapter.commit(first)
    encode_calls_after_commit = len(model.encode_shapes)

    replay = adapter.prepare(_payload(16, 0))
    assert adapter.status(replay)["replay"] is True
    torch.testing.assert_close(adapter.prefixes(replay)[0], first_prefix)
    assert len(model.encode_shapes) == encode_calls_after_commit
    adapter.commit(replay)

    changed_action = _payload(16, 0)
    changed_action["local_memory"]["evidence"][-1]["executed_action"][0] += 1.0
    with pytest.raises(ValueError, match="replay action evidence changed"):
        adapter.prepare(changed_action)

    images["left-changed"] = _frame(77)
    changed_visual = _payload(16, 0)
    changed_visual["local_memory"]["evidence"][-1]["left_image"] = "left-changed"
    with pytest.raises(ValueError, match="replay visual evidence changed"):
        adapter.prepare(changed_visual)
    assert len(model.encode_shapes) == encode_calls_after_commit


def test_off_mode_rejects_hidden_evidence() -> None:
    service = SimpleNamespace(model=_FakeModel())
    adapter = RoboCasaLocalMemoryPolicyAdapter(service, mode="off", decode_image=lambda _: _frame(0))
    assert adapter.prepare({}) is None
    with pytest.raises(ValueError, match="off"):
        adapter.prepare(_payload(0, 0, reset=True))
