from __future__ import annotations

from types import SimpleNamespace

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

    def _encode_vision_item(self, clip: torch.Tensor, *, num_views: int) -> torch.Tensor:
        assert num_views == 1 and clip.shape[:2] == (1, 3)
        self.encode_calls += 1
        n = (clip.shape[2] - 1) // 4 + 1
        latent = torch.empty(1, 48, n, 16, 16, device=clip.device)
        for index in range(n):
            latent[:, :, index].fill_(float(index + 1))
        return latent


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


def test_required_mode_cold_start_then_one_prefix_encode_per_camera() -> None:
    model = _FakeModel()
    images = {}
    for index in range(16):
        images[f"left-{index}"] = _frame(index)
        images[f"wrist-{index}"] = _frame(100 + index)
    service = SimpleNamespace(model=model)
    adapter = RoboCasaLocalMemoryPolicyAdapter(
        service,
        mode="required",
        decode_image=lambda key: images[key],
    )

    cold = adapter.prepare(_payload(0, 0, reset=True))
    assert adapter.prefixes(cold) == (None,)
    adapter.commit(cold)

    update = adapter.prepare(_payload(16, 0))
    prefix = adapter.prefixes(update)[0]
    assert prefix is not None and prefix.shape == (4, 32)
    assert model.encode_calls == 2
    status = adapter.status(update)
    assert status["consumer_step"] == 16
    assert status["adapted_steps"] == 16
    assert status["fast_update_norm"] > 0
    adapter.commit(update)
    assert len(adapter._visual_records["s"].left_frames) == 16


def test_off_mode_rejects_hidden_evidence() -> None:
    service = SimpleNamespace(model=_FakeModel())
    adapter = RoboCasaLocalMemoryPolicyAdapter(service, mode="off", decode_image=lambda _: _frame(0))
    assert adapter.prepare({}) is None
    try:
        adapter.prepare(_payload(0, 0, reset=True))
    except ValueError as error:
        assert "off" in str(error)
    else:
        raise AssertionError("off mode must reject supplied Local-TTT evidence")


def test_visual96_uses_four_frame_causal_endpoint_groups() -> None:
    model = _FakeModel()
    service = SimpleNamespace(model=model)
    adapter = RoboCasaLocalMemoryPolicyAdapter(
        service,
        mode="required",
        decode_image=lambda _: _frame(0),
    )
    left = tuple(_frame(index) for index in range(16))
    wrist = tuple(_frame(100 + index) for index in range(16))
    summary = adapter._visual96(left, wrist, tuple(range(16)))
    assert summary.shape == (16, 96)
    for start in (0, 4, 8, 12):
        for index in range(start + 1, start + 4):
            torch.testing.assert_close(summary[index], summary[start])
        if start:
            assert not torch.equal(summary[start], summary[start - 1])
    assert model.encode_calls == 2
