from __future__ import annotations

import sys
import threading
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from cosmos_framework.inference.robocasa_composite_visual import prepare_robocasa_composite_frame

with patch("cosmos_framework.inference.common.init._init_script", lambda **kwargs: None):
    sys.modules.pop("cosmos_framework.scripts.action_policy_server_robocasa", None)
    from cosmos_framework.scripts import action_policy_server_robocasa as server  # noqa: E402


def _service(monkeypatch, *, mode="required", fail=False):
    service = server.ActionModelService.__new__(server.ActionModelService)
    service.cfg = SimpleNamespace(
        action_chunk_size=1, max_action_dim=15, fps=10, guidance=1, seed=0, num_steps=1, dump_dir=None
    )
    service.raw_action_dim = 15
    service._req_id_lock = threading.Lock()
    service._req_id = 0
    service._lock = threading.Lock()
    service._input_video_key = lambda: "video"
    service._denormalize_action = lambda action: action
    service._should_dump = lambda _: False
    service._prep_policy_item = lambda _: {
        "img_chw_uint8": torch.zeros(3, 256, 512, dtype=torch.uint8),
        "video_padded": torch.zeros(3, 2, 16, 16, dtype=torch.uint8),
        "padded_image_size": torch.tensor([256, 512]),
        "augmented_prompt": "prompt",
        "sequence_plan": None,
        "domain_name": "robocasa",
        "image_size": 256,
        "state_token": None,
    }
    events = []
    adapter = SimpleNamespace(mode=mode)
    adapter.prepare = Mock(side_effect=lambda _: events.append("prepare") or "candidate")
    adapter.prefixes = Mock(side_effect=lambda _: events.append("prefix") or (torch.ones(2, 4),))
    adapter.status = Mock(side_effect=lambda _: events.append("status") or {"encoded_steps": 1})
    adapter.commit = Mock(side_effect=lambda _: events.append("commit"))
    adapter.abort = Mock(side_effect=lambda _: events.append("abort"))
    service.local_memory_adapter = adapter

    def generate(*args, **kwargs):
        events.append("generate")
        if fail:
            raise RuntimeError("generation failed")
        return {"action": [torch.zeros(1, 1, 15)], "vision": [torch.empty(0)]}

    service.model = SimpleNamespace(
        training=False, generate_samples_from_batch=generate, decode=lambda _: torch.zeros(1, 3, 1, 16, 16)
    )
    monkeypatch.setattr(server, "get_domain_id", lambda _: 0)
    monkeypatch.setattr(server, "remove_reflection_padding", lambda video, _: video)
    monkeypatch.setattr(server, "_video_tensor_to_pil_images", lambda _: [])
    original_to = torch.Tensor.to

    def cpu_to(tensor, *args, **kwargs):
        if kwargs.get("device") == "cuda":
            kwargs["device"] = "cpu"
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", cpu_to)
    return service, adapter, events


def test_server_policy_uses_shared_composite_spatial_preprocessing(monkeypatch):
    service = server.ActionModelService.__new__(server.ActionModelService)
    service.cfg = SimpleNamespace(action_chunk_size=4, fps=10)
    service.requires_state = False
    service.raw_action_dim = 15
    service._prompt_json_formatter = None
    service.append_duration_fps = False
    service.append_resolution_info = False
    frame = torch.arange(3 * 256 * 512).reshape(3, 256, 512).to(torch.uint8)
    monkeypatch.setattr(server, "_decode_base64_png_to_rgb_uint8", lambda _: frame)
    request = {"image": "composite", "prompt": "task", "domain_name": "robocasa", "image_size": 128}
    prepared = service._prep_policy_item(request)
    shared = prepare_robocasa_composite_frame(frame, 128)
    assert torch.equal(prepared["img_chw_uint8"], shared.source_uint8)
    assert torch.equal(prepared["video_padded"], shared.padded_single_frame.repeat(1, 5, 1, 1))
    assert torch.equal(prepared["padded_image_size"], shared.padded_image_size)


def test_server_raw15_state_and_prompt_contract(monkeypatch):
    service = server.ActionModelService.__new__(server.ActionModelService)
    service.cfg = SimpleNamespace(action_chunk_size=4, max_action_dim=15, fps=10)
    service.requires_state = True
    service.raw_action_dim = 15
    service._prompt_json_formatter = None
    service.append_duration_fps = True
    service.append_resolution_info = True
    monkeypatch.setattr(
        server, "_decode_base64_png_to_rgb_uint8", lambda _: torch.zeros(3, 256, 512, dtype=torch.uint8)
    )
    request = {"image": "composite", "prompt": "task", "domain_name": "robocasa", "image_size": 256}
    with pytest.raises(ValueError, match="is required"):
        service._prep_policy_item(request)
    request["state"] = [float(index) for index in range(15)]
    prepared = service._prep_policy_item(request)
    action = service._build_action_input(prepared["state_token"])
    assert action.shape == (5, 15)
    torch.testing.assert_close(action[0], torch.arange(15, dtype=torch.float32))
    assert torch.count_nonzero(action[1:]) == 0
    assert "task" in prepared["augmented_prompt"]
    assert "256" in prepared["augmented_prompt"]


def test_required_serial_predict_prepare_generate_commit(monkeypatch):
    service, adapter, events = _service(monkeypatch)
    with pytest.raises(ValueError, match="serial /predict only"):
        service.predict_policy_batch([])
    assert events == []
    output = service.predict_policy({})
    assert events == ["prepare", "prefix", "generate", "status", "commit"]
    assert output["local_memory"] == {"encoded_steps": 1}
    adapter.abort.assert_not_called()


def test_generation_failure_aborts_candidate(monkeypatch):
    service, adapter, events = _service(monkeypatch, fail=True)
    with pytest.raises(RuntimeError, match="generation failed"):
        service.predict_policy({})
    assert events == ["prepare", "prefix", "generate", "abort"]
    adapter.commit.assert_not_called()


def test_off_mode_does_not_call_encode1(monkeypatch):
    service, adapter, events = _service(monkeypatch, mode="off")
    adapter.prepare = Mock(side_effect=lambda _: events.append("prepare") or None)
    adapter.prefixes = Mock(side_effect=lambda _: None)
    adapter.status = Mock(side_effect=lambda _: None)
    service.predict_policy({})
    assert events == ["prepare", "generate", "commit"]
    assert adapter.commit.call_args.args == (None,)
