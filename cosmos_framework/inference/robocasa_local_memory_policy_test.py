from __future__ import annotations

import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from cosmos_framework.inference import robocasa_local_memory_policy as policy
from cosmos_framework.inference.robocasa_local_memory_contract import PREPROCESS_PROFILE
from cosmos_framework.model.generator.mot.memory_prefix import LocalMemoryRuntime


class _Model(nn.Module):
    def __init__(self, *, with_scan: bool = True) -> None:
        super().__init__()
        self.net = nn.Module()
        self.net.local_memory_runtime = LocalMemoryRuntime(
            evidence_dim=16, local_dim=4, ttt_dim=8, fast_hidden_dim=16, ttt_tbptt_steps=8, k_local=2
        )
        runtime = self.net.local_memory_runtime
        self.scan_calls: list[tuple[int, bool]] = []
        if with_scan:

            def scan(visual, action, valid, state, *, create_graph=True):
                self.scan_calls.append((visual.shape[1], state is not None))
                return runtime.core.scan_segment_masked_encoded_many(
                    runtime.encoder, visual, action, valid, state, create_graph=create_graph
                )

            self.net.scan_local_memory = scan
        self.tokenizer_vision_gen = SimpleNamespace(encode=lambda value: value)


def _request(step: int, start: int, *, reset: bool = False, session: str = "s", episode: str = "e") -> dict:
    return {
        "image_size": 256,
        "local_memory": {
            "session_id": session,
            "episode_id": episode,
            "consumer_step": step,
            "reset": reset,
            "evidence_version": policy.EVIDENCE_VERSION,
            "evidence_format": policy.EVIDENCE_FORMAT,
            "image_size": 256,
            "preprocess_profile": PREPROCESS_PROFILE,
            "evidence": [
                {
                    "source_step": index,
                    "composite_image": f"frame-{index}",
                    "executed_action": [0.01 * (index + 1)] * 15,
                }
                for index in range(start, step)
            ],
        },
    }


@pytest.fixture
def adapter(monkeypatch):
    model = _Model()
    frames = {f"frame-{index}": torch.full((3, 256, 512), index, dtype=torch.uint8) for index in range(3)}
    encoded = []

    def encode(_tokenizer, prepared):
        encoded.append(int(prepared.source_uint8[0, 0, 0]))
        return torch.empty(0), torch.full((96,), float(encoded[-1]))

    monkeypatch.setattr(policy, "encode_current_composite_visual96", encode)
    result = policy.RoboCasaLocalMemoryPolicyAdapter(
        SimpleNamespace(model=model), mode="required", decode_image=lambda key: frames[key]
    )
    return result, model, frames, encoded


def _cold(adapter, *, session="s", episode="e"):
    update = adapter.prepare(_request(0, 0, reset=True, session=session, episode=episode))
    assert adapter.prefixes(update) == (None,)
    assert adapter.status(update)["adapted_steps"] == 0
    adapter.commit(update)


def test_composite_only_and_reject_historical_b1_wire(adapter):
    memory, _, _, _ = adapter
    _cold(memory)
    for field, value in (("evidence_version", "b1"), ("evidence_format", "left_wrist")):
        bad = _request(1, 0)
        bad["local_memory"][field] = value
        with pytest.raises(ValueError, match="historical B1"):
            memory.prepare(bad)
    bad = _request(1, 0)
    row = bad["local_memory"]["evidence"][0]
    row["left_image"] = row.pop("composite_image")
    row["wrist_image"] = "frame-0"
    with pytest.raises(ValueError, match="composite_image"):
        memory.prepare(bad)
    bad = _request(2, 0)
    bad["local_memory"]["evidence"][1]["source_step"] = 2
    with pytest.raises(ValueError, match="contiguous"):
        memory.prepare(bad)


def test_session_preprocess_identity_and_episode_reset(adapter):
    memory, _, _, _ = adapter
    _cold(memory)
    first = memory.prepare(_request(1, 0))
    memory.commit(first)
    for change in ("image_size", "preprocess_profile", "episode_id"):
        bad = _request(2, 1)
        bad["local_memory"][change] = 512 if change == "image_size" else "other"
        if change == "image_size":
            bad["image_size"] = 512
        with pytest.raises(ValueError, match="changed without reset|preprocess_profile"):
            memory.prepare(bad)
    reset = memory.prepare(_request(0, 0, reset=True, episode="new"))
    assert memory.prefixes(reset) == (None,)
    memory.commit(reset)
    assert memory.status(reset)["consumer_step"] == 0
    assert memory.memory._records["s"].episode_id == "new"


def test_lost_response_replay_is_exact_and_does_not_encode(adapter):
    memory, model, frames, encoded = adapter
    _cold(memory)
    first_request = _request(2, 0)
    first = memory.prepare(first_request)
    assert memory.status(first)["encoded_steps"] == 2
    assert model.scan_calls == [(2, False)]
    token = memory.prefixes(first)[0]
    memory.commit(first)
    replay = memory.prepare(deepcopy(first_request))
    assert memory.status(replay)["replay"] is True
    assert memory.status(replay)["encoded_steps"] == 0
    torch.testing.assert_close(memory.prefixes(replay)[0], token)
    assert len(encoded) == 2 and model.scan_calls == [(2, False)]
    memory.commit(replay)
    bad = deepcopy(first_request)
    bad["local_memory"]["evidence"][1]["executed_action"][0] += 1
    with pytest.raises(ValueError, match="bytes/action changed"):
        memory.prepare(bad)
    frames["changed"] = torch.full((3, 256, 512), 17, dtype=torch.uint8)
    bad = deepcopy(first_request)
    bad["local_memory"]["evidence"][1]["composite_image"] = "changed"
    with pytest.raises(ValueError, match="bytes/action changed"):
        memory.prepare(bad)
    assert len(encoded) == 2


def test_abort_and_reset_isolation(adapter):
    memory, _, _, encoded = adapter
    _cold(memory)
    candidate = memory.prepare(_request(1, 0))
    memory.abort(candidate)
    assert memory.memory._records["s"].consumer_step == 0
    retry = memory.prepare(_request(1, 0))
    memory.commit(retry)
    assert len(encoded) == 2
    memory.reset("s")
    _cold(memory, episode="other")
    assert memory.memory._records["s"].state is None


def test_required_needs_model_owned_scan_and_status_has_no_b1_fields(adapter):
    memory, _, _, _ = adapter
    _cold(memory)
    update = memory.prepare(_request(1, 0))
    status = memory.status(update)
    assert not {"visual_endpoint_step", "visual_tail_frames", "left_image", "wrist_image"} & status.keys()
    memory.abort(update)
    with pytest.raises(ValueError, match="model-owned scan"):
        policy.RoboCasaLocalMemoryPolicyAdapter(
            SimpleNamespace(model=_Model(with_scan=False)), mode="required", decode_image=lambda _: None
        )


def test_off_mode_rejects_evidence_without_encoding():
    adapter = policy.RoboCasaLocalMemoryPolicyAdapter(SimpleNamespace(), mode="off", decode_image=lambda _: None)
    assert adapter.prepare({}) is None
    assert adapter.info() == {"enabled": False, "mode": "off"}
    with pytest.raises(ValueError, match="off"):
        adapter.prepare(_request(0, 0, reset=True))


def test_corrected_active_modules_do_not_import_historical_b1() -> None:
    root = Path(__file__).resolve().parents[1]
    modules = (
        root / "inference/robocasa_local_memory_policy.py",
        root / "inference/robocasa_composite_visual.py",
        root / "scripts/action_policy_server_robocasa.py",
        root / "simulation/robocasa/local_memory_client.py",
        root / "simulation/robocasa/closed_loop_eval.py",
    )
    forbidden = {"robocasa_causal_evidence", "robocasa_latent_evidence"}
    for module in modules:
        tree = ast.parse(module.read_text(encoding="utf-8"))
        imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        imports += [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
        assert not any(any(name in imported for name in forbidden) for imported in imports if imported), module
