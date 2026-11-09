from __future__ import annotations

import ast
import json
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
    assert {
        key: adapter.status(update)[key]
        for key in ("adapted_steps", "inner_loss_mean", "fast_state_norm", "fast_update_norm")
    } == {
        "adapted_steps": 0.0,
        "inner_loss_mean": 0.0,
        "fast_state_norm": 0.0,
        "fast_update_norm": 0.0,
    }
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
    assert {key: memory.status(replay)[key] for key in ("adapted_steps", "inner_loss_mean", "fast_update_norm")} == {
        "adapted_steps": 0.0,
        "inner_loss_mean": 0.0,
        "fast_update_norm": 0.0,
    }
    assert memory.status(replay)["fast_state_norm"] == pytest.approx(memory.status(first)["fast_state_norm"])
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


def _feature_adapter(*, max_sessions=1, history_k=16):
    from cosmos_framework.inference.robocasa_feature_evidence_test import EvidenceHarness

    harness = EvidenceHarness(bind_lookup=False, max_sessions=max_sessions, history_k=history_k)
    model = _Model()

    def no_history_decode(_):
        raise AssertionError("C history must use verified feature bytes without PNG decode or Encode1")

    adapter = policy.RoboCasaLocalMemoryPolicyAdapter(
        SimpleNamespace(model=model),
        mode="required",
        decode_image=no_history_decode,
        feature_evidence_owner=harness.owner,
        max_sessions=max_sessions,
    )
    return harness, adapter, model


def _feature_request(harness, step, rows):
    payload = harness.payload(rows, consumer_step=step)
    payload["preprocess_profile"] = PREPROCESS_PROFILE
    return {"image_size": 256, "local_memory": payload}


def _prepare_feature(adapter, harness, step, rows):
    from cosmos_framework.inference.robocasa_feature_evidence_test import CLOCK

    return adapter.prepare(
        _feature_request(harness, step, rows),
        lease=harness.lease,
        deadline_ns=9000,
        clock_domain=CLOCK,
        request_key=f"request-{step}",
    )


def _commit_feature_cold(adapter, harness):
    from cosmos_framework.inference.local_memory_state import CachedMemoryResponse
    from cosmos_framework.inference.robocasa_vae_protocol import canonical_json

    update = _prepare_feature(adapter, harness, 0, [])
    assert adapter.committed_boundary(harness.lease["session_id"], 0) is None
    adapter.stage_boundary(update, harness.boundary())
    private_status = adapter.status(update)
    assert private_status["committed_boundary"] == harness.boundary().record()
    assert adapter.committed_boundary(harness.lease["session_id"], 0) is None
    response = CachedMemoryResponse("request-0", "f" * 64, canonical_json({"local_memory": private_status}))
    adapter.commit(update, response)
    harness.commit_chunk()
    return response


def test_feature_boundary_response_fast_state_publish_once_and_exact_replay():
    from cosmos_framework.inference.local_memory_state import CachedMemoryResponse
    from cosmos_framework.inference.robocasa_vae_protocol import canonical_json

    harness, adapter, model = _feature_adapter()
    _commit_feature_cold(adapter, harness)
    rows = harness.rows()
    before = adapter.snapshot
    update = _prepare_feature(adapter, harness, 16, rows)
    assert adapter.snapshot is before
    assert model.scan_calls == [(1, False)] + [(1, True)] * 15
    assert adapter.status(update)["encoded_steps"] == 0
    context = adapter.boundary_context(update)
    assert context["lease"] == harness.lease and context["source_step"] == 16
    assert context["chunk_id"] == context["request_key"] == "request-16"
    context["lease"]["session_id"] = "caller-mutated"
    assert adapter.boundary_context(update)["lease"] == harness.lease
    adapter.stage_boundary(update, harness.boundary(16))
    assert adapter.committed_boundary(harness.lease["session_id"], 16) is None
    body = canonical_json({"local_memory": adapter.status(update)})
    cached = CachedMemoryResponse("request-16", "e" * 64, body)
    adapter.commit(update, cached)
    snapshot = adapter.snapshot
    session = harness.lease["session_id"]
    assert snapshot.fast[session].consumer_step == snapshot.visual[session].consumer_step == 16
    assert snapshot.visual[session].boundary == harness.boundary(16)
    assert (
        json.loads(snapshot.responses[session].body)["local_memory"]["committed_boundary"]
        == harness.boundary(16).record()
    )
    assert not adapter._visual_pending and not adapter._boundary_pending and not adapter.memory._pending
    replay = _prepare_feature(adapter, harness, 16, deepcopy(rows))
    assert replay.memory.replay and len(model.scan_calls) == 16
    adapter.commit(replay, cached)
    assert adapter.snapshot.fast[session] is snapshot.fast[session]
    assert adapter.snapshot.visual[session].boundary == snapshot.visual[session].boundary
    changed = deepcopy(rows)
    changed[1]["post_step_receipt"]["source_rgb_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="replay"):
        _prepare_feature(adapter, harness, 16, changed)


def test_feature_abort_and_end_discard_private_response_and_boundary():
    from cosmos_framework.inference.robocasa_feature_evidence_test import TOKEN

    harness, adapter, _ = _feature_adapter()
    cold_response = _commit_feature_cold(adapter, harness)
    rows = harness.rows()
    before = adapter.snapshot
    update = _prepare_feature(adapter, harness, 16, rows)
    adapter.stage_boundary(update, harness.boundary(16))
    assert adapter.status(update)["committed_boundary"] == harness.boundary(16).record()
    assert harness.owner.boundary_record(TOKEN, harness.lease, 0) == harness.boundary().record()
    with pytest.raises(ValueError):
        harness.owner.boundary_record(TOKEN, harness.lease, 16)
    adapter.abort(update)
    assert adapter.snapshot is before
    assert adapter.snapshot.responses[harness.lease["session_id"]] is cold_response
    assert adapter.committed_boundary(harness.lease["session_id"], 16) is None
    assert not adapter._visual_pending and not adapter._boundary_pending and not adapter.memory._pending
    adapter.reset(harness.lease["session_id"])
    harness.owner.end_session(TOKEN, harness.lease)
    assert not adapter.snapshot.fast and not adapter.snapshot.visual and not adapter.snapshot.responses


def test_feature_missing_step_or_boundary_cannot_publish_or_scan():
    harness, adapter, model = _feature_adapter()
    initial = adapter.snapshot
    cold = _prepare_feature(adapter, harness, 0, [])
    with pytest.raises(RuntimeError, match="boundary"):
        adapter.commit(cold)
    assert adapter.snapshot is initial
    adapter.abort(cold)
    _commit_feature_cold(adapter, harness)
    rows = harness.rows()
    before = adapter.snapshot
    with pytest.raises(ValueError, match="contiguous"):
        _prepare_feature(adapter, harness, 16, rows[:-1])
    assert adapter.snapshot is before and not model.scan_calls and not adapter.memory._pending
    with pytest.raises(ValueError):
        adapter.prepare(_feature_request(harness, 16, rows))


def test_feature_revoke_between_prepare_and_commit_cannot_publish():
    from cosmos_framework.inference.robocasa_feature_evidence_test import TOKEN

    harness, adapter, _ = _feature_adapter()
    _commit_feature_cold(adapter, harness)
    rows = harness.rows()
    before = adapter.snapshot
    update = _prepare_feature(adapter, harness, 16, rows)
    adapter.stage_boundary(update, harness.boundary(16))
    harness.owner.revoke_worker(TOKEN, "worker", "worker-epoch")
    with pytest.raises(ValueError, match="inactive|revoked"):
        adapter.commit(update)
    assert adapter.snapshot is before
    adapter.abort(update)
    assert not adapter._visual_pending and not adapter._boundary_pending and not adapter.memory._pending


def test_feature_profile_cannot_fallback_to_legacy_or_reset_completed_frontier():
    from cosmos_framework.inference.robocasa_feature_evidence_test import CLOCK

    harness, adapter, model = _feature_adapter()
    with pytest.raises(ValueError, match="unsupported"):
        adapter.prepare(
            _request(0, 0, reset=True),
            lease=harness.lease,
            deadline_ns=9000,
            clock_domain=CLOCK,
            request_key="request-0",
        )
    _commit_feature_cold(adapter, harness)
    harness.rows()
    before = adapter.snapshot
    with pytest.raises(ValueError, match="reset"):
        _prepare_feature(adapter, harness, 0, [])
    assert adapter.snapshot is before and not model.scan_calls and not adapter.memory._pending


def test_feature_batch_missing_second_boundary_never_partially_publishes():
    from cosmos_framework.inference.robocasa_feature_evidence_test import CLOCK, TOKEN

    harness, adapter, _ = _feature_adapter(max_sessions=2)
    other = {**harness.lease, "session_id": "session-2", "episode_id": "episode-2", "slot_id": "slot-1"}
    harness.owner.register_session(
        TOKEN, other, worker_id="worker", actor_id="actor-2", actor_epoch="actor-epoch-2", episode_deadline_ns=10000
    )
    before = adapter.snapshot
    first = _prepare_feature(adapter, harness, 0, [])
    request = _feature_request(harness, 0, [])
    request["local_memory"].update(session_id=other["session_id"], episode_id=other["episode_id"])
    second = adapter.prepare(request, lease=other, deadline_ns=9000, clock_domain=CLOCK, request_key="request-other-0")
    adapter.stage_boundary(first, harness.boundary())
    with pytest.raises(RuntimeError, match="boundary"):
        adapter.commit_many(((first, None), (second, None)))
    assert adapter.snapshot is before and not adapter.snapshot.fast and not adapter.snapshot.visual
    assert len(adapter._visual_pending) == len(adapter.memory._pending) == 2
    adapter.abort(first)
    adapter.abort(second)
    assert not adapter._boundary_pending and not adapter._visual_pending and not adapter.memory._pending
