"""Synthetic CPU contracts for unpublished, model-owned per-frame Local-TTT.

Wan encoding is the explicit synthetic boundary. The evidence encoder, fast core,
model-owned scan, adapter, immutable publication and pending capabilities are real.
"""

from __future__ import annotations

import copy
import threading
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from cosmos_framework.inference import local_memory_pipeline as pipeline_module
from cosmos_framework.inference import robocasa_local_memory_policy as policy
from cosmos_framework.inference.local_memory_state import CachedMemoryResponse
from cosmos_framework.inference.robocasa_composite_visual import (
    CapturedCurrentVisual,
    composite_frame_digest,
    prepare_robocasa_composite_frame,
)
from cosmos_framework.inference.robocasa_local_memory_contract import PREPROCESS_PROFILE
from cosmos_framework.model.generator.mot.memory_prefix import LocalMemoryRuntime


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Module()
        self.net.local_memory_runtime = LocalMemoryRuntime(
            evidence_dim=16, local_dim=4, ttt_dim=8, fast_hidden_dim=16, ttt_tbptt_steps=16, k_local=2
        )
        self.net.scan_local_memory = self.scan
        self.tokenizer_vision_gen = SimpleNamespace(encode=lambda value: value, encoded=[])
        self.calls = []
        self.before_scan = None

    def scan(self, visual, action, valid, state, *, create_graph=True):
        self.calls.append(
            {
                "steps": visual.shape[1],
                "create_graph": create_graph,
                "grad": torch.is_grad_enabled(),
                "inference": torch.is_inference_mode_enabled(),
            }
        )
        if self.before_scan is not None:
            self.before_scan()
        runtime = self.net.local_memory_runtime
        return runtime.core.scan_segment_masked_encoded_many(
            runtime.encoder, visual, action, valid, state, create_graph=create_graph
        )


def _frame(index: int) -> torch.Tensor:
    return torch.full((3, 256, 512), index, dtype=torch.uint8)


def _visual(index: int) -> torch.Tensor:
    return torch.linspace(-0.5, 0.5, 96) + index * 0.001


def _decode(key: str) -> torch.Tensor:
    return _frame(int(key.removeprefix("frame-")))


def _lease(index: int = 0) -> dict:
    return {
        "server_epoch": "a" * 32,
        "run_lease": "run",
        "manifest_digest": "b" * 64,
        "slot_id": f"slot-{index}",
        "generation": 1,
        "task": "CloseFridge",
        "episode_index": index,
        "attempt_id": f"attempt-{index}",
        "session_id": f"session-{index}",
        "episode_id": f"episode-{index}",
        "capability": f"capability-{index}",
    }


def _completed(lease: dict, step: int) -> dict:
    return {
        "session_id": lease["session_id"],
        "episode_id": lease["episode_id"],
        "image_size": 256,
        "preprocess_profile": PREPROCESS_PROFILE,
        "source_step": step,
        "composite_image": f"frame-{step}",
        "executed_action": [0.01 * (step + 1)] * 15,
    }


def _request(lease: dict, step: int, start: int = 0, *, reset: bool = False) -> dict:
    return {
        "image_size": 256,
        "local_memory": {
            "session_id": lease["session_id"],
            "episode_id": lease["episode_id"],
            "consumer_step": step,
            "reset": reset,
            "evidence_version": policy.EVIDENCE_VERSION,
            "evidence_format": policy.EVIDENCE_FORMAT,
            "image_size": 256,
            "preprocess_profile": PREPROCESS_PROFILE,
            "evidence": [
                {key: row[key] for key in ("source_step", "composite_image", "executed_action")}
                for row in (_completed(lease, index) for index in range(start, step))
            ],
        },
    }


def _capture(step: int) -> CapturedCurrentVisual:
    prepared = prepare_robocasa_composite_frame(_frame(step), 256)
    return CapturedCurrentVisual(
        source_digest=composite_frame_digest(prepared.source_uint8),
        visual96=_visual(step),
        image_size=256,
        preprocess_profile=PREPROCESS_PROFILE,
    )


@pytest.fixture
def make_adapter(monkeypatch):
    def encode(tokenizer, prepared, *, timing=None):
        index = int(prepared.source_uint8[0, 0, 0])
        if timing is None:
            tokenizer.encoded.append(index)
        else:
            with timing.measure("independent_encode1"):
                tokenizer.encoded.append(index)
        return torch.empty(0), _visual(index)

    monkeypatch.setattr(policy, "encode_current_composite_visual96", encode)
    # The separate candidate engine may own this same explicit Wan boundary.
    monkeypatch.setattr(pipeline_module, "encode_current_composite_visual96", encode, raising=False)

    def make(*, enabled=True, sessions=2, reference=None, **kwargs):
        torch.manual_seed(372)
        model = _Model()
        if reference is not None:
            model.load_state_dict(reference.state_dict())
        if enabled is not None:
            kwargs["pipeline_enabled"] = enabled
        adapter = policy.RoboCasaLocalMemoryPolicyAdapter(
            SimpleNamespace(model=model),
            mode="required",
            decode_image=_decode,
            max_sessions=sessions,
            **kwargs,
        )
        return adapter, model

    return make


def _cold(adapter, lease, *, pipeline=True):
    update = adapter.prepare(_request(lease, 0, reset=True))
    assert adapter.prefixes(update) == (None,)
    if pipeline:
        update = adapter.bind_current(update, _capture(0))
    adapter.commit(update)


def _drain(adapter, count: int):
    timings = []
    for _ in range(count):
        work = adapter.next_completed()
        assert work is not None
        timings.append(adapter.run_completed(work))
    assert adapter.next_completed() is None
    return timings


def test_sixteen_incremental_candidates_match_full_scan_and_publish_once(make_adapter):
    adapter, model = make_adapter()
    reference, reference_model = make_adapter(enabled=False, reference=model)
    lease = _lease()
    _cold(adapter, lease)
    _cold(reference, lease, pipeline=False)
    before = adapter.snapshot
    parameters = {key: value.detach().clone() for key, value in model.state_dict().items()}
    for step in range(16):
        receipt = adapter.admit_completed(lease, _completed(lease, step))
        assert receipt["accepted"] is True and receipt["duplicate"] is False
    assert adapter.pipeline_ready(lease["session_id"], 16) is False
    with torch.inference_mode():
        for step in range(16):
            work = adapter.next_completed()
            assert work.lease == lease
            timing = adapter.run_completed(work)
            assert isinstance(timing, dict)
            assert adapter.snapshot is before
            assert adapter.memory._records[lease["session_id"]].state is None
            assert adapter.pipeline_ready(lease["session_id"], 16) is (step == 15)
    assert adapter.next_completed() is None
    assert model.tokenizer_vision_gen.encoded == list(range(1, 16))
    assert model.calls == [{"steps": 1, "create_graph": False, "grad": True, "inference": False}] * 16
    update = adapter.prepare(_request(lease, 16))
    baseline = reference.prepare(_request(lease, 16))
    assert [call["steps"] for call in reference_model.calls] == [16]
    assert adapter.snapshot is before
    assert adapter.status(update)["pipeline"] == "per_frame_v1"
    assert adapter.status(update)["encoded_steps"] == 15
    assert adapter.status(update)["adapted_steps"] == 16
    torch.testing.assert_close(adapter.prefixes(update)[0], reference.prefixes(baseline)[0], rtol=2e-5, atol=2e-6)
    for actual, expected in zip(update.memory.replacement.state, baseline.memory.replacement.state, strict=True):
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
        assert not actual.requires_grad and actual.grad_fn is None
    for key in ("inner_loss_mean", "fast_update_norm", "fast_state_norm"):
        assert adapter.status(update)[key] == pytest.approx(reference.status(baseline)[key], rel=2e-5, abs=2e-6)
    assert model.calls == [{"steps": 1, "create_graph": False, "grad": True, "inference": False}] * 16
    rebound = adapter.bind_current(update, _capture(16))
    assert rebound is not update and rebound.memory is update.memory
    with pytest.raises(RuntimeError, match="pending|capability"):
        adapter.commit(update)
    response = CachedMemoryResponse("request-key", "c" * 64, b'{"committed":true}')
    adapter.commit(rebound, response)
    snapshot = adapter.snapshot
    assert snapshot is not before
    assert snapshot.fast[lease["session_id"]] is rebound.memory.replacement
    assert snapshot.visual[lease["session_id"]] is rebound.visual_replacement
    assert snapshot.responses[lease["session_id"]] is response
    assert adapter.pipeline_info()["items"] == adapter.pipeline_info()["bytes"] == 0
    for step in range(16):
        assert adapter.admit_completed(lease, _completed(lease, step))["duplicate"] is True
    assert len(model.calls) == 16 and model.tokenizer_vision_gen.encoded == list(range(1, 16))
    adapter.admit_completed(lease, _completed(lease, 16))
    _drain(adapter, 1)
    assert len(model.calls) == 17 and model.tokenizer_vision_gen.encoded == list(range(1, 16))
    adapter.cancel_completed(lease)
    assert all(parameter.grad is None for parameter in model.parameters())
    for key, value in model.state_dict().items():
        assert torch.equal(value, parameters[key])


def test_duplicate_and_replay_never_reencode_or_readapt(make_adapter):
    adapter, model = make_adapter()
    lease = _lease()
    _cold(adapter, lease)
    row = _completed(lease, 0)
    adapter.admit_completed(lease, row)
    for _ in range(2):
        receipt = adapter.admit_completed(copy.deepcopy(lease), copy.deepcopy(row))
        assert receipt["duplicate"] is True
    _drain(adapter, 1)
    assert adapter.admit_completed(lease, row)["duplicate"] is True
    request = _request(lease, 1)
    update = adapter.bind_current(adapter.prepare(request), _capture(1))
    adapter.commit(update)
    assert adapter.admit_completed(lease, row)["duplicate"] is True
    replay = adapter.prepare(copy.deepcopy(request))
    assert adapter.status(replay)["replay"] is True
    assert adapter.status(replay)["encoded_steps"] == adapter.status(replay)["adapted_steps"] == 0
    adapter.commit(replay)
    assert len(model.calls) == 1 and model.tokenizer_vision_gen.encoded == []
    changed = copy.deepcopy(row)
    changed["executed_action"][0] += 0.1
    with pytest.raises((ValueError, RuntimeError)):
        adapter.admit_completed(lease, changed)
    assert len(model.calls) == 1


@pytest.mark.parametrize("stage", ["queued", "computed", "committed"])
def test_changed_duplicate_bytes_fail_at_every_stage(make_adapter, stage):
    adapter, model = make_adapter()
    lease = _lease()
    _cold(adapter, lease)
    adapter.admit_completed(lease, _completed(lease, 0))
    if stage != "queued":
        _drain(adapter, 1)
    if stage == "committed":
        adapter.commit(adapter.bind_current(adapter.prepare(_request(lease, 1)), _capture(1)))
    changed = _completed(lease, 0)
    changed["composite_image"] = "frame-1"
    with pytest.raises((ValueError, RuntimeError)):
        adapter.admit_completed(lease, changed)
    assert model.tokenizer_vision_gen.encoded == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("session_id", "foreign-session"),
        ("episode_id", "foreign-episode"),
        ("image_size", 512),
        ("preprocess_profile", "foreign-profile"),
        ("source_step", 1),
        ("source_step", True),
        ("source_step", -1),
        ("executed_action", [0.0] * 14),
        ("executed_action", [float("nan")] * 15),
    ],
)
def test_invalid_completed_identity_and_chronology_never_enqueue(make_adapter, field, value):
    adapter, model = make_adapter()
    lease = _lease()
    _cold(adapter, lease)
    row = _completed(lease, 0)
    row[field] = value
    with pytest.raises((ValueError, RuntimeError)):
        adapter.admit_completed(lease, row)
    assert adapter.pipeline_info()["items"] == adapter.pipeline_info()["bytes"] == 0
    assert adapter.next_completed() is None and model.calls == []


@pytest.mark.parametrize(
    "field",
    [
        "server_epoch",
        "run_lease",
        "manifest_digest",
        "slot_id",
        "generation",
        "task",
        "episode_index",
        "attempt_id",
        "capability",
    ],
)
def test_same_session_cannot_change_lease_identity(make_adapter, field):
    adapter, _ = make_adapter()
    lease = _lease()
    _cold(adapter, lease)
    adapter.admit_completed(lease, _completed(lease, 0))
    changed = dict(lease)
    changed[field] = 2 if field in ("generation", "episode_index") else "f" * 64
    with pytest.raises((ValueError, RuntimeError)):
        adapter.admit_completed(changed, _completed(lease, 0))
    assert adapter.pipeline_info()["items"] == 1


@pytest.mark.parametrize("extra", [False, True])
def test_completed_admission_requires_exact_full_lease_shape(make_adapter, extra):
    adapter, _ = make_adapter()
    lease = _lease()
    _cold(adapter, lease)
    malformed = dict(lease)
    if extra:
        malformed["foreign"] = "unexpected"
    else:
        del malformed["capability"]
    with pytest.raises((ValueError, RuntimeError)):
        adapter.admit_completed(malformed, _completed(lease, 0))
    assert adapter.pipeline_info()["items"] == adapter.pipeline_info()["bytes"] == 0


def test_admission_freezes_caller_owned_payload_and_assignment(make_adapter):
    adapter, model = make_adapter()
    lease = _lease()
    _cold(adapter, lease)
    supplied, row = copy.deepcopy(lease), _completed(lease, 0)
    adapter.admit_completed(supplied, row)
    supplied["generation"] += 1
    row["composite_image"] = "frame-1"
    row["executed_action"][0] = 100.0
    work = adapter.next_completed()
    assert work.lease == lease
    adapter.run_completed(work)
    update = adapter.prepare(_request(lease, 1))
    assert adapter.status(update)["adapted_steps"] == 1
    assert model.tokenizer_vision_gen.encoded == []
    adapter.abort(update)


def test_h16_and_global_item_byte_bounds_are_fail_closed(make_adapter):
    adapter, _ = make_adapter()
    lease = _lease()
    _cold(adapter, lease)
    for step in range(16):
        adapter.admit_completed(lease, _completed(lease, step))
    with pytest.raises((ValueError, RuntimeError)):
        adapter.admit_completed(lease, _completed(lease, 16))
    assert adapter.pipeline_info()["items"] == 16
    adapter.cancel_completed(lease)
    assert adapter.pipeline_info()["items"] == adapter.pipeline_info()["bytes"] == 0
    bounded, _ = make_adapter(pipeline_max_items=1)
    for index in range(2):
        _cold(bounded, _lease(index))
    bounded.admit_completed(_lease(), _completed(_lease(), 0))
    with pytest.raises((ValueError, RuntimeError)):
        bounded.admit_completed(_lease(1), _completed(_lease(1), 0))
    assert bounded.pipeline_info()["items"] == 1
    byte_bounded, _ = make_adapter(pipeline_max_bytes=1)
    _cold(byte_bounded, lease)
    with pytest.raises((ValueError, RuntimeError)):
        byte_bounded.admit_completed(lease, _completed(lease, 0))
    assert byte_bounded.pipeline_info()["bytes"] == 0


def test_replan_requires_full_completed_frontier_and_exact_original_rows(make_adapter):
    adapter, model = make_adapter()
    lease = _lease()
    _cold(adapter, lease)
    for step in range(2):
        adapter.admit_completed(lease, _completed(lease, step))
    first = adapter.next_completed()
    assert first is not None
    adapter.run_completed(first)
    assert adapter.pipeline_info()["queued"] == 1
    with pytest.raises((ValueError, RuntimeError)):
        adapter.prepare(_request(lease, 2))
    assert adapter.memory._pending == {} and adapter._visual_pending == {}
    _drain(adapter, 1)
    changed = _request(lease, 2)
    changed["local_memory"]["evidence"][1]["executed_action"][0] += 1
    with pytest.raises((ValueError, RuntimeError)):
        adapter.prepare(changed)
    assert adapter.memory._pending == {} and adapter._visual_pending == {}
    update = adapter.prepare(_request(lease, 2))
    assert adapter.status(update)["adapted_steps"] == 2
    assert adapter.status(update)["encoded_steps"] == 1
    assert len(model.calls) == 2
    adapter.abort(update)
    assert adapter.memory._records[lease["session_id"]].consumer_step == 0
    assert adapter.pipeline_info()["items"] == adapter.pipeline_info()["bytes"] == 0


def test_boundary_digest_mismatch_cannot_fallback_to_encode1(make_adapter):
    adapter, model = make_adapter()
    lease = _lease()
    _cold(adapter, lease)
    row = _completed(lease, 0)
    row["composite_image"] = "frame-1"
    adapter.admit_completed(lease, row)
    work = adapter.next_completed()
    with pytest.raises((ValueError, RuntimeError)):
        adapter.run_completed(work)
    assert model.tokenizer_vision_gen.encoded == [] and model.calls == []
    assert adapter.memory._records[lease["session_id"]].consumer_step == 0
    assert adapter.pipeline_info()["active"] == 1 and adapter.pipeline_info()["bytes"] > 0
    # The CPU model owner has now observed the exception and proven quiescence.
    adapter.discard_completed(work)
    with pytest.raises((ValueError, RuntimeError)):
        adapter.prepare(_request(lease, 1))
    adapter.cancel_completed(lease)
    assert adapter.pipeline_info()["items"] == adapter.pipeline_info()["bytes"] == 0


def test_current_boundary_must_bind_before_transaction_publication(make_adapter):
    adapter, model = make_adapter()
    lease = _lease()
    update = adapter.prepare(_request(lease, 0, reset=True))
    before = adapter.snapshot
    with pytest.raises((ValueError, RuntimeError), match="current|boundary|capture"):
        adapter.commit(update)
    assert adapter.snapshot is before
    assert lease["session_id"] not in adapter.snapshot.fast
    assert model.calls == [] and model.tokenizer_vision_gen.encoded == []
    adapter.abort(update)


def test_bound_visual_does_not_alias_mutable_capture_tensor(make_adapter):
    adapter, model = make_adapter()
    lease = _lease()
    captured = _capture(0)
    update = adapter.bind_current(adapter.prepare(_request(lease, 0, reset=True)), captured)
    captured.visual96.fill_(float("nan"))
    adapter.commit(update)
    adapter.admit_completed(lease, _completed(lease, 0))
    _drain(adapter, 1)
    prepared = adapter.prepare(_request(lease, 1))
    assert torch.isfinite(adapter.prefixes(prepared)[0]).all()
    assert model.tokenizer_vision_gen.encoded == []
    adapter.abort(prepared)


def test_abort_of_stale_wrapper_cannot_destroy_rebound_pending_capability(make_adapter):
    adapter, _ = make_adapter()
    lease = _lease()
    original = adapter.prepare(_request(lease, 0, reset=True))
    rebound = adapter.bind_current(original, _capture(0))
    adapter.abort(original)
    assert adapter._visual_pending[lease["session_id"]] is rebound
    assert adapter.memory._pending[lease["session_id"]] is rebound.memory
    adapter.commit(rebound)
    assert adapter.memory._records[lease["session_id"]] is rebound.memory.replacement


def test_cancelled_popped_work_remains_owned_until_explicit_discard(make_adapter):
    adapter, model = make_adapter()
    lease = _lease()
    _cold(adapter, lease)
    before = adapter.snapshot
    adapter.admit_completed(lease, _completed(lease, 0))
    work = adapter.next_completed()
    assert adapter.cancel_completed(lease) is True
    info = adapter.pipeline_info()
    assert info["queued"] == 0 and info["active"] == info["items"] == 1 and info["bytes"] > 0
    assert adapter.pipeline_ready(lease["session_id"], 1) is False
    assert adapter.snapshot is before and model.calls == []
    # Popped but never run: the model owner can prove there is no device work.
    adapter.discard_completed(work)
    info = adapter.pipeline_info()
    assert info["active"] == info["items"] == info["bytes"] == 0
    with pytest.raises(RuntimeError, match="active|capability"):
        adapter.run_completed(work)
    assert adapter.snapshot is before and model.calls == []


def test_scan_failure_propagates_without_publishing_or_poisoning_peer(make_adapter):
    adapter, model = make_adapter()
    lease, peer = _lease(), _lease(1)
    _cold(adapter, lease)
    _cold(adapter, peer)
    before = adapter.snapshot
    adapter.admit_completed(lease, _completed(lease, 0))

    def failed_scan():
        raise RuntimeError("synthetic model-owned scan failure")

    model.before_scan = failed_scan
    work = adapter.next_completed()
    with pytest.raises(RuntimeError, match="synthetic model-owned scan failure"):
        adapter.run_completed(work)
    assert adapter.snapshot is before
    assert adapter.memory._pending == {} and adapter._visual_pending == {}
    assert adapter.pipeline_info()["active"] == 1 and adapter.pipeline_info()["bytes"] > 0
    adapter.discard_completed(work)
    with pytest.raises((ValueError, RuntimeError)):
        adapter.prepare(_request(lease, 1))
    model.before_scan = None
    adapter.admit_completed(peer, _completed(peer, 0))
    _drain(adapter, 1)
    assert adapter.pipeline_ready(peer["session_id"], 1) is True
    assert adapter.snapshot is before
    adapter.cancel_completed()
    assert adapter.pipeline_info()["items"] == adapter.pipeline_info()["bytes"] == 0


def test_active_cancel_keeps_charge_until_worker_quiescent_and_cannot_resurrect(make_adapter):
    adapter, model = make_adapter()
    lease, peer = _lease(), _lease(1)
    _cold(adapter, lease)
    _cold(adapter, peer)
    before = adapter.snapshot
    adapter.admit_completed(lease, _completed(lease, 0))
    work = adapter.next_completed()
    entered, release, admitted = threading.Event(), threading.Event(), threading.Event()
    errors = []

    def parked_scan():
        entered.set()
        assert release.wait(5), "test failed to release model-owned scan"

    def run():
        try:
            adapter.run_completed(work)
        except BaseException as error:
            errors.append(error)

    def admit_peer():
        try:
            adapter.admit_completed(peer, _completed(peer, 0))
        except BaseException as error:
            errors.append(error)
        finally:
            admitted.set()

    model.before_scan = parked_scan
    worker, admission = threading.Thread(target=run), threading.Thread(target=admit_peer)
    worker.start()
    try:
        assert entered.wait(3), "candidate worker did not reach model-owned scan"
        admission.start()
        assert admitted.wait(3), "completed admission waited for GPU/model computation"
        adapter.cancel_completed(lease)
        assert adapter.pipeline_info()["active"] == 1
        assert adapter.pipeline_info()["bytes"] > 0
        assert adapter.snapshot is before
    finally:
        release.set()
        worker.join(5)
        if admission.ident is not None:
            admission.join(5)
    assert not worker.is_alive() and not admission.is_alive()
    assert all(isinstance(error, RuntimeError) for error in errors)
    assert adapter.snapshot is before
    assert adapter.memory._records[lease["session_id"]].state is None
    model.before_scan = None
    _drain(adapter, 1)
    assert adapter.pipeline_ready(peer["session_id"], 1) is True
    adapter.cancel_completed()
    assert adapter.pipeline_info()["active"] == 0
    assert adapter.pipeline_info()["items"] == adapter.pipeline_info()["bytes"] == 0


def test_reset_discards_unpublished_candidates_and_preserves_other_session(make_adapter):
    adapter, _ = make_adapter()
    lease, peer = _lease(), _lease(1)
    _cold(adapter, lease)
    _cold(adapter, peer)
    peer_record = adapter.memory._records[peer["session_id"]]
    adapter.admit_completed(lease, _completed(lease, 0))
    _drain(adapter, 1)
    adapter.reset(lease["session_id"])
    assert lease["session_id"] not in adapter.snapshot.fast
    assert adapter.memory._records[peer["session_id"]] is peer_record
    assert adapter.pipeline_info()["items"] == adapter.pipeline_info()["bytes"] == 0
    renewed = {**lease, "episode_id": "renewed-episode", "generation": 2}
    _cold(adapter, renewed)
    adapter.admit_completed(renewed, _completed(renewed, 0))
    _drain(adapter, 1)
    assert adapter.pipeline_ready(renewed["session_id"], 1) is True


def test_legacy_required_and_no_ttt_keep_pipeline_off(make_adapter):
    legacy, model = make_adapter(enabled=None)
    lease = _lease()
    _cold(legacy, lease, pipeline=False)
    update = legacy.prepare(_request(lease, 1))
    assert "pipeline" not in legacy.status(update)
    assert legacy.status(update)["encoded_steps"] == 1
    assert model.tokenizer_vision_gen.encoded == [0]
    legacy.abort(update)
    off = policy.RoboCasaLocalMemoryPolicyAdapter(SimpleNamespace(), mode="off", decode_image=_decode)
    assert off.prepare({}) is None and off.info() == {"enabled": False, "mode": "off"}
    with pytest.raises((ValueError, RuntimeError)):
        off.admit_completed(lease, _completed(lease, 0))
    with pytest.raises(ValueError, match="off|required|pipeline"):
        policy.RoboCasaLocalMemoryPolicyAdapter(
            SimpleNamespace(), mode="off", decode_image=_decode, pipeline_enabled=True
        )
