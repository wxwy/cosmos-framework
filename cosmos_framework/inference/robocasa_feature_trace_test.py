"""CPU trace/output contracts; mocked CUDA query test is not GPU timing evidence."""

from __future__ import annotations

import builtins
import hashlib
import json
import os
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from cosmos_framework.inference.robocasa_feature_trace import TraceRecorder, get_trace, span, use_trace


def test_exact_raw_nanoseconds_and_overlap_are_not_presented_as_total_latency():
    clock = iter([10, 30])
    trace = TraceRecorder(clock_ns=lambda: next(clock))
    with trace.span("outer", stratum="first_no_history"):
        trace.record("overlap", 15, 25, blocking="background")
    raw = trace.snapshot()
    assert [(row["start_ns"], row["end_ns"]) for row in raw["events"]] == [(15, 25), (10, 30)]
    assert sum(row["duration_ns"] for row in raw["events"]) == 30
    assert raw["overlap_semantics"] == "intervals_overlap_do_not_sum_into_critical_path"
    assert "total_latency_ns" not in raw and raw["complete"]


def test_context_is_optional_nested_and_CPU_path_never_imports_torch(monkeypatch):
    original = builtins.__import__

    def no_torch(name, *args, **kwargs):
        if name == "torch":
            raise AssertionError("CPU trace imported torch")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_torch)
    first, second = TraceRecorder(), TraceRecorder()
    assert get_trace() is None
    with span("disabled"):
        pass
    with use_trace(first):
        with span("one"):
            assert get_trace() is first
        with use_trace(second), span("two"):
            assert get_trace() is second
        assert get_trace() is first
    assert get_trace() is None
    assert first.snapshot()["count"] == second.snapshot()["count"] == 1


def test_exception_identity_survives_trace_and_only_public_error_type_is_recorded():
    trace, sentinel = TraceRecorder(), RuntimeError("private exception content")
    with pytest.raises(RuntimeError) as result:
        with trace.span("failing"):
            raise sentinel
    assert result.value is sentinel
    event = trace.snapshot()["events"][0]
    assert event["fields"]["error_type"] == "RuntimeError"
    assert "private exception content" not in json.dumps(event)


@pytest.mark.parametrize(
    "field", ["owner_token", "signing_key", "control_capability", "actor_capability", "sessioncap", "key_hex"]
)
def test_trace_rejects_private_credentials_but_accepts_public_keys(field):
    trace = TraceRecorder()
    with pytest.raises(ValueError, match="public"):
        trace.event("unsafe", **{field: "never-persist"})
    trace.event("safe", request_key="a" * 64, key_id="public-id", source_rgb_sha256="b" * 64)
    assert trace.snapshot()["count"] == 1


@pytest.mark.parametrize("bound", ["events", "bytes"])
def test_bounded_trace_overflow_is_explicit_in_atomic_raw_and_small_metadata(tmp_path, bound):
    trace = TraceRecorder(max_events=2 if bound == "events" else 1000, max_bytes=32768)
    for index in range(100):
        trace.record("sample", index, index + 1, label="x" * 256, second="y" * 256)
    metadata = trace.dump(tmp_path / "raw.json")
    raw_bytes = (tmp_path / "raw.json").read_bytes()
    raw = json.loads(raw_bytes)
    assert len(raw_bytes) <= 32768 and raw["overflow_count"] > 0 and raw["complete"] is False
    assert metadata["sha256"] == hashlib.sha256(raw_bytes).hexdigest()
    assert metadata["count"] == len(raw["events"]) and "events" not in metadata
    assert not list(tmp_path.glob(".feature-trace-*"))


def test_raw_evidence_can_exceed_Core_JSON_bound_without_embedding_events(tmp_path):
    trace = TraceRecorder(max_events=2000, max_bytes=2 * 1024**2)
    for index in range(1000):
        trace.record("phase", index, index + 1, description="x" * 256)
    metadata = trace.dump(tmp_path / "large.json")
    assert metadata["bytes"] > 256 * 1024 and metadata["complete"]
    assert len(json.dumps(metadata)) < 16384 and "events" not in metadata


@pytest.mark.parametrize("leaf", ["existing", "symlink"])
def test_atomic_append_only_dump_preserves_foreign_bytes(tmp_path, leaf):
    foreign, target = tmp_path / "foreign.json", tmp_path / "trace.json"
    foreign.write_bytes(b"foreign")
    if leaf == "symlink":
        target.symlink_to(foreign)
    else:
        target.write_bytes(b"old")
    with pytest.raises(FileExistsError):
        TraceRecorder().dump(target)
    assert foreign.read_bytes() == b"foreign"
    assert target.is_symlink() if leaf == "symlink" else target.read_bytes() == b"old"
    assert not list(tmp_path.glob(".feature-trace-*"))


def test_atomic_dump_rejects_parent_symlink_and_cleans_failed_publication(tmp_path, monkeypatch):
    actual, linked = tmp_path / "actual", tmp_path / "linked"
    actual.mkdir()
    linked.symlink_to(actual, target_is_directory=True)
    with pytest.raises(ValueError, match="real"):
        TraceRecorder().dump(linked / "trace.json")
    sentinel = OSError("publish sentinel")

    def fail(*args, **kwargs):
        raise sentinel

    monkeypatch.setattr(os, "link", fail)
    with pytest.raises(OSError) as result:
        TraceRecorder().dump(actual / "trace.json")
    assert result.value is sentinel and not list(actual.iterdir())


def test_worker_ring_samples_deduplicate_and_overflow_or_clock_mismatch_remains_visible():
    trace = TraceRecorder(clock_domain="same-boot")
    ring = {
        "trace": [
            {"event": "ENQUEUE", "at_ns": 10, "source_step": 1},
            {"event": "ENCODE_END", "at_ns": 20, "source_step": 1},
        ],
        "trace_overflow": 0,
    }
    trace.merge("worker", ring, clock_domain="same-boot")
    trace.merge("worker", ring, clock_domain="same-boot")
    assert trace.snapshot()["count"] == 2
    ring["trace_overflow"] = 1
    trace.merge("worker", ring, clock_domain="same-boot")
    trace.merge("foreign", ring, clock_domain="another-boot")
    snapshot = trace.snapshot()
    assert snapshot["complete"] is False
    assert "producer_trace_overflow_or_incomplete" in snapshot["incomplete_reasons"]
    assert "producer_clock_domain_mismatch" in snapshot["incomplete_reasons"]


def test_policy_snapshot_keeps_overlapping_intervals_without_GPU_duration_invention():
    trace = TraceRecorder()
    trace.merge(
        "policy",
        {"events": [{"name": "outer", "start_ns": 10, "end_ns": 100}, {"name": "inner", "start_ns": 20, "end_ns": 30}]},
        deduplicate=False,
    )
    assert trace.snapshot()["count"] == 2
    assert all(row["producer"] == "policy" for row in trace.snapshot()["events"])


def test_small_Worker_encoder_snapshot_is_byte_bounded_and_nested_merge_is_once_only():
    encoder = TraceRecorder(max_events=8, max_bytes=4096, clock_domain="same-boot", producer="worker_vae")
    for index in range(40):
        encoder.record("history_vae", index, index + 1, field="x" * 256)
    profile = encoder.snapshot()
    assert len(json.dumps(profile, sort_keys=True, separators=(",", ":")).encode()) <= 4096
    assert not profile["complete"] and profile["overflow_count"] > 0
    output = TraceRecorder(clock_domain="same-boot")
    ring = {"trace": [{"event": "ENCODER_PROFILE", "at_ns": 50, "profile": profile}], "trace_overflow": 0}
    output.merge("worker", ring, clock_domain="same-boot")
    retained = output.snapshot()["count"]
    output.merge("worker", ring, clock_domain="same-boot")
    assert output.snapshot()["count"] == retained
    assert any(row["producer"] == "worker_vae" for row in output.snapshot()["events"])


def test_requested_CUDA_without_device_is_explicitly_unavailable_not_zero_timing(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))
    trace, executed = TraceRecorder(), []
    with trace.cuda_span("actual_gpu_operation"):
        executed.append(True)
    snapshot = trace.snapshot()
    assert executed == [True] and snapshot["cuda_unavailable"] == 1 and not snapshot["complete"]
    assert "gpu_elapsed_ms" not in snapshot["events"][0]["fields"]


def test_CUDA_events_record_actual_selected_stream_and_pending_query_never_syncs(monkeypatch):
    stream = SimpleNamespace(device="cuda:test", cuda_stream=7)
    events = []

    class Event:
        def __init__(self, **kwargs):
            events.append(self)
            self.recorded = None

        def record(self, actual):
            self.recorded = actual

        def query(self):
            return False

        def elapsed_time(self, other):
            raise AssertionError("pending event must not fabricate elapsed GPU time")

    cuda = SimpleNamespace(
        is_available=lambda: True,
        current_stream=lambda device: stream,
        Event=Event,
        device=lambda device: nullcontext(),
        stream=lambda actual: nullcontext(),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    trace = TraceRecorder()
    with trace.cuda_span("kernel", stream=stream):
        pass
    assert len(events) == 2 and all(event.recorded is stream for event in events)
    assert trace.resolve_cuda() == 1 and trace.snapshot()["complete"] is False
    assert trace.snapshot()["events"] == []


@pytest.mark.parametrize("profile", [None, "B", "overflow"])
def test_NativeEpisode_trace_is_opt_in_and_persists_bounded_raw_on_exit(tmp_path, profile):
    import numpy as np

    from cosmos_framework.simulation.robocasa.eval_v2_adapter import NativeEpisode

    class Env:
        def __init__(self):
            self.steps = 0

        def reset(self):
            return {}

        def _check_success(self):
            return self.steps > 0

        def get_ep_meta(self):
            return {"lang": "Close the fridge"}

    class Media:
        def prediction(self, *args):
            pass

        def close(self):
            return {}

    def policy(env, **kwargs):
        assert ("trace" in kwargs) == (profile is not None)
        local = kwargs["local_memory_client"]
        local.begin()
        status = local.payload()
        if "trace" in kwargs:
            assert get_trace() is kwargs["trace"]
            kwargs["trace"].event("synthetic_CPU_policy")
        env.steps = 1
        assert kwargs["success_check"](env)
        kwargs["on_prediction"]({"action": np.zeros((16, 15)).tolist(), "local_memory": status}, 0)
        kwargs["on_cleanup"]({"session_id": status["session_id"], "reset_ok": True})
        local.end()
        return True, 1, "Close the fridge"

    settings = dict(server_url="http://cpu", horizon=1, predict_timeout_s=1, save_rollout_mp4=False)
    if profile is not None:
        settings["single_latency_arm"] = "B"
    if profile == "overflow":
        settings["feature_trace_max_events"] = 1
    episode = NativeEpisode(Env(), settings, policy=policy, media_factory=lambda *args, **kwargs: Media())
    assert episode.initialize()["initial_success"] is False
    directory = tmp_path / "attempt"
    attempt = dict(
        task="CloseFridge", episode=0, attempt_id="attempt", stream_id="stream", seed=0, manifest_digest="a" * 64
    )
    result = episode.execute(attempt, reset_index=0, directory=directory)
    if profile is None:
        assert "single_latency_trace" not in result and not (directory / "feature_trace.json").exists()
    else:
        reference = result["single_latency_trace"]
        assert set(reference) == {"path", "sha256", "count"}
        raw = (directory / reference["path"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == reference["sha256"]
        metadata = json.loads((directory / "feature_trace_metadata.json").read_bytes())
        assert metadata["count"] == reference["count"]
        if profile == "overflow":
            assert metadata["overflow_count"] > 0 and not metadata["complete"]
