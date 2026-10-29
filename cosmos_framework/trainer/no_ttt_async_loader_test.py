"""Real spawn/Source/Transform ordering; fake compute is exclusively a CPU oracle."""

from __future__ import annotations

import hashlib
import io
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from cosmos_framework.model.generator.mot.robocasa_exact_window_local import ExactWindowRankPlanner
from cosmos_framework.trainer.no_ttt_async_loader import (
    AsyncJob,
    NoTTTAsyncLoader,
    PreparedDataset,
    _transport_bytes,
    _transport_tree,
    _versions,
)
from cosmos_framework.trainer.no_ttt_async_recipe import AsyncWorkerState
from cosmos_framework.trainer.no_ttt_async_test_support import (
    BaselineRawRecipe,
    SyntheticAsyncRecipe,
    build_synthetic_producer,
)
from cosmos_framework.trainer.no_ttt_cpu_loader import NoTTTCpuLoader, _require_same_rng, _rng_snapshot
from cosmos_framework.utils.no_ttt_loader_profile import LoaderProfile


def setup(*, count=9, t=2, rank=0, world=4, **kwargs):
    recipe = SyntheticAsyncRecipe(count=count, t=t, world=world, **kwargs)
    producer = build_synthetic_producer(count=count, t=t, world=world, rate=recipe.rate)
    planner = ExactWindowRankPlanner(producer.catalog, rank=rank, world_size=world, b_stream=8, active_ga=4)
    return producer, planner, recipe


def equal_tree(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif hasattr(left, "__dataclass_fields__"):
        for name in left.__dataclass_fields__:
            equal_tree(getattr(left, name), getattr(right, name))
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal_tree(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            equal_tree(a, b)
    else:
        assert left == right


def seed(value=81):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def model_draws():
    return random.random(), float(np.random.rand()), float(torch.rand(()))


def consume(producer, planner, *, loader=None, windows=3, compute=None):
    frontier = planner.initial_frontier()
    output, draws, records = [], [], []
    for iteration in range(windows):
        plan = planner.plan_window(frontier)
        if loader is not None:
            before = _rng_snapshot()
            loader.begin(plan)
            _require_same_rng(before)
        for member, requests in enumerate(plan.members):
            for request in requests:
                segment = producer.produce(request) if loader is None else loader.materialize(request)
                output.append(segment)
            draws.append(model_draws())
            if compute:
                compute(iteration, member)
        frontier = plan.candidate_frontier  # fake successful optimizer boundary
        if loader is not None:
            records.append(loader.finish())
            before = _rng_snapshot()
            if hasattr(loader, "checkpoint_boundary"):
                loader.checkpoint_boundary()
            _require_same_rng(before)
    return output, draws, frontier, _rng_snapshot(), records


@pytest.mark.parametrize("workers", [1, 2, 4, 8])
def test_full_and_terminal_partial_three_windows_exact_data_and_interleaved_rng(workers):
    producer, planner, recipe = setup(count=17, t=16)
    seed()
    expected = consume(producer, planner)
    seed()
    loader = NoTTTAsyncLoader(producer, LoaderProfile(workers, 1, True, 90), planner, recipe=recipe)
    try:
        actual = consume(producer, planner, loader=loader)
        equal_tree(expected[:4], actual[:4])
        assert any(int(segment.consumer_valid.sum()) == 1 for segment in actual[0])
        assert all(len(record["worker_pids"]) == workers for record in actual[4])
        assert len(loader.sampler.jobs) <= 64
        assert loader.preview is not None and loader.pending is None
        assert all(record["transform_host_ms"] == 0 and record["source_read_ms"] == 0 for record in actual[4])
    finally:
        loader.close()
    assert all(row["exitcode"] is not None and not Path(f"/proc/{row['pid']}").exists() for row in loader.last_cleanup)


def test_actual_source_transform_next_member_and_next_window_overlap(tmp_path):
    producer, planner, recipe = setup(count=19, source_delay=0.003, transform_delay=0.004)
    compute_events = []

    def compute(iteration, member):
        started = time.monotonic()
        # A real CPU-only callback stands in for device work. Merely spawning
        # workers or sleeping without measured Source/Transform is not a pass.
        digest = b"fake_gpu_only"
        while time.monotonic() - started < 0.13:
            digest = hashlib.sha256(digest * 128).digest()
        compute_events.append(
            {
                "stage": "fake_gpu_compute",
                "window": iteration,
                "member": member,
                "start": started,
                "end": time.monotonic(),
                "pid": os.getpid(),
            }
        )

    loader = NoTTTAsyncLoader(producer, LoaderProfile(2, 1, True, 90), planner, recipe=recipe)
    try:
        result = consume(producer, planner, loader=loader, compute=compute)
    finally:
        loader.close()
    events = [event for record in result[4] for event in record["trace"]]
    overlaps = {"next_member": {}, "next_window": {}}
    for relation in overlaps:
        for stage in ("cpu_source", "cpu_transform"):
            values = [
                max(0.0, min(event["end"], gpu["end"]) - max(event["start"], gpu["start"])) * 1000
                for event in events
                if event["stage"] == stage
                for gpu in compute_events
                if (
                    event["window"] == gpu["window"] and event["member"] > gpu["member"]
                    if relation == "next_member"
                    else event["window"] == gpu["window"] + 1
                )
            ]
            overlaps[relation][stage + "_overlap_ms"] = sum(values)
            assert sum(values) > 0, (relation, stage)
    for event in events:
        if event["stage"] == "committed_iter":
            assert event["at"] >= max(c["end"] for c in compute_events if c["window"] == event["window"])
    # Independent serial reference: source and transform have no concurrency
    # with its own fake model compute interval and the original RNG agrees.
    sync_state = AsyncWorkerState(
        build_synthetic_producer(
            count=recipe.count,
            t=recipe.t,
            world=recipe.world,
            rate=0,
            source_delay=recipe.source_delay,
            transform_delay=recipe.transform_delay,
        ),
        rate=recipe.rate,
        cache_reader=None,
    )
    sync_trace = []
    sync_frontier = planner.initial_frontier()
    for iteration in range(3):
        plan = planner.plan_window(sync_frontier)
        for member, requests in enumerate(plan.members):
            for request in requests:
                _, stage_events = sync_state.prepare(request)
                sync_trace.extend(stage_events)
            started = time.monotonic()
            hashlib.sha256(b"synchronous_fake_compute" * 1024).digest()
            sync_trace.append({"stage": "fake_gpu_compute", "start": started, "end": time.monotonic()})
        sync_frontier = plan.candidate_frontier
    assert all(
        min(a["end"], b["end"]) <= max(a["start"], b["start"])
        for a in sync_trace
        if a["stage"].startswith("cpu_")
        for b in sync_trace
        if b["stage"] == "fake_gpu_compute"
    )
    report = {
        "schema": "nottt_async_cpu_fake_compute_timeline_v1",
        "real_gpu": False,
        "workers": 2,
        "windows": 3,
        "overlap": overlaps,
        "async_trace": events + compute_events,
        "sync_trace": sync_trace,
        "main_wait_ms": sum(r["raw_wait_host_ms"] for r in result[4]),
    }
    path = tmp_path / "overlap_trace.json"
    path.write_text(json.dumps(report, indent=2))
    external = os.environ.get("PSM_NOTTT_ASYNC_EVIDENCE")
    if external:
        directory = Path(external)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "overlap_trace.json").write_text(json.dumps(report, indent=2))
    print("CPU_FAKE_COMPUTE_OVERLAP " + json.dumps(overlaps), flush=True)


def test_unchanged_raw_worker_baseline_fails_full_pipeline_overlap_target(tmp_path):
    producer = build_synthetic_producer(count=19, t=2, world=4, source_delay=0.003, transform_delay=0.004)
    raw = producer.catalog.raw

    # The old parent source packet contains only non-visual source fields;
    # cached latent and geometry are reconstructed at its explicit RAW double.
    def source_snapshot(flat):
        row = raw[flat]
        return {"flat": flat, "action": row["action"], "ai_caption": row["ai_caption"]}

    raw.source_reader.read_at = source_snapshot
    planner = ExactWindowRankPlanner(producer.catalog, rank=0, world_size=4, b_stream=8, active_ga=4)
    events = []
    for name, stage in (("prepare", "cpu_source"), ("materialize", "cpu_transform")):
        original = getattr(producer, name)

        def track(*args, _original=original, _stage=stage, **kwargs):
            started = time.monotonic()
            value = _original(*args, **kwargs)
            events.append({"stage": _stage, "start": started, "end": time.monotonic(), "pid": os.getpid()})
            return value

        setattr(producer, name, track)

    def compute(iteration, member):
        started = time.monotonic()
        digest = b"baseline_fake_gpu"
        while time.monotonic() - started < 0.04:
            digest = hashlib.sha256(digest * 128).digest()
        events.append(
            {
                "stage": "fake_gpu_compute",
                "window": iteration,
                "member": member,
                "start": started,
                "end": time.monotonic(),
            }
        )

    loader = NoTTTCpuLoader(producer, LoaderProfile(2, 1, True, 90), recipe=BaselineRawRecipe())
    try:
        consume(producer, planner, loader=loader, compute=compute)
    finally:
        loader.close()
    assert all(
        min(a["end"], b["end"]) <= max(a["start"], b["start"])
        for a in events
        if a["stage"].startswith("cpu_")
        for b in events
        if b["stage"] == "fake_gpu_compute"
    )
    report = {
        "expected_target_verdict": "RED_NO_FULL_PIPELINE_OVERLAP",
        "real_gpu": False,
        "baseline_backend": "unchanged_NoTTTCpuLoader",
        "raw_recipe": "explicit_synthetic_double",
        "source_transform_owner": "parent",
        "windows": 3,
        "trace": events,
    }
    external = Path(os.environ.get("PSM_NOTTT_ASYNC_EVIDENCE", str(tmp_path)))
    external.mkdir(parents=True, exist_ok=True)
    (external / "old_raw_baseline_target_red.json").write_text(json.dumps(report, indent=2))


@pytest.mark.parametrize(
    "fault",
    ["error", "crash", "timeout", "ignore_term", "identity", "cache_change", "transform_error", "transform_rng"],
)
def test_source_transform_faults_close_all_owned_workers_and_preserve_foreign_process(fault, tmp_path):
    producer, planner, recipe = setup(fault=fault, witness=str(tmp_path))
    loader = NoTTTAsyncLoader(
        producer, LoaderProfile(2, 1, True, 2 if fault in ("timeout", "ignore_term") else 90), planner, recipe=recipe
    )
    foreign = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    before = _rng_snapshot()
    started = time.monotonic()
    try:
        loader.begin(planner.plan_window(planner.initial_frontier()))
        if fault in ("timeout", "ignore_term"):
            until = time.monotonic() + 45
            while len(list(tmp_path.glob("worker_*.json"))) < 2 and time.monotonic() < until:
                time.sleep(0.02)
            assert len(list(tmp_path.glob("worker_*.json"))) == 2
        with pytest.raises((OSError, RuntimeError, ValueError)):
            loader.materialize(loader.pending[0])
        _require_same_rng(before)
        assert loader.closed and loader.pending is None
        assert foreign.poll() is None
        assert time.monotonic() - started < 100
        assert all(row["exitcode"] is not None for row in loader.last_cleanup)
        if fault == "ignore_term":
            assert any(row["exitcode"] == -9 for row in loader.last_cleanup)
    finally:
        loader.close()
        foreign.terminate()
        foreign.wait(timeout=5)


def test_request_and_preview_predecessor_mismatch_fail_before_publication():
    producer, planner, recipe = setup()
    loader = NoTTTAsyncLoader(producer, LoaderProfile(1, 1, True, 90), planner, recipe=recipe)
    plan = planner.plan_window(planner.initial_frontier())
    before = _rng_snapshot()
    try:
        loader.begin(plan)
        request = loader.pending[0]
        wrong = replace(request, binding_epoch=request.binding_epoch + 1)
        with pytest.raises(ValueError, match="order"):
            loader.materialize(wrong)
        _require_same_rng(before)
        assert loader.closed
    finally:
        loader.close()
    loader = NoTTTAsyncLoader(producer, LoaderProfile(1, 1, True, 90), planner, recipe=recipe)
    try:
        loader.preview = planner.plan_window(plan.candidate_frontier)
        with pytest.raises(ValueError, match="predecessor"):
            loader.begin(plan)
        assert loader.closed and loader.iterator is None
    finally:
        loader.close()


@pytest.mark.parametrize("persistent", [False, True])
def test_persistence_option_and_speculative_checkpoint_are_rng_neutral(persistent):
    producer, planner, recipe = setup()
    loader = NoTTTAsyncLoader(producer, LoaderProfile(1, 1, persistent, 90), planner, recipe=recipe)
    try:
        consume(producer, planner, loader=loader, windows=2)
        before = _rng_snapshot()
        loader.checkpoint_boundary()
        _require_same_rng(before)
        assert (loader.iterator is not None) is persistent
    finally:
        loader.close()


def test_compact_geometry_does_not_allocate_dense_video_and_packet_limit_precedes_clone(monkeypatch):
    from cosmos_framework.trainer import no_ttt_async_loader as module

    video = torch.zeros((), dtype=torch.uint8).expand(3, 17, 192, 320)
    out = _transport_tree({"video": video})
    assert out["video"].untyped_storage().nbytes() == 1
    assert _transport_bytes({"video": video}) < 1024
    producer, planner, recipe = setup()
    state = AsyncWorkerState(
        build_synthetic_producer(t=recipe.t, count=recipe.count, rate=0), rate=recipe.rate, cache_reader=None
    )
    dataset = PreparedDataset(recipe, 1024)
    dataset.state = state
    request = planner.plan_window(planner.initial_frontier()).members[0][0]

    def forbidden(*args):
        raise AssertionError("transport clone ran before byte bound")

    monkeypatch.setattr(module, "_transport_tree", forbidden)
    with pytest.raises(ValueError, match="before transport cloning"):
        dataset[AsyncJob(0, 0, request)]


class PacketIterator:
    def __init__(self, packet):
        self.packet = packet

    def __next__(self):
        return self.packet


class CompletedOwnedProcess:
    """An explicit completed-process double for packet-only negative tests."""

    exitcode = 0

    def __init__(self, pid):
        self.pid = pid

    def is_alive(self):
        return False

    def join(self, timeout):
        pass


@pytest.mark.parametrize("mutation", ["token", "window", "generation", "timing", "consumer", "config", "duplicate"])
def test_stale_reordered_corrupt_packet_rejected_before_cfg_rng(mutation):
    producer, planner, recipe = setup()
    loader = NoTTTAsyncLoader(producer, LoaderProfile(1, 1, True, 90), planner, recipe=recipe)
    request = planner.plan_window(planner.initial_frontier()).members[0][0]
    state = AsyncWorkerState(
        build_synthetic_producer(t=recipe.t, count=recipe.count, rate=0), rate=recipe.rate, cache_reader=None
    )
    dataset = PreparedDataset(recipe, loader.profile.max_packet_bytes)
    dataset.state = state
    packet = dataset[AsyncJob(0, 0, request)]
    packet["pid"] += 1000000
    loader.context.processes = [CompletedOwnedProcess(packet["pid"])]
    for event in packet["events"]:
        event["pid"] = packet["pid"]
    loader.pending = (request,)
    loader.metrics = {"raw_wait_host_ms": 0.0, "worker_source_ms_sum": 0.0, "worker_transform_ms_sum": 0.0}
    if mutation in ("token", "window", "generation"):
        packet[mutation] = "foreign" if mutation == "token" else 9
    elif mutation == "timing":
        packet["events"][0]["end"] = -1
    elif mutation in ("consumer", "config"):
        segment, empty = torch.load(io.BytesIO(packet["packet"]), weights_only=False)
        if mutation == "consumer":
            segment.consumer_step[0, 0] = 123
        else:
            segment = replace(segment, segment_provenance=replace(segment.segment_provenance, config_digest="foreign"))
        output = io.BytesIO()
        torch.save((segment, empty), output)
        packet["packet"] = output.getvalue()
    else:
        loader.index = 1  # an already consumed result cannot be delivered again
    loader.iterator = PacketIterator(packet)
    before = _rng_snapshot()
    with pytest.raises((ValueError, RuntimeError)):
        loader.materialize(request)
    _require_same_rng(before)
    assert loader.closed


@pytest.mark.parametrize("which", ["cache", "source"])
def test_prepared_file_changed_after_worker_completion_is_rejected_before_rng(tmp_path, which):
    producer, planner, recipe = setup()
    paths = [tmp_path / "cache.pt", tmp_path / "source.parquet"]
    for path in paths:
        path.write_bytes(b"original")
    versions = _versions(paths)
    state = AsyncWorkerState(
        build_synthetic_producer(t=recipe.t, count=recipe.count, rate=0), rate=recipe.rate, cache_reader=None
    )
    loader = NoTTTAsyncLoader(producer, LoaderProfile(1, 1, True, 90), planner, recipe=recipe)
    request = planner.plan_window(planner.initial_frontier()).members[0][0]
    dataset = PreparedDataset(recipe, loader.profile.max_packet_bytes)
    dataset.state = state
    packet = dataset[AsyncJob(0, 0, request, versions)]
    packet["pid"] += 1000000
    loader.context.processes = [CompletedOwnedProcess(packet["pid"])]
    loader.pending = (request,)
    loader.metrics = {"raw_wait_host_ms": 0.0}
    loader._request_versions = lambda _: _versions(paths)
    loader.iterator = PacketIterator(packet)
    paths[0 if which == "cache" else 1].write_bytes(b"changed after CPU preparation")
    before = _rng_snapshot()
    with pytest.raises(ValueError, match="file changed"):
        loader.materialize(request)
    _require_same_rng(before)
    assert loader.closed


def test_missing_and_escaped_source_witness_files_reject_before_worker_start(tmp_path):
    producer, planner, recipe = setup()
    loader = NoTTTAsyncLoader(producer, LoaderProfile(1, 1, True, 90), planner, recipe=recipe)
    raw = producer.catalog.raw
    raw.catalog.cache_root = tmp_path
    raw.source_reader.source_root = tmp_path
    request = planner.plan_window(planner.initial_frontier()).members[0][0]
    raw.source_reader._bound[request.episode.key].data_file = "source.parquet"
    loader.file_witnesses = True
    with pytest.raises(FileNotFoundError):
        loader.begin(planner.plan_window(planner.initial_frontier()))
    assert loader.closed and loader.iterator is None
    loader = NoTTTAsyncLoader(producer, LoaderProfile(1, 1, True, 90), planner, recipe=recipe)
    loader.file_witnesses = True
    raw.source_reader._bound[request.episode.key].data_file = "../outside"
    with pytest.raises(ValueError, match="escaped"):
        loader._request_versions(request)
    loader.close()


def test_out_of_order_worker_completion_never_changes_consumer_rng_order(tmp_path):
    producer, planner, recipe = setup(fault="late_order", witness=str(tmp_path))
    seed(700)
    expected = consume(producer, planner, windows=1)
    seed(700)
    loader = NoTTTAsyncLoader(producer, LoaderProfile(2, 1, True, 90), planner, recipe=recipe)
    try:
        actual = consume(producer, planner, loader=loader, windows=1)
        equal_tree(expected[:4], actual[:4])
        events = [e for e in actual[4][0]["trace"] if e["stage"] == "cpu_transform"]
        assert events[1]["end"] < events[0]["start"], "slot1 really finished before admitted slot0"
        assert len(actual[4][0]["worker_pids"]) == 2
    finally:
        loader.close()
