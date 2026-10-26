"""Real spawned CPU Worker parity and failure contracts, synthetic RAW only."""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cosmos_framework.model.generator.mot.robocasa_exact_window_local import (
    ExactWindowRankPlanner,
    ExactWindowSegmentProducer,
)
from cosmos_framework.model.generator.mot.robocasa_exact_window_local_test import _setup
from cosmos_framework.trainer import no_ttt_cpu_loader as module
from cosmos_framework.trainer.no_ttt_loader_test_support import SyntheticRecipe
from cosmos_framework.utils.no_ttt_loader_profile import LoaderProfile, profile_from_args


def source_setup(*, rank=0, world=1, count=33, t=16, b=8, ga=4):
    raw, transform, catalog, producer = _setup((count,) * max(256, b * world * 4), t=t)
    raw.source_reader.read_at = raw.__getitem__
    planner = ExactWindowRankPlanner(catalog, rank=rank, world_size=world, b_stream=b, active_ga=ga)
    return raw, transform, planner, producer


def rows(segment):
    return tuple(
        (p["task_class"], p["episode_index"], p["start_frame"], p["action_raw"], p["video_latent"], p.get("randoms"))
        for p in segment.consumer_payload[0]
        if p is not None
    )


def equal_rows(left, right):
    assert len(left) == len(right)
    for a, b in zip(left, right, strict=True):
        assert a[:3] == b[:3]
        torch.testing.assert_close(a[3], b[3], rtol=0, atol=0)
        torch.testing.assert_close(a[4], b[4], rtol=0, atol=0)
        assert a[5] == b[5]


def assert_closed(loader):
    loader.close()
    assert all(row["exitcode"] is not None and not Path(f"/proc/{row['pid']}").exists() for row in loader.last_cleanup)


@pytest.mark.parametrize("workers", [1, 2, 4, 8])
@pytest.mark.parametrize("world,count", [(1, 17), (1, 64), (4, 17), (4, 64)])
@torch.enable_grad()
def test_true_spawned_workers_match_legacy_ga4_rank_order_partial_and_rng(workers, world, count):
    from cosmos_framework.trainer.local_memory_grouped import collate_grouped_native_batch
    from cosmos_framework.trainer.matched_no_ttt import MatchedNoTTTWindow
    from cosmos_framework.trainer.matched_no_ttt_test import Model

    witnesses = []
    for rank in range(world):
        raw, transform, planner, producer = source_setup(rank=rank, world=world, count=count)
        original = producer.catalog.wrapped_sft._transform

        def stochastic(item, resolution):
            out = original(item, resolution)
            out["randoms"] = (random.random(), float(torch.rand(())), float(np.random.rand()))
            out["video"] = torch.zeros(3, 17, 2, 4)
            out["text_token_ids"] = torch.tensor([1, 2, 3])
            out["sequence_plan"] = f"{out['episode_index']}:{out['start_frame']}"
            out["action_mask"] = torch.cat((torch.ones(15), torch.zeros(49)))
            return out

        producer.catalog.wrapped_sft._transform = stochastic
        plan = planner.plan_window(planner.initial_frontier())
        random.seed(51 + rank)
        np.random.seed(51 + rank)
        torch.manual_seed(51 + rank)
        expected = [producer.produce(r) for member in plan.members for r in member]
        expected_rng = (random.getstate(), np.random.get_state(), torch.get_rng_state())
        random.seed(51 + rank)
        np.random.seed(51 + rank)
        torch.manual_seed(51 + rank)
        loader = module.NoTTTCpuLoader(producer, LoaderProfile(workers, 1, True, 60), recipe=SyntheticRecipe())
        try:
            loader.begin(plan)
            actual = [loader.materialize(r) for member in plan.members for r in member]
            record = loader.finish()
            assert len(record["worker_pids"]) == workers and os.getpid() not in record["worker_pids"]
            assert record["loader_profile"]["pin_memory"] is False
            for left, right in zip(expected, actual, strict=True):
                equal_rows(rows(left), rows(right))
                for field in (
                    "consumer_valid",
                    "consumer_step",
                    "evidence_valid",
                    "evidence_source_step",
                    "evidence_executed_action_prev",
                    "consumer_visual_summary",
                ):
                    torch.testing.assert_close(getattr(left, field), getattr(right, field), rtol=0, atol=0)
            assert random.getstate() == expected_rng[0]
            assert np.array_equal(np.random.get_state()[1], expected_rng[1][1])
            assert np.random.get_state()[2:] == expected_rng[1][2:]
            torch.testing.assert_close(torch.get_rng_state(), expected_rng[2], rtol=0, atol=0)
            # Full/partial/native GA weights and generation gradient oracle.
            a = torch.tensor(2.0, requires_grad=True)
            b = torch.tensor(2.0, requires_grad=True)
            windows = [MatchedNoTTTWindow(Model(), planner) for _ in range(2)]
            for window in windows:
                window.begin()

            def equal_tree(left, right):
                if isinstance(left, torch.Tensor):
                    torch.testing.assert_close(left, right, rtol=0, atol=0)
                elif isinstance(left, dict):
                    assert left.keys() == right.keys()
                    for key in left:
                        equal_tree(left[key], right[key])
                elif isinstance(left, (tuple, list)):
                    assert len(left) == len(right)
                    for x, y in zip(left, right, strict=True):
                        equal_tree(x, y)
                else:
                    assert left == right

            for member in range(4):
                lo = expected[member * 8 : member * 8 + 8]
                ro = actual[member * 8 : member * 8 + 8]
                for window, segments in zip(windows, (lo, ro), strict=True):
                    payloads, positions = window.pack(tuple(segments))
                    window.member += 1
                    if window is windows[0]:
                        # randoms is test-only RNG evidence, not a model ABI field.
                        packed = collate_grouped_native_batch(
                            tuple({k: v for k, v in p.items() if k != "randoms"} for p in payloads)
                        )
                    else:
                        equal_tree(
                            packed,
                            collate_grouped_native_batch(
                                tuple({k: v for k, v in p.items() if k != "randoms"} for p in payloads)
                            ),
                        )
                lhs = torch.stack([p[3][1].mean() for s in lo for p in rows(s)]).mean()
                rhs = torch.stack([p[3][1].mean() for s in ro for p in rows(s)]).mean()
                weight = plan.member_counts[member] / plan.n_window
                (a * lhs * weight).backward()
                (b * rhs * weight).backward()
            torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
            witnesses.append(
                {
                    "rank": rank,
                    "world": world,
                    "workers": workers,
                    "trainer_pid": os.getpid(),
                    "worker_pids": record["worker_pids"],
                    "counts": plan.member_counts,
                }
            )
        finally:
            assert_closed(loader)
    print("CPU_MULTIPROCESS_WITNESS " + json.dumps(witnesses), flush=True)


@pytest.mark.parametrize("persistent", [False, True])
def test_lifecycle_two_windows_reuses_only_if_requested_and_preserves_parent_rng(persistent):
    _, _, planner, producer = source_setup(count=80, t=2, b=8)
    loader = module.NoTTTCpuLoader(producer, LoaderProfile(2, 2, persistent, 60), recipe=SyntheticRecipe())
    before = torch.get_rng_state().clone()
    frontier = planner.initial_frontier()
    pids = []
    try:
        for _ in range(2):
            plan = planner.plan_window(frontier)
            loader.begin(plan)
            for member in plan.members:
                for request in member:
                    loader.materialize(request)
            pids.append(loader.finish()["worker_pids"])
            frontier = plan.candidate_frontier
        assert (pids[0] == pids[1]) is persistent
        torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)
        assert loader.profile.workers * loader.profile.prefetch_factor == 4
    finally:
        assert_closed(loader)


@pytest.mark.parametrize("fault", ["error", "crash", "timeout", "ignore_term", "order", "abi", "source"])
def test_faults_abort_owned_workers_boundedly_without_foreign_kill(fault, tmp_path):
    _, _, planner, producer = source_setup(t=2)
    timeout = 1.0 if fault in ("timeout", "ignore_term") else 60
    loader = module.NoTTTCpuLoader(
        producer, LoaderProfile(2, 1, True, timeout), recipe=SyntheticRecipe(fault, str(tmp_path))
    )
    foreign = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    started = time.monotonic()
    try:
        plan = planner.plan_window(planner.initial_frontier())
        with pytest.raises((RuntimeError, OSError, ValueError)):
            loader.begin(plan)
            if fault in ("timeout", "ignore_term"):
                until = time.monotonic() + 30
                while len(list(tmp_path.glob("ready_*"))) < 2 and time.monotonic() < until:
                    time.sleep(0.05)
                assert len(list(tmp_path.glob("ready_*"))) == 2, "fault must actually be injected before timeout"
            for member in plan.members:
                for request in member:
                    loader.materialize(request)
        assert loader.closed and loader.pending is None
        assert foreign.poll() is None
        assert time.monotonic() - started < 70
        assert_closed(loader)
        assert all(row["exitcode"] is not None for row in loader.last_cleanup)
        if fault == "ignore_term":
            assert any(row["exitcode"] == -9 for row in loader.last_cleanup)
    finally:
        loader.close()
        foreign.terminate()
        foreign.wait(timeout=5)


def test_wrong_request_order_and_partial_spawn_failure_are_closed(monkeypatch):
    _, _, planner, producer = source_setup(t=2)
    loader = module.NoTTTCpuLoader(producer, LoaderProfile(2, 1, True, 60), recipe=SyntheticRecipe())
    plan = planner.plan_window(planner.initial_frontier())
    loader.begin(plan)
    with pytest.raises(ValueError, match="order"):
        loader.materialize(plan.members[0][1])
    assert_closed(loader)
    loader = module.NoTTTCpuLoader(producer, LoaderProfile(2, 1, True, 60), recipe=SyntheticRecipe())
    original = loader.context.Process
    calls = []

    def process(*args, **kwargs):
        p = original(*args, **kwargs)
        calls.append(p)
        if len(calls) == 2:

            def fail():
                raise OSError("injected second worker spawn failure")

            monkeypatch.setattr(p, "start", fail)
        return p

    monkeypatch.setattr(loader.context, "Process", process)
    with pytest.raises(OSError, match="spawn failure"):
        loader.begin(plan)
    assert_closed(loader)


@pytest.mark.parametrize("workers", [-1, 3, 5, 16, True, "2"])
def test_bad_worker_count_rejected(workers):
    with pytest.raises(ValueError):
        LoaderProfile(workers)


@pytest.mark.parametrize("kwargs", [{"prefetch_factor": 1}, {"persistent_workers": True}, {"timeout_s": 1}])
def test_zero_worker_options_fail(kwargs):
    with pytest.raises(ValueError, match="workers=0"):
        LoaderProfile(**kwargs)


@pytest.mark.parametrize("workers", [2, 8])
@pytest.mark.parametrize(
    "change", [{"no_ttt": False}, {"world_size": 8}, {"ga": 2}, {"job_name": "no_ttt_wrong_world8_ga2"}]
)
def test_cli_worker_profile_rejects_old_job_and_other_routes(change, workers):
    args = SimpleNamespace(
        no_ttt=True,
        world_size=4,
        ga=4,
        b=8,
        t=16,
        job_name=f"no_ttt_new_mpw{workers}_world4_ga4",
        no_ttt_loader_workers=workers,
    )
    args.__dict__.update(change)
    with pytest.raises(ValueError, match="NEW"):
        profile_from_args(args)


@pytest.mark.parametrize("workers", [1, 2, 4, 8])
def test_optin_workers_are_allowed_on_original_world4_nott_job(workers):
    args = SimpleNamespace(
        no_ttt=True,
        world_size=4,
        ga=4,
        b=8,
        t=16,
        job_name="no_ttt_packed128_droid_fresh_30k_world4_ga4",
        no_ttt_loader_workers=workers,
    )
    assert profile_from_args(args).workers == workers


def test_real_synthetic_cache_factory_spawn_mmap_and_native_raw_values(tmp_path):
    from cosmos_framework.data.generator.action.datasets.action_sft_dataset import ActionSFTDataset
    from cosmos_framework.data.generator.action.datasets.robocasa_exact_window_cached_sft_test import _raw, roots
    from cosmos_framework.model.generator.mot.robocasa_exact_window_local import ExactWindowLocalCatalog
    from cosmos_framework.model.generator.mot.robocasa_exact_window_local_test import _request, _Transform

    raw = _raw(roots.__wrapped__(tmp_path))
    catalog = ExactWindowLocalCatalog(ActionSFTDataset(raw, _Transform(), None), ttt_tbptt_steps=16)
    producer = ExactWindowSegmentProducer(catalog, config_digest="config")
    request = _request(catalog)
    expected = producer.produce(request)
    plan = SimpleNamespace(members=((request,),))
    loader = module.NoTTTCpuLoader(producer, LoaderProfile(1, 1, True, 60))
    try:
        loader.begin(plan)
        actual = loader.materialize(request)
        equal_rows(rows(expected), rows(actual))
        for a, b in zip(expected.consumer_payload[0], actual.consumer_payload[0], strict=True):
            if a is not None:
                assert a.keys() == b.keys()
                for key, value in a.items():
                    if isinstance(value, torch.Tensor):
                        torch.testing.assert_close(value, b[key], rtol=0, atol=0)
                    else:
                        assert value == b[key]
        assert loader.finish()["worker_pids"] != [os.getpid()]
    finally:
        assert_closed(loader)


def test_mmap_reader_preserves_original_window_rejection_and_bound(tmp_path):
    from cosmos_framework.data.generator.action.datasets.robocasa_exact_window_policy_test import _catalog

    catalog = _catalog(tmp_path)
    record = catalog.episodes[0]
    reader = module.MappedEpisodeReader(catalog, 1, 2 * 1024**3)
    value = reader.read_window(record.key, 0)
    assert value.shape == (5, 48, 2, 3) and len(reader._loaded) == 1
    bad = module.MappedEpisodeReader(catalog, 1, 1024)
    with pytest.raises(ValueError, match="bound"):
        bad.read_window(record.key, 0)
    path = catalog.cache_root / record.relative_path
    payload = torch.load(path, weights_only=True)
    payload["metadata"]["camera_set"] = "foreign"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="camera"):
        module.MappedEpisodeReader(catalog, 1, 2 * 1024**3).read_window(record.key, 0)


def test_source_rng_is_fail_closed_and_cancel_discards_all_inflight():
    _, _, planner, producer = source_setup(t=2)
    loader = module.NoTTTCpuLoader(producer, LoaderProfile(1, 1, True, 60), recipe=SyntheticRecipe())
    original = producer.catalog.raw.source_reader.read_at

    def bad_source(index):
        random.random()
        return original(index)

    producer.catalog.raw.source_reader.read_at = bad_source
    plan = planner.plan_window(planner.initial_frontier())
    with pytest.raises(ValueError, match="RNG"):
        loader.begin(plan)
    assert loader.closed and loader.context.processes == []
    producer.catalog.raw.source_reader.read_at = original
    loader = module.NoTTTCpuLoader(producer, LoaderProfile(1, 1, True, 60), recipe=SyntheticRecipe())
    loader.begin(plan)
    first = loader.materialize(plan.members[0][0])
    loader.close()
    assert loader.pending is None
    restarted = module.NoTTTCpuLoader(producer, LoaderProfile(1, 1, True, 60), recipe=SyntheticRecipe())
    try:
        restarted.begin(plan)
        equal_rows(rows(first), rows(restarted.materialize(plan.members[0][0])))
    finally:
        assert_closed(restarted)


def test_encoded_packet_memory_limit_cleans_workers():
    _, _, planner, producer = source_setup(t=2)
    loader = module.NoTTTCpuLoader(
        producer, LoaderProfile(1, 1, True, 60, max_packet_bytes=65536), recipe=SyntheticRecipe("large_packet")
    )
    try:
        with pytest.raises((ValueError, RuntimeError), match="bound"):
            plan = planner.plan_window(planner.initial_frontier())
            loader.begin(plan)
            loader.materialize(plan.members[0][0])
    finally:
        assert_closed(loader)
    assert loader.last_cleanup, "size fault must be injected in an actual Worker"


def test_spawn_is_cpu_before_import_and_restores_parent_gpu_rank_environment(monkeypatch):
    _, _, planner, producer = source_setup(t=2)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5,6,7")
    monkeypatch.setenv("RANK", "2")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "192")
    loader = module.NoTTTCpuLoader(producer, LoaderProfile(1, 1, True, 60), recipe=SyntheticRecipe())
    try:
        plan = planner.plan_window(planner.initial_frontier())
        loader.begin(plan)
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "4,5,6,7"
        assert os.environ["RANK"] == "2" and os.environ["OPENBLAS_NUM_THREADS"] == "192"
        loader.materialize(plan.members[0][0])
    finally:
        assert_closed(loader)
