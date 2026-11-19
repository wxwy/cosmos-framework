"""G2 exact10 benchmark boundary is separate from permanent training semantics."""

from types import SimpleNamespace

import pytest

from examples import psm_wma_robocasa_corrected_phase5 as phase5


def benchmark_args(tmp_path, *, kind="r1", phase="fresh", count=10):
    return SimpleNamespace(
        phase=phase,
        g2_recipe=tmp_path / "fixture-recipe.json",
        no_ttt=False,
        job_name=kind + "_g2_bench_solo_20261119",
        output_root=tmp_path,
        stop_after_iter=None,
        formal_execution_until=None,
        g2_benchmark_steps=count,
        max_iter=30000,
        warmup=0,
        save_iter=100,
        audit_missing_grads=False,
    )


@pytest.mark.parametrize("steps", [2, 10])
def test_benchmark_step_cap_preserves_native_training_horizon(tmp_path, monkeypatch, steps):
    monkeypatch.setattr(phase5, "_g2_recipe", lambda a: SimpleNamespace(training=SimpleNamespace(profile="R1")))
    args = benchmark_args(tmp_path, count=steps)
    assert phase5._execution_stop_iteration(args) == steps
    assert (args.max_iter, args.warmup, args.save_iter) == (30000, 0, 100)


@pytest.mark.parametrize(
    "change",
    [
        dict(phase="resume"),
        dict(g2_benchmark_steps=11),
        dict(job_name="r2_g2_bench_solo_20261119"),
        dict(job_name="r1_g2_formal_async8_30000_20261119"),
        dict(formal_execution_until=10),
        dict(stop_after_iter=10),
        dict(warmup=500),
        dict(save_iter=1),
        dict(audit_missing_grads=True),
    ],
)
def test_benchmark_denies_unsafe_overrides(tmp_path, monkeypatch, change):
    monkeypatch.setattr(phase5, "_g2_recipe", lambda a: SimpleNamespace(training=SimpleNamespace(profile="R1")))
    args = benchmark_args(tmp_path)
    for k, v in change.items():
        setattr(args, k, v)
    with pytest.raises(ValueError, match="benchmark requires"):
        phase5._execution_stop_iteration(args)


def test_benchmark_refuses_existing_namespace(tmp_path, monkeypatch):
    monkeypatch.setattr(phase5, "_g2_recipe", lambda a: SimpleNamespace(training=SimpleNamespace(profile="R1")))
    args = benchmark_args(tmp_path)
    path = phase5.job_namespace(args)
    path.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="new job"):
        phase5._execution_stop_iteration(args)


def test_g2_spawn_profile_preserves_runtime_defaults_and_conflicts(tmp_path):
    args = benchmark_args(tmp_path)
    args.world_size, args.ga, args.b, args.t = 4, 4, 8, 16
    assert phase5._g2_runtime_loader(args, None) is None
    args.g2_spawn_workers = 2
    profile = phase5._g2_runtime_loader(args, object())
    assert (profile.workers, profile.prefetch_factor, profile.persistent_workers, profile.timeout_s) == (
        2,
        1,
        True,
        120,
    )
    args.num_workers = 1
    with pytest.raises(ValueError, match="legacy_global and no Raw threads"):
        phase5._g2_runtime_loader(args, object())
    args.num_workers = 0
    args.transform_rng_mode = "consumer_keyed_v1"
    with pytest.raises(ValueError, match="legacy_global"):
        phase5._g2_runtime_loader(args, object())


def test_g2_trace_is_bounded(tmp_path):
    args = benchmark_args(tmp_path)
    args.g2_profile_trace = True
    args.g2_benchmark_steps = None
    with pytest.raises(ValueError, match="only in a bounded benchmark"):
        phase5._g2_runtime_loader(args, object())
