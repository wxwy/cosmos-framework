"""Device timing API mechanics; mocked CUDA events are not hardware evidence."""

from contextlib import nullcontext

import pytest

from cosmos_framework.inference import local_memory_pipeline_timing as timing
from cosmos_framework.inference.local_memory_pipeline_metrics import DurationWindow


def test_cpu_records_separate_stages_without_sleep(monkeypatch):
    ticks = iter((10.0, 10.025, 20.0, 20.005, 30.0, 30.1))
    monkeypatch.setattr(timing.time, "perf_counter", lambda: next(ticks))
    clock = timing.PipelineTiming("cpu")
    for name in ("independent_encode1", "visual96", "fast_update"):
        with clock.measure(name):
            pass
    result = clock.finish()
    assert result["timing_source"] == "cpu_perf_counter"
    assert result["stages"]["independent_encode1"] == {"count": 1, "total_ms": pytest.approx(25)}
    assert result["stages"]["visual96"] == {"count": 1, "total_ms": pytest.approx(5)}
    assert result["stages"]["fast_update"] == {"count": 1, "total_ms": pytest.approx(100)}


def test_cuda_event_elapsed_is_read_only_after_completion(monkeypatch):
    events, actions = [], []

    class Event:
        def __init__(self, *, enable_timing):
            assert enable_timing is True
            self.index = len(events)
            events.append(self)

        def record(self):
            actions.append(("record", self.index))

        def synchronize(self):
            actions.append(("synchronize", self.index))

        def elapsed_time(self, end):
            assert ("synchronize", len(events) - 1) in actions
            actions.append(("elapsed", self.index))
            assert end.index == self.index + 1
            return 1.25

    monkeypatch.setattr(timing.torch.cuda, "Event", Event)
    monkeypatch.setattr(timing.torch.cuda, "device", lambda device: nullcontext())
    clock = timing.PipelineTiming("cuda:6")
    for name in ("independent_encode1", "visual96", "fast_update"):
        with clock.measure(name):
            pass
    assert not any(action[0] == "elapsed" for action in actions)
    result = clock.finish()
    assert result["timing_source"] == "cuda_events"
    assert all(stage == {"count": 1, "total_ms": 1.25} for stage in result["stages"].values())
    assert sum(action[0] == "synchronize" for action in actions) == 1


def test_scopes_reject_nesting_unknown_stages_and_reuse():
    clock = timing.PipelineTiming("cpu")
    with pytest.raises(ValueError, match="unknown"):
        with clock.measure("future_frames"):
            pass
    with clock.measure("fast_update"):
        with pytest.raises(RuntimeError, match="non-nested"):
            with clock.measure("visual96"):
                pass
    clock.finish()
    with pytest.raises(RuntimeError):
        clock.finish()
    with pytest.raises(RuntimeError):
        with clock.measure("fast_update"):
            pass


def test_stage_does_not_swallow_work_exception():
    clock = timing.PipelineTiming("cpu")
    with pytest.raises(ArithmeticError, match="primary"):
        with clock.measure("fast_update"):
            raise ArithmeticError("primary")
    assert clock.finish()["stages"]["fast_update"]["count"] == 1


def test_duration_window_is_bounded_but_keeps_total_and_discloses_recent_percentiles():
    window = DurationWindow(2)
    assert window.summary()["p95_ms"] is None
    for value in (100, 1, 3, 9):
        window.add(value)
    summary = window.summary()
    assert len(window.values) == summary["recent_count"] == 2
    assert summary["count"] == 4 and summary["total_ms"] == 113
    assert summary["p50_ms"] == 3 and summary["p95_ms"] == 9
    assert summary["percentile_scope"] == "recent_bounded_nearest_rank"
    with pytest.raises(ValueError):
        window.add(float("nan"))
