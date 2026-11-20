import json
from dataclasses import dataclass

import pytest
import torch

from cosmos_framework.inference import g2_session_cache
from cosmos_framework.inference.g2_session_cache import CacheBudget, SessionConditionCache, Usage, content_key


@dataclass(frozen=True)
class Value:
    hidden: torch.Tensor
    kv: tuple
    identity: str


def test_same_content_new_tensor_and_new_frame_reuses_owned_storage():
    cache = SessionConditionCache("weights+config")
    source = torch.ones(2)
    calls = []

    def build():
        calls.append(1)
        return Value(source, (source + 1,), "full")

    # Frame/source provenance belongs to the request snapshot, not this key.
    key = content_key({"tokens": [1, 2], "position": [0, 1], "recipe": "v1"})
    with cache.request("session") as request:
        first = request.get_or_compute("text", key, build)
        source.add_(9)
        assert torch.equal(first.hidden, torch.ones(2))
    with cache.request("session") as request:
        second = request.get_or_compute("text", key, build)
        assert second is first and request.events[0]["hit"]
    assert len(calls) == 1
    info = cache.info()
    assert info["stages"]["text"]["hit"] == info["stages"]["text"]["miss"] == 1
    assert info["usage"]["handles"] == 2


def test_no_cross_session_reuse_and_reset_and_reload():
    cache = SessionConditionCache("w1")
    key, calls = content_key("same text"), []

    def build():
        calls.append(1)
        return torch.tensor(len(calls))

    for session in ("a", "b", "a"):
        with cache.request(session) as request:
            request.get_or_compute("global", key, build)
    assert len(calls) == 2
    cache.reset("a")
    with cache.request("a") as request:
        request.get_or_compute("global", key, build)
    cache.reload("w2")
    assert cache.info()["entries"] == 0 and cache.info()["usage"]["handles"] == 0
    with cache.request("a") as request:
        request.get_or_compute("global", key, build)
    assert len(calls) == 4


def test_cache_off_is_a_real_recompute_path_with_bounded_transient_lifetime():
    cache, key, calls = SessionConditionCache("w"), content_key("x"), []
    for _ in range(2):
        with cache.request("s", enabled=False) as request:
            request.get_or_compute("global_kv", key, lambda: calls.append(1) or torch.ones(2))
            info = cache.info()
            assert info["entries"] == 0 and info["request_entries"] == 1
            assert info["usage"]["handles"] == 1
        assert cache.info()["usage"]["handles"] == 0
    assert len(calls) == 2


def test_capacity_lru_and_pinning_and_cleanup_after_exception():
    cache = SessionConditionCache("w", CacheBudget(max_entries=1))
    with pytest.raises(MemoryError, match="budget"):
        with cache.request("s") as request:
            request.get_or_compute("text", content_key("1"), lambda: torch.ones(1))
            request.get_or_compute("text", content_key("2"), lambda: torch.ones(1))
    assert cache.info()["active_requests"] == 0
    with cache.request("s") as request:
        request.get_or_compute("text", content_key("2"), lambda: torch.ones(1))
    assert cache.info()["entries"] == 1
    assert cache.info()["stages"]["text"]["evictions"] == 1


@pytest.mark.parametrize("dimension", ["cpu_bytes", "gpu_bytes", "entries", "handles"])
def test_pinned_pressure_reports_each_actual_blocking_dimension(monkeypatch, dimension):
    budget_field = "max_entries" if dimension == "entries" else dimension
    cache = SessionConditionCache("w", CacheBudget(**{budget_field: 1}))
    # Inject accounting records to exercise GPU-byte pressure on a CPU host.
    size = Usage(cpu_bytes=1, gpu_bytes=1, handles=1)
    monkeypatch.setattr(g2_session_cache, "_owned_tree", lambda value, **_: (value, size))
    with cache.request("first") as first, cache.request("second") as second:
        first.get_or_compute("text", content_key("first"), lambda: "one")
        with pytest.raises(MemoryError, match="pinned entries") as exc:
            second.get_or_compute("text_kv", content_key("second"), lambda: "two")
        info = cache.info()
        record = info["rejections"]["last"]
        assert json.loads(str(exc.value).split(": ", 1)[1]) == record
        assert record["stage"] == "text_kv"
        assert record["exceeded"] == [dimension]
        assert record["current"][dimension] == record["requested"][dimension] == record["pinned"][dimension] == 1
        assert record["limits"] == {dimension: 1}
        assert info["high_water"]["active_leases"] == record["current"]["active_leases"] == 2
        assert info["rejections"]["by_dimension"][dimension] == 1
        assert info["entries"] == 1
    assert cache.info()["pinned"]["entries"] == 0
    assert cache.info()["high_water"]["active_leases"] == 2


def test_pinned_rejection_keeps_eligible_cold_entries_and_counts_transient(monkeypatch):
    cache = SessionConditionCache("w", CacheBudget(gpu_bytes=8))
    monkeypatch.setattr(g2_session_cache, "_owned_tree", lambda value, **_: (value, Usage(gpu_bytes=value, handles=1)))
    cold_key = content_key("cold")
    with cache.request("cold") as cold:
        cold.get_or_compute("text", cold_key, lambda: 2)
    with cache.request("active", enabled=False) as active:
        active.get_or_compute("text", content_key("warm"), lambda: 6)
        with pytest.raises(MemoryError, match="pinned entries"):
            active.get_or_compute("text_kv", content_key("miss"), lambda: 4)
        info = cache.info()
        assert info["usage"]["gpu_bytes"] == info["high_water"]["gpu_bytes"] == 8
        assert info["rejections"]["last"]["pinned"]["gpu_bytes"] == 6
        assert info["entries"] == info["request_entries"] == 1
        assert info["stages"]["text"]["evictions"] == 0
    with cache.request("cold") as cold:
        assert cold.get_or_compute("text", cold_key, lambda: 3) == 2
    cache.reset("cold")
    assert cache.info()["usage"]["gpu_bytes"] == 0
    assert cache.info()["high_water"]["gpu_bytes"] == 8


@pytest.mark.parametrize("field", ["cpu_bytes", "handles"])
def test_bytes_and_handle_limits_reject_oversized_single_entry(field):
    cache = SessionConditionCache("w", CacheBudget(**{field: 1}))
    with pytest.raises(MemoryError):
        with cache.request("s") as request:
            request.get_or_compute("text", content_key("x"), lambda: (torch.ones(2), torch.ones(2)))
    assert cache.info()["entries"] == 0 and cache.info()["usage"]["handles"] == 0
    assert cache.info()["rejections"]["last"]["exceeded"] == [field]


def test_active_request_cannot_reset_reload_or_be_evicted():
    cache = SessionConditionCache("w", CacheBudget(max_sessions=1))
    with cache.request("s"):
        with pytest.raises(RuntimeError, match="active"):
            cache.reset("s")
        with pytest.raises(RuntimeError, match="quiescence"):
            cache.reload("w2")
        with pytest.raises(MemoryError, match="Sessions"):
            with cache.request("other"):
                pass
    with cache.request("other"):
        assert cache.info()["sessions"] == 1
    assert cache.info()["rejections"]["last"]["exceeded"] == ["sessions"]


def test_GPU_accounting_limit_without_allocating_a_GPU_or_evicting_valid_content():
    cache = SessionConditionCache("w", CacheBudget(gpu_bytes=32))
    with cache.request("s") as request:
        request.get_or_compute("text", content_key("ok"), lambda: torch.ones(1))
    with pytest.raises(MemoryError, match="single condition"):
        cache._make_room(Usage(gpu_bytes=33), extra_entry=True)
    assert cache.info()["entries"] == 1 and cache.info()["usage"]["gpu_bytes"] == 0


def test_capacity_is_reserved_before_owned_tensor_copies(monkeypatch):
    cache = SessionConditionCache("w", CacheBudget(handles=1))
    value = (torch.ones(2), torch.ones(2))
    copied = []
    original = torch.Tensor.clone

    def clone(tensor, *args, **kwargs):
        copied.append(1)
        return original(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "clone", clone)
    with cache.request("s") as request, pytest.raises(MemoryError, match="single condition"):
        request.get_or_compute("text", content_key("too-big"), lambda: value)
    assert not copied


def test_failed_factory_and_trainable_outputs_are_not_published():
    cache = SessionConditionCache("w")
    with cache.request("s") as request:
        with pytest.raises(ValueError, match="autograd"):
            request.get_or_compute("global", content_key("x"), lambda: torch.ones(2, requires_grad=True))
        with pytest.raises(ZeroDivisionError):
            request.get_or_compute("global", content_key("x"), lambda: 1 / 0)
        assert cache.info()["entries"] == 0
        assert torch.is_grad_enabled()  # The Local inner caller's mode is untouched.
    with pytest.raises(ValueError, match="live lease"):
        request.get_or_compute("global", content_key("x"), lambda: torch.ones(2))


def test_bad_value_handles_and_non_content_keys_rejected():
    cache = SessionConditionCache("w")
    with pytest.raises(TypeError):
        content_key(torch.ones(1))
    with cache.request("s") as request:
        with pytest.raises(TypeError, match="cache values"):
            request.get_or_compute("text", content_key("x"), object)
        with pytest.raises(ValueError, match="content SHA"):
            request.get_or_compute("text", "episode-id", lambda: None)


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_explicit_budget_validation(value):
    with pytest.raises(ValueError, match="budgets"):
        CacheBudget(max_sessions=value)
