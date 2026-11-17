from dataclasses import dataclass

import pytest
import torch

from cosmos_framework.inference.g2_session_cache import CacheBudget, SessionConditionCache, content_key


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


@pytest.mark.parametrize("field", ["cpu_bytes", "handles"])
def test_bytes_and_handle_limits_reject_oversized_single_entry(field):
    cache = SessionConditionCache("w", CacheBudget(**{field: 1}))
    with pytest.raises(MemoryError):
        with cache.request("s") as request:
            request.get_or_compute("text", content_key("x"), lambda: (torch.ones(2), torch.ones(2)))
    assert cache.info()["entries"] == 0 and cache.info()["usage"]["handles"] == 0


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
