"""Bounded, process-local fixed-weight condition caches owned by existing Sessions.

Keys describe *content dependencies*, never tensor addresses or decision frames.
Request leases pin values until sampling finishes; eviction cannot release a live
request's accounting. This module neither schedules requests nor owns Local state.
"""

from __future__ import annotations

import hashlib
import json
import sys
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, fields, is_dataclass, replace
from typing import Callable

import torch

STAGES = ("view_latent", "text", "text_kv", "task_queries", "global", "global_projection", "global_kv")


def content_key(*dependencies) -> str:
    """Hash explicit JSON dependencies; tensor identity is intentionally unsupported."""
    raw = json.dumps(dependencies, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CacheBudget:
    # Candidate safety bounds, NOT measured deployment recommendations.
    max_sessions: int = 32
    max_entries: int = 256
    cpu_bytes: int = 256 * 1024**2
    gpu_bytes: int = 256 * 1024**2
    handles: int = 4096

    def __post_init__(self):
        if any(type(v) is not int or v <= 0 for v in vars(self).values()):
            raise ValueError("positive explicit condition-cache budgets required")


@dataclass(frozen=True)
class Usage:
    cpu_bytes: int = 0
    gpu_bytes: int = 0
    handles: int = 0

    def __add__(self, other):
        return Usage(*(a + b for a, b in zip(vars(self).values(), vars(other).values(), strict=True)))

    def __sub__(self, other):
        return Usage(*(a - b for a, b in zip(vars(self).values(), vars(other).values(), strict=True)))


def _owned_tree(value, *, clone=True):
    """Own immutable containers and detached tensor storage; reject trainable graphs/FDs."""
    if isinstance(value, torch.Tensor):
        if value.requires_grad or value.device.type not in ("cpu", "cuda"):
            raise ValueError("cache accepts only fixed-weight CPU/CUDA tensors without autograd graphs")
        result = value.detach().clone(memory_format=torch.contiguous_format) if clone else value
        size = result.numel() * result.element_size()
        return result, Usage(size if result.device.type == "cpu" else 0, size if result.is_cuda else 0, 1)
    if value is None or type(value) in (str, bytes, bool, int, float):
        return value, Usage(cpu_bytes=sys.getsizeof(value))
    if isinstance(value, (tuple, list)):
        owned = [_owned_tree(item, clone=clone) for item in value]
        result = tuple(item for item, _ in owned)
        usage = Usage(cpu_bytes=sys.getsizeof(result))
        for _, size in owned:
            usage += size
        return result, usage
    if is_dataclass(value) and not isinstance(value, type) and value.__dataclass_params__.frozen:
        owned = {field.name: _owned_tree(getattr(value, field.name), clone=clone) for field in fields(value)}
        result = replace(value, **{key: item for key, (item, _) in owned.items()}) if clone else value
        usage = Usage(cpu_bytes=sys.getsizeof(result))
        for _, size in owned.values():
            usage += size
        return result, usage
    raise TypeError("condition cache values must be tensor/primitive trees or frozen public dataclasses")


@dataclass(eq=False)
class _Entry:
    key: tuple
    value: object
    usage: Usage
    pins: int = 0


class CacheLease:
    """Single sampling request. Values are read-only to the model hook consuming them."""

    def __init__(self, owner, session_id, enabled):
        self.owner, self.session_id, self.enabled = owner, session_id, enabled
        self._entries: dict[tuple, _Entry] = {}
        self.events: list[dict] = []
        self.closed = False

    def get_or_compute(self, stage: str, key: str, build: Callable[[], object]):
        started = time.perf_counter_ns()
        if self.closed or stage not in STAGES or not isinstance(key, str) or len(key) != 64:
            raise ValueError("live lease, declared stage and content SHA256 required")
        full_key = (self.session_id, self.owner.revision, stage, key)
        with self.owner._lock:
            if full_key in self._entries:
                return self._entries[full_key].value
            entry = self.owner._entries.get(full_key) if self.enabled else None
            hit = entry is not None
            if entry is None:
                # Existing model-owner serialization is retained. Cache lock also
                # prevents duplicate fills if used by another caller in this process.
                computed = build()
                _, expected = _owned_tree(computed, clone=False)
                # Evict/reserve BEFORE allocating owned tensor copies. The
                # model's temporary working output is outside the cache budget.
                self.owner._make_room(expected, extra_entry=True)
                value, usage = _owned_tree(computed)
                if usage != expected:
                    raise RuntimeError("condition metadata changed while capturing fixed request storage")
                entry = _Entry(full_key, value, usage)
                self.owner._usage += usage
                if self.enabled:
                    self.owner._entries[full_key] = entry
                else:
                    self.owner._transient.add(entry)
            else:
                self.owner._entries.move_to_end(full_key)
            entry.pins += 1
            self._entries[full_key] = entry
            event = {"stage": stage, "hit": hit, "elapsed_ns": time.perf_counter_ns() - started}
            self.events.append(event)
            stats = self.owner._stats[stage]
            bucket = "hit" if hit else "miss"
            stats[bucket] += 1
            stats[bucket + "_ns"] += event["elapsed_ns"]
            return entry.value

    def close(self):
        with self.owner._lock:
            if self.closed:
                return
            for entry in self._entries.values():
                entry.pins -= 1
                if entry in self.owner._transient:
                    self.owner._transient.remove(entry)
                    self.owner._usage -= entry.usage
            self._entries.clear()
            self.closed = True
            self.owner._active[self.session_id] -= 1


class SessionConditionCache:
    def __init__(self, revision: str, budget: CacheBudget = CacheBudget()):
        if not isinstance(revision, str) or not revision or len(revision) > 512:
            raise ValueError("explicit fixed-weight/config revision required")
        self.revision, self.budget = revision, budget
        self._lock = threading.RLock()
        self._entries: OrderedDict[tuple, _Entry] = OrderedDict()
        self._sessions: OrderedDict[str, None] = OrderedDict()
        self._active: dict[str, int] = {}
        self._transient: set[_Entry] = set()
        self._usage = Usage()
        self._stats = {s: {"hit": 0, "miss": 0, "hit_ns": 0, "miss_ns": 0, "evictions": 0} for s in STAGES}

    def _drop(self, key):
        entry = self._entries.pop(key)
        self._usage -= entry.usage
        self._stats[key[2]]["evictions"] += 1

    def _make_room(self, usage, *, extra_entry):
        if (
            usage.cpu_bytes > self.budget.cpu_bytes
            or usage.gpu_bytes > self.budget.gpu_bytes
            or usage.handles > self.budget.handles
        ):
            raise MemoryError("single condition exceeds explicit CPU/GPU/handle budget")

        def fits():
            total = self._usage + usage
            return (
                len(self._entries) + len(self._transient) + int(extra_entry) <= self.budget.max_entries
                and total.cpu_bytes <= self.budget.cpu_bytes
                and total.gpu_bytes <= self.budget.gpu_bytes
                and total.handles <= self.budget.handles
            )

        while not fits():
            victim = next((key for key, entry in self._entries.items() if not entry.pins), None)
            if victim is None:
                raise MemoryError("condition cache/request exceeds explicit CPU/GPU/entry/handle budget")
            self._drop(victim)

    @contextmanager
    def request(self, session_id: str, *, enabled=True):
        if not isinstance(session_id, str) or not session_id or len(session_id) > 256 or type(enabled) is not bool:
            raise ValueError("bounded Session identity and explicit cache switch required")
        with self._lock:
            if session_id not in self._sessions:
                if len(self._sessions) >= self.budget.max_sessions:
                    victim = next((s for s in self._sessions if not self._active[s]), None)
                    if victim is None:
                        raise MemoryError("all bounded cache Sessions have active requests")
                    self.reset(victim)
                self._sessions[session_id] = None
                self._active[session_id] = 0
            self._sessions.move_to_end(session_id)
            self._active[session_id] += 1
        lease = CacheLease(self, session_id, enabled)
        try:
            yield lease
        finally:
            lease.close()

    def reset(self, session_id):
        """Only called after the existing transaction owner proves quiescence."""
        with self._lock:
            if self._active.get(session_id, 0):
                raise RuntimeError("cannot clear conditions during active sampling")
            for key in tuple(self._entries):
                if key[0] == session_id:
                    self._drop(key)
            self._sessions.pop(session_id, None)
            self._active.pop(session_id, None)

    def reload(self, revision):
        with self._lock:
            if any(self._active.values()):
                raise RuntimeError("model reload requires existing owner quiescence")
            if not isinstance(revision, str) or not revision or len(revision) > 512:
                raise ValueError("explicit fixed-weight/config revision required")
            for session in tuple(self._sessions):
                self.reset(session)
            self.revision = revision

    def info(self):
        with self._lock:
            return {
                "revision": self.revision,
                "budget": vars(self.budget).copy(),
                "usage": vars(self._usage).copy(),
                "sessions": len(self._sessions),
                "entries": len(self._entries),
                "request_entries": len(self._transient),
                "active_requests": sum(self._active.values()),
                "stages": {key: value.copy() for key, value in self._stats.items()},
                "counter_scope": "process lifetime; model reload empties all resident derived values",
                "timing": "host elapsed; hit/miss separated; no CUDA speed inference",
            }
