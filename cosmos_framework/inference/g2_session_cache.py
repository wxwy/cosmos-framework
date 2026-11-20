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
from copy import deepcopy
from dataclasses import dataclass, fields, is_dataclass, replace
from itertools import chain
from typing import Callable

import torch

from cosmos_framework.inference.g2_cache_budget import CacheBudget as CacheBudget

STAGES = ("view_latent", "text", "text_kv", "task_queries", "global", "global_projection", "global_kv")


def content_key(*dependencies) -> str:
    """Hash explicit JSON dependencies; tensor identity is intentionally unsupported."""
    raw = json.dumps(dependencies, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


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
                self.owner._make_room(expected, extra_entry=True, stage=stage)
                value, usage = _owned_tree(computed)
                if usage != expected:
                    raise RuntimeError("condition metadata changed while capturing fixed request storage")
                entry = _Entry(full_key, value, usage)
                self.owner._usage += usage
                if self.enabled:
                    self.owner._entries[full_key] = entry
                else:
                    self.owner._transient.add(entry)
                self.owner._track_high_water(self.owner._resident())
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
        self._limits = dict(
            cpu_bytes=budget.cpu_bytes,
            gpu_bytes=budget.gpu_bytes,
            handles=budget.handles,
            entries=budget.max_entries,
            sessions=budget.max_sessions,
        )
        self._high_water = dict.fromkeys((*self._limits, "active_leases"), 0)
        self._reject_count = 0
        self._reject_by_dimension = dict.fromkeys(self._limits, 0)
        self._last_rejection = None
        self._stats = {s: {"hit": 0, "miss": 0, "hit_ns": 0, "miss_ns": 0, "evictions": 0} for s in STAGES}

    def _drop(self, key):
        entry = self._entries.pop(key)
        self._usage -= entry.usage
        self._stats[key[2]]["evictions"] += 1

    def _resident(self):
        return {"entries": len(self._entries) + len(self._transient), **vars(self._usage)}

    def _pinned(self):
        usage, entries = Usage(), 0
        for entry in chain(self._entries.values(), self._transient):
            if entry.pins:
                entries += 1
                usage += entry.usage
        return {"entries": entries, **vars(usage)}

    def _track_high_water(self, values):
        for dimension, value in values.items():
            self._high_water[dimension] = max(self._high_water[dimension], value)

    def _exceeded(self, usage, requested):
        return [name for name, value in requested.items() if usage.get(name, 0) + value > self._limits[name]]

    def _reject(self, message, *, stage, dimensions, requested, pinned):
        self._reject_count += 1
        for name in dimensions:
            self._reject_by_dimension[name] += 1
        self._last_rejection = {
            "stage": stage,
            "exceeded": dimensions,
            "requested": requested,
            "current": {
                **self._resident(),
                "sessions": len(self._sessions),
                "active_leases": sum(self._active.values()),
            },
            "pinned": pinned,
            "limits": {name: self._limits[name] for name in dimensions},
            "high_water": self._high_water.copy(),
        }
        raise MemoryError(f"{message}: {json.dumps(self._last_rejection, sort_keys=True)}")

    def _make_room(self, usage, *, extra_entry, stage=None):
        requested = {"entries": int(extra_entry), **vars(usage)}
        oversized = self._exceeded({}, requested)
        if oversized:
            self._reject(
                "single condition exceeds budget",
                stage=stage,
                dimensions=oversized,
                requested=requested,
                pinned=self._pinned(),
            )
        if not self._exceeded(self._resident(), requested):
            return
        pinned = self._pinned()
        blocked = self._exceeded(pinned, requested)
        if blocked:
            self._reject(
                "condition cache/request exceeds budget with pinned entries",
                stage=stage,
                dimensions=blocked,
                requested=requested,
                pinned=pinned,
            )
        # Evict LRU unpinned values only when the reservation is feasible.
        for key, entry in tuple(self._entries.items()):
            if not entry.pins:
                self._drop(key)
                if not self._exceeded(self._resident(), requested):
                    return
        raise RuntimeError("inconsistent Condition Cache reservation accounting")

    @contextmanager
    def request(self, session_id: str, *, enabled=True):
        if not isinstance(session_id, str) or not session_id or len(session_id) > 256 or type(enabled) is not bool:
            raise ValueError("bounded Session identity and explicit cache switch required")
        with self._lock:
            if session_id not in self._sessions:
                if len(self._sessions) >= self.budget.max_sessions:
                    victim = next((s for s in self._sessions if not self._active[s]), None)
                    if victim is None:
                        self._reject(
                            "all bounded cache Sessions have active requests",
                            stage="session",
                            dimensions=["sessions"],
                            requested={"sessions": 1},
                            pinned={**self._pinned(), "sessions": len(self._sessions)},
                        )
                    self.reset(victim)
                self._sessions[session_id] = None
                self._active[session_id] = 0
            self._sessions.move_to_end(session_id)
            self._active[session_id] += 1
            self._track_high_water(dict(sessions=len(self._sessions), active_leases=sum(self._active.values())))
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
                "pinned": self._pinned(),
                "high_water": self._high_water.copy(),
                "rejections": {
                    "count": self._reject_count,
                    "by_dimension": self._reject_by_dimension.copy(),
                    "last": deepcopy(self._last_rejection),
                },
                "stages": {key: value.copy() for key, value in self._stats.items()},
                "counter_scope": "process lifetime; model reload empties all resident derived values",
                "timing": "host elapsed; hit/miss separated; no CUDA speed inference",
            }
