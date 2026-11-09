"""Optional bounded profiling evidence, separate from scientific Core JSON.

CPU intervals may overlap and must not be added into a critical-path latency.
CUDA intervals use events on the actual stream, resolve only after query(), and
never synchronize a device or substitute host time for GPU execution time.
"""

from __future__ import annotations

import contextvars
import copy
import hashlib
import json
import math
import os
import stat
import tempfile
import threading
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path

_CURRENT = contextvars.ContextVar("robocasa_feature_trace", default=None)
_RESERVE = 16384


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _fields(fields):
    if len(fields) > 16:
        raise ValueError("bounded trace fields required")
    result = {}
    for key, value in fields.items():
        if (
            type(key) is not str
            or not key.isascii()
            or not 1 <= len(key) <= 64
            or any(
                word in key.lower()
                for word in (
                    "capability",
                    "secret",
                    "owner_token",
                    "access_token",
                    "signing_key",
                    "private_key",
                    "sessioncap",
                    "key_hex",
                )
            )
        ):
            raise ValueError("public bounded trace fields required")
        if type(value) is str:
            if len(value) > 256:
                raise ValueError("bounded trace scalar required")
        elif value is not None and type(value) not in (int, bool, float):
            raise ValueError("trace metadata must contain public scalars")
        if type(value) is float and not math.isfinite(value):
            raise ValueError("finite trace scalar required")
        if type(value) is int and abs(value) >= 2**127:
            raise ValueError("bounded trace integer required")
        result[key] = value
    return result


class TraceRecorder:
    def __init__(
        self,
        *,
        max_events=8192,
        max_bytes=4 * 1024**2,
        max_pending_cuda=128,
        clock_ns=time.monotonic_ns,
        clock_domain="process_monotonic_ns",
        producer="sim",
    ):
        if type(max_events) is not int or not 1 <= max_events <= 100000:
            raise ValueError("bounded trace event count required")
        if type(max_bytes) is not int or not 4096 <= max_bytes <= 64 * 1024**2:
            raise ValueError("bounded raw trace byte budget required")
        if type(max_pending_cuda) is not int or not 1 <= max_pending_cuda <= 1024:
            raise ValueError("bounded CUDA trace queue required")
        _fields({"clock_domain": clock_domain, "producer": producer})
        self.max_events, self.max_bytes, self.max_pending_cuda = max_events, max_bytes, max_pending_cuda
        self._reserve = min(_RESERVE, max_bytes // 2)
        self._max_names = min(64, max(1, (max_bytes // 2 - 512) // 128))
        self.clock_ns, self.clock_domain = clock_ns, clock_domain
        self.producer = producer
        self._lock = threading.RLock()
        self._events, self._sizes, self._counts, self._cuda = [], [], {}, []
        self._bytes = self._attempted = self._overflow = self._cuda_unavailable = self._active = 0
        self._incomplete = set()
        self._watermarks = {}
        self._cuda_reserved = 0

    def now_ns(self):
        value = self.clock_ns()
        if type(value) is not int or value < 0:
            raise ValueError("nonnegative monotonic nanoseconds required")
        return value

    def record(self, name, start_ns, end_ns, *, kind="cpu_span", producer=None, stratum="unspecified", **fields):
        if type(name) is not str or not name.isascii() or not 1 <= len(name) <= 96 or any(ord(c) < 32 for c in name):
            raise ValueError("bounded trace name required")
        if any(type(value) is not int or value < 0 for value in (start_ns, end_ns)) or end_ns < start_ns:
            raise ValueError("ordered monotonic nanosecond interval required")
        _fields({"producer": producer or self.producer, "stratum": stratum, "kind": kind})
        row = {
            "name": name,
            "phase": name,
            "producer": producer or self.producer,
            "stratum": stratum,
            "kind": kind,
            "start_ns": start_ns,
            "end_ns": end_ns,
            "duration_ns": end_ns - start_ns,
            "pid": os.getpid(),
            "thread_id": threading.get_native_id(),
            "fields": _fields(fields),
        }
        raw_size = len(_json(row)) + 1
        with self._lock:
            self._attempted += 1
            if name not in self._counts and len(self._counts) >= self._max_names:
                self._overflow += 1
            else:
                self._counts[name] = self._counts.get(name, 0) + 1
            if len(self._events) >= self.max_events or self._bytes + raw_size > self.max_bytes - self._reserve:
                self._overflow += 1
                return False
            self._events.append(row)
            self._sizes.append(raw_size)
            self._bytes += raw_size
            return True

    def event(self, name, **fields):
        now = self.now_ns()
        return self.record(name, now, now, kind="event", **fields)

    @contextmanager
    def span(self, name, **fields):
        fields = _fields(fields)
        if len(fields) > 14:
            raise ValueError("reserve trace status fields within the bound")
        started, error_type = self.now_ns(), None
        with self._lock:
            self._active += 1
        try:
            yield
        except BaseException as error:
            error_type = type(error).__name__
            raise
        finally:
            ended = self.now_ns()
            with self._lock:
                try:
                    self.record(
                        name,
                        started,
                        ended,
                        **{**fields, "status": "error" if error_type else "ok", "error_type": error_type},
                    )
                finally:
                    self._active -= 1

    def mark_incomplete(self, reason):
        _fields({"reason": reason})
        if type(reason) is not str or not reason.isascii() or not 1 <= len(reason) <= 96:
            raise ValueError("bounded public incompleteness reason required")
        with self._lock:
            if len(self._incomplete) < min(16, self.max_bytes // 512):
                self._incomplete.add(reason)
            else:
                self._overflow += 1

    def merge(self, producer, trace, *, clock_domain=None, deduplicate=True):
        """Merge bounded raw producer snapshots; repeated ring samples deduplicate."""
        _fields({"producer": producer})
        if clock_domain is not None and clock_domain != self.clock_domain:
            self.mark_incomplete("producer_clock_domain_mismatch")
            return
        if type(trace) is list:
            rows, overflow, complete = trace, 0, True
        elif type(trace) is dict:
            rows = trace.get("events", trace.get("trace", []))
            overflow = trace.get("overflow_count", trace.get("trace_overflow", 0))
            complete = trace.get("complete", True)
        else:
            raise ValueError("bounded producer trace snapshot required")
        if type(rows) is not list:
            raise ValueError("producer events must be a bounded list")
        if overflow or complete is not True:
            self.mark_incomplete("producer_trace_overflow_or_incomplete")
        with self._lock:
            if producer not in self._watermarks and len(self._watermarks) >= 8:
                self.mark_incomplete("producer_capacity_exhausted")
                return
            watermark, seen = self._watermarks.get(producer, (-1, set()))
            for row in rows[: self.max_events]:
                if type(row) is not dict:
                    self.mark_incomplete("malformed_producer_event")
                    continue
                start = row.get("start_ns", row.get("at_ns"))
                end = row.get("end_ns", start)
                if type(start) is str and start.isascii() and start.isdecimal():
                    start = int(start)
                if type(end) is str and end.isascii() and end.isdecimal():
                    end = int(end)
                if type(start) is not int or type(end) is not int:
                    self.mark_incomplete("malformed_producer_timestamp")
                    continue
                fingerprint = hashlib.sha256(_json(row)).hexdigest()
                if deduplicate and (end < watermark or (end == watermark and fingerprint in seen)):
                    continue
                if deduplicate:
                    if end > watermark:
                        watermark, seen = end, set()
                    if len(seen) >= 256:
                        self.mark_incomplete("producer_same_timestamp_capacity")
                        continue
                    seen.add(fingerprint)
                fields = row.get("fields", {})
                if type(fields) is not dict:
                    fields = {}
                fields = {k: v for k, v in fields.items() if k not in ("producer", "stratum")}
                for key in ("source_step", "job", "session", "batch", "size", "reason", "items", "bytes"):
                    if key in row:
                        fields[key] = row[key]
                self.record(
                    row.get("phase", row.get("name", row.get("event", "producer_event"))),
                    start,
                    end,
                    kind=row.get("kind", "producer_event"),
                    producer=producer,
                    stratum=row.get("stratum", "unspecified"),
                    **fields,
                )
                if "profile" in row:
                    if producer == "worker":
                        self.merge("worker_vae", row["profile"], clock_domain=clock_domain, deduplicate=False)
                    else:
                        self.mark_incomplete("unexpected_nested_producer_profile")
            if deduplicate:
                self._watermarks[producer] = watermark, seen
            if len(rows) > self.max_events:
                self.mark_incomplete("producer_snapshot_truncated")

    @contextmanager
    def cuda_span(self, name, *, device=None, stream=None, **fields):
        """Measure the selected actual stream; no global/device sync is performed."""
        fields = _fields(fields)
        if len(fields) > 10:
            raise ValueError("reserve bounded CUDA provenance fields")
        import torch  # Lazy: ordinary CPU profiling never imports torch.

        if not torch.cuda.is_available():
            with self._lock:
                self._cuda_unavailable += 1
            self.event(name, gpu_timing="unavailable", **fields)
            yield
            return
        with self._lock:
            full = len(self._cuda) + self._cuda_reserved >= self.max_pending_cuda
            if full:
                self._overflow += 1
            else:
                self._cuda_reserved += 1
        if full:
            yield
            return
        reserved, body_error = True, None
        try:
            selected = torch.cuda.current_stream(device) if stream is None else stream
            if device is not None:
                expected = torch.device("cuda", device) if type(device) is int else torch.device(device)
                if expected.index is None:
                    expected = torch.device("cuda", torch.cuda.current_device())
                if selected.device != expected:
                    raise ValueError("CUDA trace stream/device mismatch")
            started, ended = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            host_start = self.now_ns()
            with torch.cuda.device(selected.device), torch.cuda.stream(selected):
                started.record(selected)
                try:
                    yield
                except BaseException as error:
                    body_error = error
                    raise
                finally:
                    try:
                        ended.record(selected)
                        pending = (
                            name,
                            host_start,
                            self.now_ns(),
                            started,
                            ended,
                            str(selected.device),
                            int(selected.cuda_stream),
                            {**fields, "error_type": None if body_error is None else type(body_error).__name__},
                        )
                        with self._lock:
                            self._cuda_reserved -= 1
                            reserved = False
                            self._cuda.append(pending)
                    except Exception as error:
                        self.mark_incomplete("cuda_event_record_failed")
                        if body_error is None:
                            raise
                        body_error.trace_error = type(error).__name__
        finally:
            if reserved:
                with self._lock:
                    self._cuda_reserved -= 1
                    self.mark_incomplete("cuda_event_setup_failed")

    def resolve_cuda(self):
        """Call after an existing transfer/quiesce; pending events stay explicit."""
        with self._lock:
            pending, self._cuda = self._cuda, []
            for row in pending:
                name, start_ns, end_ns, started, ended, device, stream, fields = row
                try:
                    ready = ended.query() and started.query()
                    elapsed = float(started.elapsed_time(ended)) if ready else None
                    if ready and (not math.isfinite(elapsed) or elapsed < 0):
                        raise ValueError("invalid actual CUDA elapsed time")
                except Exception:
                    self.mark_incomplete("cuda_event_query_failed")
                    self._cuda.append(row)
                    continue
                if not ready:
                    self._cuda.append(row)
                    continue
                self.record(
                    name,
                    start_ns,
                    end_ns,
                    kind="cuda_event_span",
                    device=device,
                    stream=stream,
                    gpu_elapsed_ns=int(elapsed * 1_000_000),
                    host_interval="submission_only",
                    gpu_scope="selected_stream",
                    **fields,
                )
            return len(self._cuda)

    def _metadata(self):
        complete = not (
            self._overflow
            or self._cuda
            or self._cuda_reserved
            or self._cuda_unavailable
            or self._active
            or self._incomplete
        )
        return {
            "schema": "psm24_feature_trace_v1",
            "clock_domain": self.clock_domain,
            "count": len(self._events),
            "attempted_count": self._attempted,
            "counts": dict(self._counts),
            "overflow_count": self._overflow,
            "cuda_pending": len(self._cuda),
            "cuda_active": self._cuda_reserved,
            "cuda_unavailable": self._cuda_unavailable,
            "active_spans": self._active,
            "incomplete_reasons": sorted(self._incomplete),
            "complete": complete,
            "max_events": self.max_events,
            "max_bytes": self.max_bytes,
            "overlap_semantics": "intervals_overlap_do_not_sum_into_critical_path",
        }

    def snapshot(self):
        with self._lock:
            self._bounded_raw()
            return {**self._metadata(), "events": copy.deepcopy(self._events)}

    def _bounded_raw(self):
        raw = _json({**self._metadata(), "events": self._events})
        while len(raw) > self.max_bytes and self._events:
            self._events.pop()
            self._bytes -= self._sizes.pop()
            self._overflow += 1
            raw = _json({**self._metadata(), "events": self._events})
        if len(raw) > self.max_bytes:
            raise ValueError("trace metadata exceeds bounded evidence file")
        return raw

    def dump(self, path):
        """Publish one append-only bounded raw file; return only small metadata."""
        self.resolve_cuda()
        with self._lock:
            raw = self._bounded_raw()
            metadata = self._metadata()
        path = Path(path)
        for parent in (path.parent, *path.parent.parents):
            if not stat.S_ISDIR(parent.lstat().st_mode):
                raise ValueError("trace parents must be real existing directories")
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        temporary = None
        try:
            descriptor, temporary = tempfile.mkstemp(prefix=".feature-trace-", dir=path.parent)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            os.link(temporary, path, follow_symlinks=False)
            os.fsync(directory)
        finally:
            if temporary is not None:
                os.unlink(temporary)
            os.close(directory)
        return {**metadata, "path": path.name, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def get_trace():
    return _CURRENT.get()


@contextmanager
def use_trace(recorder):
    token = _CURRENT.set(recorder)
    try:
        yield recorder
    finally:
        _CURRENT.reset(token)


def span(name, **fields):
    recorder = get_trace()
    return nullcontext() if recorder is None else recorder.span(name, **fields)
