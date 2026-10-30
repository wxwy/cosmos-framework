"""Production Episode slots, scoped faults and recoverable initialization.

No legacy ingestion, inference, ffmpeg, simulator, GPU or subprocess exists here.
A single coordinator holds the run lease; real multi-GPU transport is a later Gate.
"""

from __future__ import annotations

import contextlib
import copy
import fcntl
import json
import os
import secrets
import stat
import time
import uuid
from collections import deque
from functools import lru_cache, wraps
from pathlib import Path

from cosmos_framework.evaluation.robocasa_v2.contract import (
    canonical,
    digest,
    integer,
    validate_manifest,
    validate_result,
)

MAX_JSON = 256 * 1024
MAX_REPORT = 16 * 1024 * 1024
MAX_RUNTIME_PROBE = 64 * 1024 * 1024


def safe(path):
    path = Path(os.path.abspath(path))
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError(f"symlink path forbidden: {path}")
    return path


def sync_dir(path):
    fd = os.open(safe(path), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def mkdir(path):
    path = safe(path)
    if path.exists():
        if not path.is_dir():
            raise ValueError("directory required")
        return
    mkdir(path.parent)
    try:
        path.mkdir()
    except FileExistsError:
        # A concurrent creator may have won; accept only a real directory.
        path = safe(path)
        if not path.is_dir():
            raise ValueError("directory required")
    sync_dir(path.parent)


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


def read(path):
    path = safe(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_JSON:
            raise ValueError("ordinary bounded JSON required")
        raw = stream.read(MAX_JSON + 1)
        after = os.fstat(stream.fileno())

    def witness(s):
        return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns

    if len(raw) > MAX_JSON or witness(before) != witness(after) or witness(after) != witness(path.stat()):
        raise ValueError("JSON changed during read")
    return json.loads(
        raw, object_pairs_hook=_pairs, parse_constant=lambda x: (_ for _ in ()).throw(ValueError("nonfinite JSON"))
    )


def _publish(path, value, *, replace=False, limit=MAX_JSON):
    path = safe(path)
    payload = canonical(value) + b"\n"
    if len(payload) > limit:
        raise ValueError("oversized JSON")
    mkdir(path.parent)
    temporary = path.parent / f".pending-{uuid.uuid4().hex}"
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path, follow_symlinks=False)
        sync_dir(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def write(path, value, *, replace=False):
    _publish(path, value, replace=replace, limit=MAX_JSON)


def write_report(path, value):
    if Path(path).name != "summary.json":
        raise ValueError("report limit cannot be used for Core/ledger records")
    _publish(path, value, replace=True, limit=MAX_REPORT)


def write_runtime_probe(path, value):
    """Append bounded diagnostics only within the runtime probe namespace."""
    path = safe(path)
    if path.parent.name != "runtime_probe" or path.parent.parent.name != "_internal" or path.suffix != ".json":
        raise ValueError("runtime probe UUID JSON path required")
    identifier(path.stem)
    _publish(path, value, limit=MAX_RUNTIME_PROBE)


def sealed(value):
    return {"body": value, "sha256": digest(value)}


READ_METRICS = {"disk_reads": 0, "cache_hits": 0}


def witness(path):
    path = Path(path)
    value = path.lstat()
    if not stat.S_ISREG(value.st_mode):
        raise ValueError("ordinary immutable record required")
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


@lru_cache(maxsize=32768)
def _cached_document(path, stamp):
    READ_METRICS["disk_reads"] += 1
    doc = read(path)
    if witness(path) != stamp:
        raise ValueError("record changed during cached verification")
    if not isinstance(doc, dict) or set(doc) != {"body", "sha256"} or doc["sha256"] != digest(doc["body"]):
        raise ValueError("corrupt immutable record")
    return doc["body"]


def load(path):
    path = safe(path)
    before = READ_METRICS["disk_reads"]
    value = _cached_document(str(path), witness(path))
    READ_METRICS["cache_hits"] += READ_METRICS["disk_reads"] == before
    return copy.deepcopy(value)  # Callers never mutate the validated cache.


def save(path, value):
    write(path, sealed(value))


def identifier(value):
    if not isinstance(value, str) or len(value) != 32 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("invalid identifier")
    return value


def records(directory):
    directory = safe(directory)
    if not directory.exists():
        return []
    result = []
    for path in sorted(directory.iterdir()):
        if path.name.startswith(".pending-"):
            continue  # Never published; retained for diagnosis, not authority.
        if path.suffix != ".json":
            raise ValueError("unexpected ledger artifact")
        result.append(load(path))
    return result


def timed(name):
    def decorate(method):
        @wraps(method)
        def call(self, *args, **kwargs):
            start = time.perf_counter()
            ok = False
            try:
                result = method(self, *args, **kwargs)
                ok = True
                return result
            finally:
                self.record_timing(name, time.perf_counter() - start, ok)

        return call

    return decorate


class Store:
    """Single coordinator; all worker responses require current leased attempt tokens.

    The native runtime supplies verified process/session/runtime proof references.
    This persistence layer never discovers or kills processes by itself.
    """

    def __init__(self, root, manifest, fault=None):
        self.root = safe(root)
        self.manifest = validate_manifest(manifest)
        self.held = False
        self.active_tasks = {}
        self.active_attempts = {}
        self.fault = fault or (lambda stage: None)
        self.scheduler_attached = False
        self.dispatch_tickets = {}
        self._rowcache = {}
        self._reset_counts = {}
        self._timings = {}
        self._read_baseline = dict(READ_METRICS)

    def record_timing(self, name, seconds, ok=True):
        entry = self._timings.setdefault(name, {"count": 0, "failed": 0, "samples": deque(maxlen=4096)})
        entry["count"] += 1
        entry["failed"] += not ok
        entry["samples"].append(seconds * 1000)

    def metrics(self):
        result = {}
        for name, entry in self._timings.items():
            values = sorted(entry["samples"])
            if not values:
                continue
            result[name] = {
                "count": entry["count"],
                "failed": entry["failed"],
                "sample_count": len(values),
                "p50_ms": values[int((len(values) - 1) * 0.5)],
                "p95_ms": values[int((len(values) - 1) * 0.95)],
                "max_ms": values[-1],
            }
        return {
            "coordination": result,
            "record_io": {k: READ_METRICS[k] - self._read_baseline[k] for k in READ_METRICS},
        }

    def create(self):
        mkdir(self.root.parent)
        self.root.mkdir()  # No adoption of empty/old431 directories.
        sync_dir(self.root.parent)
        mkdir(self.root / "_internal")
        fd = os.open(self.root / "_internal/.run.lock", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.fsync(fd)
        os.close(fd)
        sync_dir(self.root / "_internal")
        write(self.root / "manifest.json", self.manifest)
        mkdir(self.root / "tasks")
        mkdir(self.root / "logs")

    @contextlib.contextmanager
    def locked(self):
        if self.held:
            raise RuntimeError("nested lease")
        if read(self.root / "manifest.json") != self.manifest:
            raise ValueError("new run experiment identity mismatch")
        path = safe(self.root / "_internal/.run.lock")
        fd = os.open(path, os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise ValueError("ordinary lease required")
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if (os.fstat(fd).st_dev, os.fstat(fd).st_ino) != (path.stat().st_dev, path.stat().st_ino):
                raise ValueError("lease inode changed")
            if read(self.root / "manifest.json") != self.manifest:
                raise ValueError("manifest changed before lease")
            self.held, self.lease_id = True, uuid.uuid4().hex
            try:
                yield self
            finally:
                self.held = False
                self.active_tasks.clear()
                self.active_attempts.clear()
                self.scheduler_attached = False
                self.dispatch_tickets.clear()
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def _lease(self):
        if not self.held:
            raise RuntimeError("coordinator lease required")

    def _task(self, task):
        if task not in self.manifest["tasks"]:
            raise ValueError("unplanned Task")
        return safe(self.root / "tasks" / task)

    def _internal(self, task):
        self._task(task)
        return self.root / "_internal/tasks" / task

    def _slot(self, task, ep):
        integer(ep, 0, self.manifest["episodes_per_task"] - 1, "episode")
        return self._task(task) / "episodes" / f"ep_{ep:03d}"

    def _ledger(self, task, ep):
        self._slot(task, ep)
        return self._internal(task) / "slots" / f"ep_{ep:03d}"

    def _claim(self, task, ep=None):
        return {"manifest_digest": self.manifest["digest"], "task": task, "episode": ep}

    def _initialize(self, task, ep=None):
        """Intent -> directory -> ready; intent is never removed or overwritten.

        No producer token is issued before ready. Thus missing ready can recover
        ONLY if the public subtree contains no producer evidence whatsoever.
        """
        self._lease()
        public = self._task(task) / "episodes" if ep is None else self._slot(task, ep)
        ledger = self._internal(task) if ep is None else self._ledger(task, ep)
        intent, ready = ledger / "init_intent.json", ledger / "ready.json"
        expected = self._claim(task, ep)
        if not safe(intent).exists():
            if safe(ready).exists():
                raise ValueError("ready exists but init intent lost")
            if public.exists() and any(public.iterdir()):
                raise ValueError("ambiguous unclaimed public data")
            self.fault("before_intent")
            save(intent, expected)
            self.fault("after_intent")
        if load(intent) != expected:
            raise ValueError("foreign init claim")
        if safe(ready).exists():
            if load(ready) != expected or not public.is_dir():
                raise ValueError("lost ready data; cannot reconstruct")
            return
        if ledger.exists() and any(
            p.name not in ("init_intent.json", "ready.json") and not p.name.startswith(".pending-")
            for p in ledger.iterdir()
        ):
            raise ValueError("producer ledger before ready; ambiguous")
        if public.exists() and any(public.iterdir()):
            raise ValueError("producer evidence before ready; ambiguous")
        mkdir(public)
        self.fault("after_directory")
        save(ready, expected)
        self.fault("after_ready")

    def _streams(self, task):
        values = records(self._internal(task) / "streams")
        ids = set()
        for v in values:
            if (
                set(v) != {"task", "stream_id", "worker_id", "lease_id", "seed", "manifest_digest", "kind", "time_ns"}
                or v["task"] != task
                or v["manifest_digest"] != self.manifest["digest"]
                or v["kind"] not in ("initial_dynamic_slot_stream", "resume_dynamic_slot_stream")
            ):
                raise ValueError("foreign stream")
            for key in ("stream_id", "worker_id", "lease_id"):
                identifier(v[key])
            integer(v["seed"], 0, 2**32 - 1, "stream seed")
            if v["stream_id"] in ids:
                raise ValueError("duplicate stream")
            if load(self._internal(task) / "streams" / f"{v['stream_id']}.json") != v:
                raise ValueError("stream file identity mismatch")
            ids.add(v["stream_id"])
        return values

    def _stream(self, task, stream_id):
        identifier(stream_id)
        matches = [s for s in self._streams(task) if s["stream_id"] == stream_id]
        if len(matches) != 1:
            raise ValueError("unknown stream")
        return matches[0]

    def cleanup_stream(self, task, stream_id, probe):
        self._lease()
        stream = self._stream(task, stream_id)
        proof = probe(dict(stream))
        if (
            not isinstance(proof, dict)
            or set(proof) != {"probe_id", "owned_processes_gone", "local_sessions_closed"}
            or not isinstance(proof["probe_id"], str)
            or not proof["probe_id"]
            or proof["owned_processes_gone"] is not True
            or proof["local_sessions_closed"] is not True
        ):
            raise ValueError("unsafe cleanup: ownership/session proof required")
        value = {"stream": stream, "proof": proof, "time_ns": time.time_ns()}
        path = self._internal(task) / "cleanup" / f"{stream_id}.json"
        if safe(path).exists():
            return load(path)
        save(path, value)
        self.active_tasks.pop(task, None)
        self.active_attempts = {k: a for k, a in self.active_attempts.items() if a["stream_id"] != stream_id}
        return value

    def _cleaned(self, task, stream_id):
        path = safe(self._internal(task) / "cleanup" / f"{identifier(stream_id)}.json")
        if not path.exists():
            return None
        value = load(path)
        if value["stream"] != self._stream(task, stream_id) or any(
            value["proof"].get(k) is not True for k in ("owned_processes_gone", "local_sessions_closed")
        ):
            raise ValueError("invalid cleanup receipt")
        return value

    def allocate_environment(self, task, physical_slot_id):
        """Durable per-Task environment seed sequence, including Resume rebuilds.

        Called under the coordinator's Slot startup lock. A claimed but unused
        grant is retained and consumes its seed, so crashes never reuse it.
        """
        self._lease()
        identifier(physical_slot_id)
        self._healthy_stream(task)
        directory = self._internal(task) / "environments"
        prior = records(directory)
        ordinals = [v["ordinal"] for v in prior]
        if sorted(ordinals) != list(range(len(prior))):
            raise ValueError("environment grant sequence damaged")
        ordinal = len(prior)
        if ordinal >= 2**32:
            raise ValueError("environment seed space exhausted")
        value = {
            "task": task,
            "environment_id": uuid.uuid4().hex,
            "physical_slot_id": physical_slot_id,
            "ordinal": ordinal,
            "seed": (self.manifest["seed"] + ordinal) % 2**32,
            "manifest_digest": self.manifest["digest"],
        }
        save(directory / f"{value['environment_id']}.json", value)
        return value

    def _check_sampling(self, a, sampling):
        grant = load(self._internal(a["task"]) / "environments" / f"{identifier(sampling['environment_id'])}.json")
        if (
            grant["task"] != a["task"]
            or grant["manifest_digest"] != self.manifest["digest"]
            or grant["seed"] != sampling["actual_seed"]
            or grant["physical_slot_id"] != sampling["physical_slot_id"]
        ):
            raise ValueError("scene sampling contradicts durable environment seed grant")

    def bind_sampling(self, a, sampling):
        self._lease()
        self._attempt(a)
        if self.active_attempts.get((a["task"], a["episode"])) != a or self.events(a)[-1]["kind"] != "ADMITTED":
            raise ValueError("owned admitted scene required for sampling binding")
        self._check_sampling(a, sampling)
        save(
            self._ledger(a["task"], a["episode"]) / "sampling" / f"{a['attempt_id']}.json",
            {"attempt": a, "sampling": sampling},
        )

    def _validate_sampling(self, a, row):
        if "sampling" in row:
            self._check_sampling(a, row["sampling"])
            if load(self._ledger(a["task"], a["episode"]) / "sampling" / f"{a['attempt_id']}.json") != {
                "attempt": a,
                "sampling": row["sampling"],
            }:
                raise ValueError("Core sampling is not the admitted scene")

    def begin_task(self, task, *, seed=None, recovery_only=False):
        self._lease()
        if task in self.active_tasks:
            raise RuntimeError("Task already scheduled")
        self._initialize(task)
        previous = self._streams(task)
        if any(not self._cleaned(task, s["stream_id"]) for s in previous):
            raise ValueError("safe stop: previous owned stream not cleaned")
        if not recovery_only and len(self.rows(task)) == self.manifest["episodes_per_task"]:
            return {"action": "reuse_complete_task", "task": task}
        stream = {
            "task": task,
            "stream_id": uuid.uuid4().hex,
            "worker_id": uuid.uuid4().hex,
            "lease_id": self.lease_id,
            "seed": self.manifest["seed"] if not previous else secrets.randbits(32),
            "manifest_digest": self.manifest["digest"],
            "kind": "initial_dynamic_slot_stream" if not previous else "resume_dynamic_slot_stream",
            "time_ns": time.time_ns(),
        }
        if seed is not None:
            stream["seed"] = integer(seed, 0, 2**32 - 1, "seed")
        save(self._internal(task) / "streams" / f"{stream['stream_id']}.json", stream)
        self.fault("after_stream")
        self.active_tasks[task] = stream
        return {"action": "start_stream", **stream}

    def _attempts(self, task, ep):
        values = records(self._ledger(task, ep) / "attempts")
        for i, a in enumerate(values):
            if (
                set(a)
                != {
                    "task",
                    "episode",
                    "generation",
                    "attempt_id",
                    "stream_id",
                    "worker_id",
                    "lease_id",
                    "seed",
                    "manifest_digest",
                    "authorization",
                    "time_ns",
                }
                or a["generation"] != i
                or type(a["generation"]) is not int
                or a["task"] != task
                or a["episode"] != ep
                or a["manifest_digest"] != self.manifest["digest"]
            ):
                raise ValueError("invalid attempt claim/order")
            if a["authorization"] is not None:
                auth = a["authorization"]
                identifier(auth["id"])
                stored = load(self._ledger(task, ep) / "authorizations" / f"{auth['id']}.json")
                if (
                    auth != stored
                    or i == 0
                    or auth["prior_attempt"] != values[i - 1]
                    or auth["task"] != task
                    or auth["episode"] != ep
                    or not auth["owner_ref"].strip()
                ):
                    raise ValueError("invalid persisted Owner authorization")
            if load(self._ledger(task, ep) / "attempts" / f"{i:06d}.json") != a:
                raise ValueError("attempt file sequence mismatch")
            identifier(a["attempt_id"])
            s = self._stream(task, a["stream_id"])
            if any(a[k] != s[k] for k in ("worker_id", "lease_id", "seed")):
                raise ValueError("attempt stream/worker mismatch")
        return values

    def _attempt(self, a):
        matches = [x for x in self._attempts(a["task"], a["episode"]) if x["attempt_id"] == a["attempt_id"]]
        if matches != [a]:
            raise ValueError("unknown/changed attempt token")
        return a

    def events(self, a):
        self._attempt(a)
        values = records(self._ledger(a["task"], a["episode"]) / "events" / a["attempt_id"])
        for i, value in enumerate(values):
            if (
                set(value) != {"attempt", "sequence", "kind", "data", "time_ns"}
                or value["attempt"] != a
                or value["sequence"] != i
            ):
                raise ValueError("corrupt attempt event identity/order")
            if value["kind"] in ("ADMITTED", "DEGENERATE_INITIAL_SUCCESS") and (
                i == 0 or values[i - 1]["kind"] != "INIT_BEGIN" or value["data"] != values[i - 1]["data"]
            ):
                raise ValueError("admission observation contradicts reset sample")
            if load(self._ledger(a["task"], a["episode"]) / "events" / a["attempt_id"] / f"{i:06d}.json") != value:
                raise ValueError("event file sequence mismatch")
        return values

    def _event(self, a, kind, data):
        self._lease()
        self._attempt(a)
        if self.active_attempts.get((a["task"], a["episode"])) != a:
            raise ValueError("stale attempt/coordinator response")
        self._healthy_stream(a["task"], allow_fault=kind in ("UNSAFE_INIT", "INVALID"))
        values = self.events(a)
        save(
            self._ledger(a["task"], a["episode"]) / "events" / a["attempt_id"] / f"{len(values):06d}.json",
            {"attempt": a, "sequence": len(values), "kind": kind, "data": data, "time_ns": time.time_ns()},
        )
        self.fault("after_event_" + kind)

    def _healthy_stream(self, task, *, allow_fault=False):
        stream = self.active_tasks.get(task)
        if stream is None or stream["lease_id"] != self.lease_id or self._cleaned(task, stream["stream_id"]):
            raise ValueError("stream not owned or already closed")
        if not allow_fault and any(
            e["stream_id"] == stream["stream_id"] and e["resolution"] is None for e in self.errors(task)
        ):
            raise ValueError("safe stop: current process fault")
        return stream

    def _admission_state(self, task, ep):
        count, consecutive, status = 0, 0, "pending"
        last = None
        attempts = self._attempts(task, ep)
        for a in attempts:
            if status in ("admitted", "invalid", "interrupted", "initializing"):
                status = "pending"
            if a["authorization"]:
                count, consecutive, status = 0, 0, "pending"  # Explicit Owner-approved new budget round.
            last_kind = None
            for e in self.events(a):
                kind, data = e["kind"], e["data"]
                if kind == "INIT_BEGIN":
                    if last_kind == "INIT_BEGIN" or status in (
                        "admitted",
                        "skipped_init_error",
                        "scene_admission_exhausted",
                    ):
                        raise ValueError("illegal initialization transition")
                    if data["sample_no"] != count + 1 or count >= self.manifest["protocol"]["admission_limit"]:
                        raise ValueError("admission budget/order mismatch")
                    integer(data["reset_index"], 0, 100000, "reset index")
                    count += 1
                    status = "initializing"
                elif kind in ("INIT_ERROR", "DEGENERATE_INITIAL_SUCCESS", "ADMITTED", "UNSAFE_INIT"):
                    if last_kind != "INIT_BEGIN":
                        raise ValueError("initialization outcome without reset")
                    if kind == "INIT_ERROR":
                        if (
                            set(data) != {"stage", "type", "message", "safe_to_retry"}
                            or data["safe_to_retry"] is not True
                        ):
                            raise ValueError("invalid recoverable error evidence")
                        consecutive += 1
                        status = "skipped_init_error" if consecutive == 4 else "pending"
                    elif kind == "UNSAFE_INIT":
                        status = "invalid"
                    else:
                        consecutive = 0
                        status = "admitted" if kind == "ADMITTED" else "pending"
                elif kind == "ENV_REBUILT":
                    identifier(data["environment_id"])
                elif kind == "INVALID":
                    status = "invalid"
                elif kind == "INTERRUPTED":
                    status = "interrupted"  # Not an invented initialization error.
                else:
                    raise ValueError("unknown admission event")
                last_kind = kind
                last = e
            # A new attempt cannot inherit an already-admitted scene.
            if status == "admitted":
                status = "interrupted"
        if last and last["kind"] == "ADMITTED" and last["attempt"] == attempts[-1]:
            status = "admitted"
        if status == "pending" and count >= self.manifest["protocol"]["admission_limit"]:
            status = "scene_admission_exhausted"
        return {"count": count, "consecutive_errors": consecutive, "status": status, "last_event": last}

    def _read_result(self, task, ep):
        path = safe(self._slot(task, ep) / "result.json")
        if not path.exists():
            return None
        for ledger, expected in (
            (self._internal(task), self._claim(task)),
            (self._ledger(task, ep), self._claim(task, ep)),
        ):
            if load(ledger / "ready.json") != expected or load(ledger / "init_intent.json") != expected:
                raise ValueError("Core result lacks initialization authority")
        row = load(path)
        validate_result(self.manifest, row)
        attempts = self._attempts(task, ep)
        if not attempts:
            raise ValueError("Core result lacks attempt")
        a = attempts[-1]
        if any(
            row[k] != a[k] for k in ("task", "episode", "generation", "attempt_id", "stream_id", "worker_id", "seed")
        ):
            raise ValueError("foreign/stale Core result")
        events = self.events(a)
        if not events or events[-1]["kind"] != "ADMITTED" or row["reset_index"] != events[-1]["data"]["reset_index"]:
            raise ValueError("Core result lacks admitted scene")
        self._validate_sampling(a, row)
        if row["outcome"] not in ("success", "failure"):
            raise ValueError("only valid environment outcomes can Core Commit")
        return row

    def _result(self, task, ep):
        from cosmos_framework.evaluation.robocasa_v2.evidence import FILES, validate

        path = self._slot(task, ep) / "result.json"
        if not path.exists():
            return None
        keys = [path, path.parent / "core_inventory.json", *(path.parent / f for f in FILES)]
        # Guard private provenance too: corruption must not be hidden by a cached
        # public result. Immutable records are re-read only when stat changes.
        keys += list(self._ledger(task, ep).rglob("*.json"))
        keys += [self._internal(task) / f for f in ("ready.json", "init_intent.json")]
        stamps = tuple((str(p), witness(p)) for p in keys)
        cached = self._rowcache.get((task, ep))
        if cached is not None and cached[0] == stamps:
            return copy.deepcopy(cached[1])
        row = self._read_result(task, ep)
        inventory = load(path.parent / "core_inventory.json")
        if inventory != {"row_digest": digest(row), "files": validate(path.parent, row)}:
            raise ValueError("Core evidence inventory mismatch")
        self._rowcache[(task, ep)] = (stamps, copy.deepcopy(row))
        return row

    def rows(self, task):
        directory = safe(self._task(task) / "episodes")
        if not directory.exists():
            ready = safe(self._internal(task) / "ready.json")
            if ready.exists():
                raise ValueError("lost ready Task directory")
            return []
        values = []
        for epdir in sorted(directory.iterdir()):
            safe(epdir)
            try:
                index = int(epdir.name.removeprefix("ep_"))
            except ValueError as exc:
                raise ValueError("unexpected Episode directory") from exc
            if epdir.name != f"ep_{index:03d}" or not epdir.is_dir():
                raise ValueError("unexpected Episode directory")
            self._slot(task, index)
            row = self._result(task, index)
            if row is not None:
                values.append(row)
        return values  # Deliberately permits gaps, unlike superseded r1.

    def slot(self, task, ep):
        row = self._result(task, ep)
        if row is not None:
            return {"status": row["outcome"], "result": row}
        state = self._admission_state(task, ep)
        if state["status"] in ("initializing", "admitted"):
            state = dict(state, status="interrupted")
        return {**state, "result": None}

    def authorize_retry(self, task, ep, *, owner_ref):
        self._lease()
        if not isinstance(owner_ref, str) or not owner_ref.strip():
            raise ValueError("explicit Owner decision reference required")
        if self.slot(task, ep)["status"] not in ("skipped_init_error", "scene_admission_exhausted"):
            raise ValueError("only blocked missing slots may be authorized")
        a = self._attempts(task, ep)[-1]
        value = {"task": task, "episode": ep, "prior_attempt": a, "owner_ref": owner_ref, "id": uuid.uuid4().hex}
        save(self._ledger(task, ep) / "authorizations" / f"{value['id']}.json", value)
        return value

    def _permit(self, task, ep):
        attempts = self._attempts(task, ep)
        if not attempts:
            return None
        values = records(self._ledger(task, ep) / "authorizations")
        available = [v for v in values if v["prior_attempt"] == attempts[-1]]
        for v in available:
            if v["task"] != task or v["episode"] != ep or not v["owner_ref"].strip():
                raise ValueError("invalid Owner authorization")
        return available[0] if available else None

    @timed("claim")
    def begin_episode(self, task, ep, *, dispatch_ticket=None):
        self._lease()
        if self.scheduler_attached and (
            dispatch_ticket is None or self.dispatch_tickets.get((task, ep)) != dispatch_ticket
        ):
            raise ValueError("round scheduler dispatch ticket required")
        if self._result(task, ep) is not None:
            raise ValueError("valid Episode immutable, including behavioral_failure")
        stream = self._healthy_stream(task)
        status = self.slot(task, ep)["status"]
        permit = self._permit(task, ep)
        if status in ("skipped_init_error", "scene_admission_exhausted") and permit is None:
            raise ValueError("Owner authorization required, no automatic skip retry")
        for i in range(ep):
            earlier = self.slot(task, i)
            if earlier["result"] is None and (
                earlier["status"] not in ("skipped_init_error", "scene_admission_exhausted") or self._permit(task, i)
            ):
                raise ValueError("execute missing logical slots in order")
        self._initialize(task, ep)
        attempts = self._attempts(task, ep)
        if attempts:
            old = attempts[-1]
            if not self._cleaned(task, old["stream_id"]):
                raise ValueError("previous attempt cleanup required")
        if (task, ep) in self.active_attempts:
            raise ValueError("Episode already owned in this lease")
        a = {k: stream[k] for k in ("task", "stream_id", "worker_id", "lease_id", "seed", "manifest_digest")}
        a.update(
            episode=ep,
            generation=len(attempts),
            attempt_id=uuid.uuid4().hex,
            authorization=permit,
            time_ns=time.time_ns(),
        )
        save(self._ledger(task, ep) / "attempts" / f"{len(attempts):06d}.json", a)
        self.fault("after_attempt_claim")
        self.active_attempts[(task, ep)] = a
        return a

    def begin_initialization(self, a):
        state = self._admission_state(a["task"], a["episode"])
        if state["status"] in ("admitted", "skipped_init_error", "scene_admission_exhausted", "invalid"):
            raise ValueError("admission already terminal/blocked")
        if state["last_event"] and state["last_event"]["kind"] == "INIT_BEGIN":
            # Only a NEW owned attempt can recover an ambiguous unfinished reset.
            if state["last_event"]["attempt"] == a:
                raise ValueError("reset still in flight")
            self._event(a, "INTERRUPTED", {"reason": "previous initialization did not publish an outcome"})
        key = a["task"], a["stream_id"]
        if key not in self._reset_counts:
            resets = [
                e["data"]["reset_index"]
                for i in range(self.manifest["episodes_per_task"])
                for x in self._attempts(a["task"], i)
                if x["stream_id"] == a["stream_id"]
                for e in self.events(x)
                if e["kind"] == "INIT_BEGIN"
            ]
            self._reset_counts[key] = max(resets, default=-1) + 1
        data = {"sample_no": state["count"] + 1, "reset_index": self._reset_counts[key]}
        self._reset_counts[key] += 1
        self._event(a, "INIT_BEGIN", data)
        return data

    def initialization_outcome(self, a, *, initial_success=None, error=None, stage=None, safe_to_retry=False):
        events = self.events(a)
        if not events or events[-1]["kind"] != "INIT_BEGIN":
            raise ValueError("initialization not in flight")
        data = events[-1]["data"]
        if error is not None:
            if stage not in ("reset", "initial_success") or not isinstance(error, Exception):
                raise ValueError("typed initialization error and stage required")
            detail = {
                "stage": stage,
                "type": type(error).__name__,
                "message": str(error)[:2048],
                "safe_to_retry": safe_to_retry is True,
            }
            if safe_to_retry is not True:
                self.record_process_error(a["task"], "unsafe_initialization", str(error), attempt=a)
                self.fault("after_unsafe_fault_before_event")
            self._event(a, "INIT_ERROR" if safe_to_retry is True else "UNSAFE_INIT", detail)
        else:
            # Type errors are initial-success errors, never truthy coercions.
            legal = type(initial_success) is bool or (
                type(initial_success).__module__ == "numpy" and type(initial_success).__name__ in ("bool", "bool_")
            )
            if not legal:
                return self.initialization_outcome(
                    a,
                    error=ValueError("illegal predicate return type"),
                    stage="initial_success",
                    safe_to_retry=safe_to_retry,
                )
            self._event(a, "DEGENERATE_INITIAL_SUCCESS" if bool(initial_success) else "ADMITTED", data)
        return self._admission_state(a["task"], a["episode"])

    def finish_invalid(self, a, reason):
        self.record_process_error(a["task"], "core_invalid", reason, attempt=a)
        self.fault("after_invalid_fault_before_event")
        self._event(a, "INVALID", {"reason": reason})

    @timed("core_commit")
    def commit(self, row, *, staging, dispatch_ticket=None):
        self._lease()
        if self.scheduler_attached and (
            dispatch_ticket is None or self.dispatch_tickets.get((row["task"], row["episode"])) != dispatch_ticket
        ):
            raise ValueError("round scheduler commit ticket required")
        validate_result(self.manifest, row)
        key = row["task"], row["episode"]
        a = self.active_attempts.get(key)
        if a is None:
            raise ValueError("stale producer/coordinator")
        self._healthy_stream(row["task"])
        if any(
            row[k] != a[k] for k in ("task", "episode", "generation", "attempt_id", "stream_id", "worker_id", "seed")
        ):
            raise ValueError("foreign/misrouted attempt response")
        if row["outcome"] not in ("success", "failure"):
            raise ValueError("invalid Core outcome belongs to attempt ledger")
        events = self.events(a)
        if not events or events[-1]["kind"] != "ADMITTED" or row["reset_index"] != events[-1]["data"]["reset_index"]:
            raise ValueError("no admitted scene for Core outcome")
        from cosmos_framework.evaluation.robocasa_v2.evidence import validate

        staging = safe(staging)
        work = self._ledger(*key) / "work" / a["attempt_id"]
        if staging != work or not staging.is_dir():
            raise ValueError("owned attempt staging required")
        self._validate_sampling(a, row)
        inventory = validate(staging, row)
        save(staging / "core_inventory.json", {"row_digest": digest(row), "files": inventory})
        save(staging / "result.json", row)
        sync_dir(staging)
        target = self._slot(*key)
        if any(target.iterdir()):
            raise ValueError("immutable/ambiguous existing public slot")
        self.fault("before_core_commit")
        os.replace(staging, target)  # One atomic publication of result AND Core.
        sync_dir(target.parent)
        self.fault("after_core_commit")
        self.active_attempts.pop(key)
        self._rowcache.pop(key, None)

    def recovery_error(self, task, stream_id, stage, message):
        """Coordinator-only orphan recovery; scope to an existing old stream."""
        self._lease()
        if task in self.active_tasks:
            raise ValueError("cannot recover faults into an active new stream")
        stream = self._stream(task, stream_id)
        return self._record_fault(stream, stage, message, None)

    def record_process_error(self, task, stage, message, *, attempt=None):
        self._lease()
        stream = self.active_tasks.get(task)
        if stream is None:
            raise ValueError("owned stream required for process fault")
        return self._record_fault(stream, stage, message, attempt)

    def _record_fault(self, stream, stage, message, attempt):
        task = stream["task"]
        if not all(isinstance(x, str) and x for x in (stage, message)):
            raise ValueError("typed process fault required")
        if attempt is not None and self._attempt(attempt)["stream_id"] != stream["stream_id"]:
            raise ValueError("foreign process fault scope")
        value = {
            "id": uuid.uuid4().hex,
            "task": task,
            "stream_id": stream["stream_id"],
            "worker_id": stream["worker_id"],
            "attempt": attempt,
            "stage": stage,
            "message": message[:2048],
            "time_ns": time.time_ns(),
            "core_committed_at_fault": self._result(task, attempt["episode"]) is not None if attempt else False,
        }
        save(self._internal(task) / "errors" / f"{value['id']}.json", value)
        return value

    def certify_recovery(self, task, old_stream_id, probe):
        self._lease()
        old = self._stream(task, old_stream_id)
        cleanup = self._cleaned(task, old_stream_id)
        new = self._healthy_stream(task)
        if cleanup is None or old == new or cleanup["time_ns"] >= new["time_ns"]:
            raise ValueError("cleanup then independent fresh stream required")
        proof = probe(dict(new))
        if (
            not isinstance(proof, dict)
            or set(proof) != {"probe_id", "healthy"}
            or proof["healthy"] is not True
            or not proof["probe_id"]
        ):
            raise ValueError("recovery health proof required")
        value = {
            "old": old,
            "new": new,
            "cleanup": cleanup,
            "proof": proof,
            "id": uuid.uuid4().hex,
            "time_ns": time.time_ns(),
        }
        save(self._internal(task) / "recovery" / f"{value['id']}.json", value)
        return value

    def resolve_error(self, task, error_id, certificate, *, replacement=None):
        self._lease()
        identifier(error_id)
        error = load(self._internal(task) / "errors" / f"{error_id}.json")
        cert = load(self._internal(task) / "recovery" / f"{identifier(certificate['id'])}.json")
        if certificate != cert or cert["old"]["stream_id"] != error["stream_id"] or cert["old"]["task"] != task:
            raise ValueError("recovery certificate does not cover fault")
        if cert["time_ns"] < error["time_ns"] or self._cleaned(task, error["stream_id"]) != cert["cleanup"]:
            raise ValueError("recovery proof predates fault/invalid cleanup")
        if replacement is not None:
            valid = self._result(task, replacement["episode"])
            if (
                valid != replacement
                or error["attempt"] is None
                or replacement["episode"] != error["attempt"]["episode"]
                or replacement["attempt_id"] == error["attempt"]["attempt_id"]
                or replacement["stream_id"] != cert["new"]["stream_id"]
            ):
                raise ValueError("retry must replace only same missing logical Episode")
            mode = "valid_retry_after_cleanup"
        else:
            if error["attempt"] is not None and not error["core_committed_at_fault"]:
                raise ValueError("uncompleted failed Episode needs valid retry, not health-only resolution")
            mode = "execution_only_after_committed_result_or_worker_fault"
        value = {
            "error": error,
            "certificate": cert,
            "replacement": replacement,
            "mode": mode,
            "time_ns": time.time_ns(),
        }
        save(self._internal(task) / "resolutions" / f"{error_id}.json", value)
        return value

    def _validate_resolution(self, task, error, value):
        if (
            not isinstance(value, dict)
            or set(value) != {"error", "certificate", "replacement", "mode", "time_ns"}
            or value["error"] != error
        ):
            raise ValueError("fault resolution identity/schema mismatch")
        cert = value["certificate"]
        if (
            not isinstance(cert, dict)
            or cert != load(self._internal(task) / "recovery" / f"{identifier(cert['id'])}.json")
            or cert["old"] != self._stream(task, error["stream_id"])
            or cert["new"] != self._stream(task, cert["new"]["stream_id"])
            or cert["old"]["stream_id"] == cert["new"]["stream_id"]
            or cert["cleanup"] != self._cleaned(task, error["stream_id"])
            or cert["cleanup"]["time_ns"] >= cert["new"]["time_ns"]
            or cert["time_ns"] < error["time_ns"]
            or value["time_ns"] < cert["time_ns"]
            or cert["proof"]["healthy"] is not True
        ):
            raise ValueError("fault resolution recovery proof mismatch")
        replacement = value["replacement"]
        if replacement is None:
            if value["mode"] != "execution_only_after_committed_result_or_worker_fault" or (
                error["attempt"] is not None and not error["core_committed_at_fault"]
            ):
                raise ValueError("failed missing Core requires valid retry resolution")
        elif (
            value["mode"] != "valid_retry_after_cleanup"
            or error["attempt"] is None
            or replacement != self._result(task, error["attempt"]["episode"])
            or replacement["task"] != task
            or replacement["episode"] != error["attempt"]["episode"]
            or replacement["attempt_id"] == error["attempt"]["attempt_id"]
            or replacement["stream_id"] != cert["new"]["stream_id"]
        ):
            raise ValueError("fault resolution replacement mismatch")

    def errors(self, task):
        values = []
        for e in records(self._internal(task) / "errors"):
            stream = self._stream(task, e["stream_id"])
            identifier(e["id"])
            if (
                set(e)
                != {
                    "id",
                    "task",
                    "stream_id",
                    "worker_id",
                    "attempt",
                    "stage",
                    "message",
                    "time_ns",
                    "core_committed_at_fault",
                }
                or e["task"] != task
                or e["worker_id"] != stream["worker_id"]
                or type(e["core_committed_at_fault"]) is not bool
                or e != load(self._internal(task) / "errors" / f"{e['id']}.json")
            ):
                raise ValueError("invalid process fault scope/file")
            if e["attempt"] is not None:
                a = self._attempt(e["attempt"])
                if a["stream_id"] != stream["stream_id"]:
                    raise ValueError("fault attempt scope mismatch")
            resolution = safe(self._internal(task) / "resolutions" / f"{e['id']}.json")
            r = load(resolution) if resolution.exists() else None
            if r is not None:
                self._validate_resolution(task, e, r)
            values.append({**e, "resolution": r})
        return values

    def summary(self):
        expected = self.manifest["episodes_per_task"]
        for path in (self.root / "tasks").iterdir():
            safe(path)
            if path.name not in self.manifest["tasks"] or not path.is_dir():
                raise ValueError("unplanned Task artifacts")
        table, indexed, errors = {}, [], []
        for task in self.manifest["tasks"]:
            self.rows(task)  # Detect unexpected/out-of-range public artifacts.
            slots = [self.slot(task, i) for i in range(expected)]
            counts = {
                k: sum(s["status"] == k for s in slots)
                for k in (
                    "success",
                    "failure",
                    "invalid",
                    "interrupted",
                    "pending",
                    "skipped_init_error",
                    "scene_admission_exhausted",
                )
            }
            valid = counts["success"] + counts["failure"]
            faults = self.errors(task)
            errors.extend(
                {
                    "id": e["id"],
                    "task": task,
                    "stream_id": e["stream_id"],
                    "episode": e["attempt"]["episode"] if e["attempt"] else None,
                    "stage": e["stage"],
                    "message": e["message"],
                    "time_ns": e["time_ns"],
                    "resolution": e["resolution"]["mode"] if e["resolution"] else None,
                }
                for e in faults
            )
            samples = [e for i in range(expected) for a in self._attempts(task, i) for e in self.events(a)]
            skipped = [
                {
                    "task": task,
                    "episode": i,
                    "status": s["status"],
                    "initialization": s,
                    "error_diagnostics": [
                        {
                            "stage": e["data"]["stage"],
                            "type": e["data"]["type"],
                            "message": e["data"]["message"],
                            "time_ns": e["time_ns"],
                            "stream_id": e["attempt"]["stream_id"],
                            "seed": e["attempt"]["seed"],
                            "attempt_id": e["attempt"]["attempt_id"],
                            "sequence": e["sequence"],
                        }
                        for e in samples
                        if e["attempt"]["episode"] == i and e["kind"] == "INIT_ERROR"
                    ],
                }
                for i, s in enumerate(slots)
                if s["status"] in ("skipped_init_error", "scene_admission_exhausted")
            ]
            unresolved = sum(e["resolution"] is None for e in faults)
            open_streams = [s["stream_id"] for s in self._streams(task) if not self._cleaned(task, s["stream_id"])]
            table[task] = {
                **counts,
                "valid": valid,
                "expected": expected,
                "sr": counts["success"] / expected if valid == expected else None,
                "partial_sr_valid_only": counts["success"] / valid if valid else None,
                "unresolved_process_errors": unresolved,
                "open_worker_streams": open_streams,
                "sampling_streams": [
                    {
                        "stream_id": s["stream_id"],
                        "worker_id": s["worker_id"],
                        "logical_stream_seed": s["seed"],
                        "kind": s["kind"],
                    }
                    for s in self._streams(task)
                ],
                "clean_execution": unresolved == 0 and not open_streams,
                "results_qualified": valid == expected and unresolved == 0 and not open_streams,
                "missing_blocked_ids": skipped,
                "admission_rejections": sum(e["kind"] == "DEGENERATE_INITIAL_SUCCESS" for e in samples),
                "admission_samples": sum(e["kind"] == "INIT_BEGIN" for e in samples),
                "initialization_errors": sum(e["kind"] in ("INIT_ERROR", "UNSAFE_INIT") for e in samples),
            }
            for i, s in enumerate(slots):
                indexed.append(
                    s["result"] or {"task": task, "episode": i, "outcome": s["status"], "prototype_only": False}
                )
        totals = {
            k: sum(v[k] for v in table.values())
            for k in (
                "success",
                "failure",
                "invalid",
                "interrupted",
                "pending",
                "skipped_init_error",
                "scene_admission_exhausted",
                "valid",
                "expected",
            )
        }
        complete = totals["valid"] == totals["expected"]
        unresolved = sum(e["resolution"] is None for e in errors)
        clean = not unresolved and all(t["clean_execution"] for t in table.values())
        indexed.sort(key=lambda r: (r["episode"], self.manifest["task_order"].index(r["task"])))
        return {
            "prototype_only": False,
            "manifest_digest": self.manifest["digest"],
            "status": "complete"
            if complete and clean
            else "complete_with_process_errors"
            if complete and unresolved
            else "complete_awaiting_worker_cleanup"
            if complete
            else "partial",
            **totals,
            "sr": totals["success"] / totals["expected"] if complete else None,
            "partial_sr_valid_only": totals["success"] / totals["valid"] if totals["valid"] else None,
            "results_qualified": complete and clean,
            "clean_execution": clean,
            "process_errors": errors,
            "unresolved_process_errors": unresolved,
            "resolved_process_errors": len(errors) - unresolved,
            "tasks": table,
        }, indexed

    @timed("rebuild_views")
    def publish_views(self):
        self._lease()
        report, rows = self.summary()
        for task, info in report["tasks"].items():
            write_report(self._task(task) / "summary.json", info)
        target = safe(self.root / "episodes.jsonl")
        temporary = self.root / f".index-{uuid.uuid4().hex}"
        try:
            with temporary.open("xb") as stream:
                for row in rows:
                    stream.write(canonical(row) + b"\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
            sync_dir(self.root)
        finally:
            temporary.unlink(missing_ok=True)
        write_report(self.root / "summary.json", report)
        return report
