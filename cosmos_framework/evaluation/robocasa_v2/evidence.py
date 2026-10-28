"""Native Core evidence contract. Media is deliberately absent from the inventory."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from pathlib import Path

FILES = ("actions.jsonl", "predictions.jsonl", "local_memory.json", "scene_provenance.json")


def finite_vector(value, width):
    return (
        isinstance(value, list)
        and len(value) == width
        and all(type(x) in (int, float) and math.isfinite(x) for x in value)
    )


def read_lines(path, count):
    from cosmos_framework.evaluation.robocasa_v2.store import safe, witness

    path = safe(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    values = []
    with os.fdopen(fd, "rb") as f:
        before = os.fstat(f.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size > 32 * 1024 * 1024:
            raise ValueError("ordinary bounded Core evidence required")
        while line := f.readline(256 * 1024 + 1):
            if len(line) > 256 * 1024 or not line.endswith(b"\n"):
                raise ValueError("truncated/oversized Core event")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("Core event must be an object")
            values.append(value)
            if len(values) > count:
                raise ValueError("too many Core events")
        after = os.fstat(f.fileno())

    def stamp(s):
        return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns

    if stamp(before) != stamp(after) or stamp(after) != witness(path):
        raise ValueError("Core evidence changed during read")
    return values


def validate(directory, row):
    from cosmos_framework.evaluation.robocasa_v2.store import read

    directory = Path(directory)
    actions = read_lines(directory / "actions.jsonl", row["steps"])
    predictions = read_lines(directory / "predictions.jsonl", row["replans"])
    if len(actions) != row["steps"] or len(predictions) != row["replans"] or not predictions:
        raise ValueError("Core chronology counts mismatch")
    session = None
    for i, p in enumerate(predictions):
        status = p.get("local_memory", {})
        if p.get("replan") != i or p.get("source_step") != i * 16 or not isinstance(status, dict):
            raise ValueError("prediction frontier mismatch")
        if (
            not isinstance(p.get("action"), list)
            or len(p["action"]) != 16
            or not all(finite_vector(x, 15) for x in p["action"])
        ):
            raise ValueError("prediction raw15 mismatch")
        if status.get("consumer_step") != p["source_step"] or not isinstance(status.get("session_id"), str):
            raise ValueError("Local acknowledgement mismatch")
        session = session or status["session_id"]
        if session != status["session_id"]:
            raise ValueError("cross-Episode Local session")
    for i, a in enumerate(actions):
        if (
            a.get("source_step") != i
            or a.get("evidence_completed") is not True
            or not all(
                finite_vector(a.get(k), n)
                for k, n in [("predicted_raw15", 15), ("canonical_executed_raw15", 15), ("submitted_env12", 12)]
            )
        ):
            raise ValueError("completed raw15/env12 action evidence missing")
        if a["predicted_raw15"] != predictions[i // 16]["action"][i % 16]:
            raise ValueError("action does not match prediction")
    cleanup = read(directory / "local_memory.json")
    if (
        cleanup.get("session_id") != session
        or cleanup.get("reset_ok") is not True
        or cleanup.get("cleanup_error") is not None
        or cleanup.get("rollout_error") is not None
    ):
        raise ValueError("Local reset acknowledgement missing")
    scene = read(directory / "scene_provenance.json")
    for k in ("task", "episode", "attempt_id", "stream_id", "seed", "reset_index", "manifest_digest"):
        if scene.get(k) != row[k]:
            raise ValueError("scene source identity mismatch")
    if scene.get("initial_success") is not False:
        raise ValueError("scene not admitted")
    result = {}
    for name in FILES:
        path = directory / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("ordinary Core evidence required")
        raw = path.read_bytes()
        result[name] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
        with path.open("rb") as f:
            os.fsync(f.fileno())
    return result
