"""Environment-authoritative outcomes and frozen production new-run semantics."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import sys
from dataclasses import dataclass, field

SCHEMA = "psm_robocasa_eval_v2"
SAMPLING = "dynamic_slot_environment_seed_then_continuous_reset_v3"
RESUME = "retry_missing_episode_no_scene_reproduction_v2"
_SHA = re.compile(r"[0-9a-f]{64}\Z")


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def integer(value, minimum, maximum, name):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"invalid {name}")
    return value


def build_manifest(*, tasks, episodes, seed, horizons, experiment, formal=False, execution=None):
    tasks = list(tasks)
    if (
        not tasks
        or len(tasks) > 18
        or len(set(tasks)) != len(tasks)
        or any(not isinstance(t, str) or not t.isidentifier() for t in tasks)
    ):
        raise ValueError("distinct Task names required")
    integer(episodes, 1, 50, "episodes")
    integer(seed, 0, 2**32 - 1, "seed")
    if formal and (len(tasks) != 18 or episodes != 50):
        raise ValueError("formal evaluation requires exactly18Task×50")
    if set(horizons) != set(tasks):
        raise ValueError("missing official per-task horizons")
    for value in horizons.values():
        integer(value, 1, 100000, "horizon")
    if set(experiment) != {"model_digest", "config_sha256", "runtime_digest", "root_sha", "child_sha", "task_metadata"}:
        raise ValueError("complete experiment identity required")
    for key in ("model_digest", "config_sha256", "runtime_digest"):
        if not isinstance(experiment[key], str) or not _SHA.fullmatch(experiment[key]):
            raise ValueError("invalid experiment digest")
    for key in ("root_sha", "child_sha"):
        if not isinstance(experiment[key], str) or not re.fullmatch(r"[0-9a-f]{40}", experiment[key]):
            raise ValueError("invalid source SHA")
    if set(experiment["task_metadata"]) != set(tasks) or any(
        not isinstance(v, str) or not _SHA.fullmatch(v) for v in experiment["task_metadata"].values()
    ):
        raise ValueError("missing Task metadata digest")
    body = {
        "schema": SCHEMA,
        "prototype_only": False,
        "tasks": tasks,
        "task_order": tasks,
        "episodes_per_task": episodes,
        "seed": seed,
        "horizons": horizons,
        "experiment": experiment,
        "scope": "formal900" if formal else "subset",
        "protocol": {
            "sampling": SAMPLING,
            "resource_architecture": "one_reusable_sim_slot_per_gpu_phase_a",
            "seed_field": "logical_task_stream; actual environment seed in sampling provenance",
            "schedule": "episode_major_frozen_order_round_barrier_v1",
            "resume": RESUME,
            "success_latch": 1,
            "scene_admission": "reject_initial_success_bounded_v1",
            "admission_limit": 32,
            "consecutive_init_errors_to_skip": 4,
            "skipped_retry": "explicit_owner_authorization_only",
            "raw_action_dim": 15,
            "state_dim": 15,
            "action_chunk_size": 16,
            "observation_frames": 17,
            "fps": 20,
            "camera": "left_wrist",
            "local_memory": "required",
            "success_authority": "env._check_success",
            "media_participates_in_SR": False,
        },
    }
    if execution is not None:
        if (
            not isinstance(execution, dict)
            or set(execution)
            != {"profile", "gpu_ids", "sims_per_gpu", "max_infer_batch_size", "batch_wait_ms", "noise"}
            or execution["profile"] != "phase_b_native_batch_v1"
            or execution["noise"] != "literal_cfg_seed_per_sample_v1"
        ):
            raise ValueError("exact native batch execution identity required")
        integer(execution["sims_per_gpu"], 1, 8, "M")
        integer(execution["max_infer_batch_size"], 1, 8, "B")
        integer(execution["batch_wait_ms"], 1, 1000, "batch_wait_ms")
        gpus = execution["gpu_ids"]
        if (
            not isinstance(gpus, list)
            or not gpus
            or len(set(gpus)) != len(gpus)
            or any(type(g) is not int or g < 0 for g in gpus)
        ):
            raise ValueError("distinct physical GPUs required")
        body["execution"] = execution
        body["protocol"]["resource_architecture"] = "multi_reusable_sim_slots_one_native_model_owner_per_gpu_phase_b"
    return json.loads(canonical({**body, "digest": digest(body)}))


def validate_manifest(value):
    if not isinstance(value, dict) or value.get("schema") != SCHEMA or value.get("prototype_only") is not False:
        raise ValueError("fresh Evaluation V2 namespace required; legacy ingestion unsupported")
    expected = build_manifest(
        tasks=value["tasks"],
        episodes=value["episodes_per_task"],
        seed=value["seed"],
        horizons=value["horizons"],
        experiment=value["experiment"],
        formal=value["scope"] == "formal900",
        execution=value.get("execution"),
    )
    if value != expected:
        raise ValueError("invalid immutable manifest or protocol")
    return value


@dataclass
class EnvironmentOutcome:
    """Receives the environment predicate AFTER a completed action, never reward.

    The native adapter supplies observations from the original rollout;
    this module has no model imports.
    """

    horizon: int
    checks: list[bool] = field(default_factory=list)

    def __post_init__(self):
        integer(self.horizon, 1, 100000, "horizon")

    def check_after_step(self, env):
        if len(self.checks) >= self.horizon or any(self.checks):
            raise ValueError("success latch/step boundary exceeded")
        value = env._check_success()  # Errors propagate: never turn an exception into policy failure.
        if (
            type(value) is not bool
            and not (type(value).__module__ == "numpy" and type(value).__name__ == "bool")
            and not (type(value).__module__ == "numpy" and type(value).__name__ == "bool_")
        ):
            raise ValueError("environment success predicate must return a boolean")
        self.checks.append(bool(value))
        return self.checks[-1]

    def finish(self, *, normal, error=None, cleanup_ok=True):
        success = any(self.checks)
        if type(normal) is not bool or type(cleanup_ok) is not bool:
            raise ValueError("typed completion/cleanup required")
        if error is not None and (not isinstance(error, str) or not error):
            raise ValueError("typed runtime error required")
        invalid = error or ("LOCAL_CLEANUP_FAILED" if not cleanup_ok else None)
        if not normal:
            invalid = invalid or "INTERRUPTED"
        if not self.checks or (not success and len(self.checks) != self.horizon):
            invalid = invalid or "INCOMPLETE_ENVIRONMENT_EPISODE"
        return {
            "outcome": "invalid" if invalid else "success" if success else "failure",
            "environment_success": success,
            "success_checks": list(self.checks),
            "steps": len(self.checks),
            "error": invalid,
            "cleanup_ok": cleanup_ok,
            "completion": "invalid" if invalid else "success_latch" if success else "official_horizon",
        }


def result_record(
    manifest,
    *,
    task,
    episode,
    generation,
    stream_id,
    observation,
    instruction,
    replans,
    elapsed_seconds,
    attempt_id="0" * 32,
    worker_id="0" * 32,
    seed=None,
    reset_index=None,
    sampling=None,
):
    validate_manifest(manifest)
    if task not in manifest["tasks"]:
        raise ValueError("unplanned task")
    integer(episode, 0, manifest["episodes_per_task"] - 1, "episode")
    integer(generation, 0, 100000, "generation")
    integer(replans, 0, 100000, "replans")
    if not isinstance(instruction, str) or len(instruction) > 16384:
        raise ValueError("actual language instruction required")
    if not isinstance(stream_id, str) or not re.fullmatch(r"[0-9a-f]{32}", stream_id):
        raise ValueError("stream identity required")
    if type(elapsed_seconds) not in (int, float) or not math.isfinite(elapsed_seconds) or elapsed_seconds < 0:
        raise ValueError("invalid duration")
    row = {
        "schema": SCHEMA,
        "prototype_only": False,
        "manifest_digest": manifest["digest"],
        "task": task,
        "episode": episode,
        "generation": generation,
        "stream_id": stream_id,
        "seed": manifest["seed"] if seed is None else seed,
        "sampling_mode": SAMPLING,
        "reset_index": episode if reset_index is None else reset_index,
        "attempt_id": attempt_id,
        "worker_id": worker_id,
        "instruction": instruction,
        "replans": replans,
        "elapsed_seconds": elapsed_seconds,
        **observation,
    }
    if sampling is not None:
        row["sampling"] = sampling
    validate_result(manifest, row)
    return row


def validate_result(manifest, row):
    base_keys = {
        "schema",
        "prototype_only",
        "manifest_digest",
        "task",
        "episode",
        "generation",
        "attempt_id",
        "worker_id",
        "stream_id",
        "seed",
        "sampling_mode",
        "reset_index",
        "instruction",
        "replans",
        "elapsed_seconds",
        "outcome",
        "environment_success",
        "success_checks",
        "steps",
        "error",
        "cleanup_ok",
        "completion",
    }
    if (
        not isinstance(row, dict)
        or set(row) not in (base_keys, base_keys | {"sampling"})
        or row["schema"] != SCHEMA
        or row["prototype_only"] is not False
        or row["manifest_digest"] != manifest["digest"]
        or row["task"] not in manifest["tasks"]
    ):
        raise ValueError("invalid Episode identity/schema")
    if "sampling" in row:
        sample = row["sampling"]
        if (
            not isinstance(sample, dict)
            or set(sample)
            != {"environment_id", "actual_seed", "environment_reset_index", "physical_slot_id", "protocol"}
            or sample["protocol"] != SAMPLING
        ):
            raise ValueError("invalid native environment sampling provenance")
        for name in ("environment_id", "physical_slot_id"):
            if not isinstance(sample[name], str) or not re.fullmatch(r"[0-9a-f]{32}", sample[name]):
                raise ValueError("invalid physical environment/Slot identity")
        integer(sample["actual_seed"], 0, 2**32 - 1, "environment seed")
        integer(sample["environment_reset_index"], 0, 100000, "environment reset index")
    integer(row["episode"], 0, manifest["episodes_per_task"] - 1, "episode")
    integer(row["generation"], 0, 100000, "generation")
    integer(row["replans"], 0, 100000, "replans")
    if (
        row["sampling_mode"] != SAMPLING
        or type(row["seed"]) is not int
        or not 0 <= row["seed"] < 2**32
        or type(row["reset_index"]) is not int
        or not 0 <= row["reset_index"] <= 100000
    ):
        raise ValueError("scene stream/seed mismatch")
    for key in ("stream_id", "attempt_id", "worker_id"):
        if not isinstance(row[key], str) or not re.fullmatch(r"[0-9a-f]{32}", row[key]):
            raise ValueError("bad attempt/worker/stream identity")
    if (
        not isinstance(row["instruction"], str)
        or len(row["instruction"]) > 16384
        or type(row["elapsed_seconds"]) not in (int, float)
        or not math.isfinite(row["elapsed_seconds"])
        or row["elapsed_seconds"] < 0
    ):
        raise ValueError("invalid Episode metadata")
    checks = row["success_checks"]
    if (
        not isinstance(checks, list)
        or any(type(value) is not bool for value in checks)
        or len(checks) > manifest["horizons"][row["task"]]
        or any(checks[:-1])
    ):
        raise ValueError("invalid success-latch observations")
    integer(row["steps"], 0, manifest["horizons"][row["task"]], "steps")
    if row["steps"] != len(checks):
        raise ValueError("incomplete environment observation")
    recomputed = EnvironmentOutcome(manifest["horizons"][row["task"]], checks).finish(
        normal=row["completion"] != "invalid", error=row["error"], cleanup_ok=row["cleanup_ok"]
    )
    if any(type(row[key]) is not type(value) or row[key] != value for key, value in recomputed.items()):
        raise ValueError("outcome contradicts environment/termination authority")
    return row


def try_media(writer):
    """Diagnostic only. Caller retains one original file; no decode/export here."""
    before = random.getstate()
    numpy = sys.modules.get("numpy")
    torch = sys.modules.get("torch")
    numpy_state = numpy.random.get_state() if numpy is not None else None
    torch_state = torch.get_rng_state() if torch is not None else None
    try:
        writer()
    except Exception as error:
        return {"status": "error", "error": f"{type(error).__name__}: {error}"}
    finally:
        random.setstate(before)
        if numpy is not None:
            numpy.random.set_state(numpy_state)
        if torch is not None:
            torch.set_rng_state(torch_state)
    return {"status": "saved", "error": None}
