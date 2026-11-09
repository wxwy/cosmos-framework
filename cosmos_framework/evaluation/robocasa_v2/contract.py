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
ROUND_SCHEDULE = "episode_major_frozen_order_round_barrier_v1"
STREAMING_SCHEDULE = "streaming_task_affinity_v2"
STREAMING_SAMPLING = "independent_slot_environment_seed_then_continuous_reset_v4"
STREAMING_RESOURCE_FIELDS = frozenset(
    (
        "max_owned_gpu_bytes",
        "sim_startup_rss_bytes",
        "sim_startup_gpu_bytes",
        "server_startup_rss_bytes",
        "server_startup_gpu_bytes",
        "initialization_concurrency",
    )
)
VAE_RESOURCE_FIELDS = STREAMING_RESOURCE_FIELDS | {"vae_startup_rss_bytes", "vae_startup_gpu_bytes"}
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


def streaming(manifest):
    return manifest["protocol"]["schedule"] == STREAMING_SCHEDULE


def validate_streaming_resources(value):
    if not isinstance(value, dict) or set(value) != STREAMING_RESOURCE_FIELDS:
        raise ValueError("explicit complete streaming_resources budget required")
    for key, entry in value.items():
        integer(
            entry,
            0 if key == "sim_startup_gpu_bytes" else 1,
            32 if key == "initialization_concurrency" else 2**63 - 1,
            key,
        )
    if any(value[k] > value["max_owned_gpu_bytes"] for k in ("server_startup_gpu_bytes", "sim_startup_gpu_bytes")):
        raise ValueError("startup GPU reservation exceeds owned GPU budget")
    return value


def validate_single_latency(value, *, local_memory):
    """New opt-in identity only; historical execution dictionaries stay exact."""
    if type(value) is not dict or set(value) != {"version", "arm", "history_source", "worker"}:
        raise ValueError("exact single-latency experiment identity required")
    if value["version"] != "stage3a_single_v1" or value["arm"] not in ("A", "B", "C"):
        raise ValueError("explicit Stage3A A/B/C arm required")
    if local_memory != ("off" if value["arm"] == "A" else "required"):
        raise ValueError("single-latency arm differs from Local-TTT mode")
    if value["arm"] != "C":
        if value["history_source"] != "server_vae" or value["worker"] is not None:
            raise ValueError("A/B must not create a historical VAE Worker")
        return value
    worker = value["worker"]
    fields = {
        "vae_weights",
        "vae_weights_sha256",
        "vae_config",
        "vae_config_sha256",
        "egl_vendor_json",
        "egl_vendor_sha256",
        "bmax",
        "wait_ms",
        "trust_model",
        "resources",
        "max_rss_bytes",
    }
    if value["history_source"] != "sim_shared_vae" or type(worker) is not dict or set(worker) != fields:
        raise ValueError("complete Stage3A Worker identity required")
    if (
        worker["bmax"] != 1
        or type(worker["bmax"]) is not int
        or worker["wait_ms"] != 0
        or type(worker["wait_ms"]) is not int
    ):
        raise ValueError("B4/B8 feature Action Path is disabled; require B1/wait0")
    if worker["trust_model"] != "registry_same_trust_domain":
        raise ValueError("explicit controlled Actor/Owner trust domain required")
    for name in ("vae_weights", "vae_config", "egl_vendor_json"):
        if type(worker[name]) is not str or not worker[name].startswith("/") or "\0" in worker[name]:
            raise ValueError("absolute frozen Worker input paths required")
    for name in ("vae_weights_sha256", "vae_config_sha256", "egl_vendor_sha256"):
        if type(worker[name]) is not str or not _SHA.fullmatch(worker[name]):
            raise ValueError("frozen Worker input SHA256 required")
    budget = worker["resources"]
    if type(budget) is not dict or set(budget) != VAE_RESOURCE_FIELDS:
        raise ValueError("complete Worker/Server/Sim startup resources required")
    validate_streaming_resources({key: budget[key] for key in STREAMING_RESOURCE_FIELDS})
    for name in ("vae_startup_rss_bytes", "vae_startup_gpu_bytes"):
        integer(budget[name], 1, 2**63 - 1, name)
    if budget["vae_startup_gpu_bytes"] > budget["max_owned_gpu_bytes"]:
        raise ValueError("Worker GPU reservation exceeds owned GPU budget")
    integer(worker["max_rss_bytes"], 1, 2**63 - 1, "Worker owned RSS budget")
    if any(
        budget[name] > worker["max_rss_bytes"]
        for name in ("sim_startup_rss_bytes", "server_startup_rss_bytes", "vae_startup_rss_bytes")
    ):
        raise ValueError("startup RSS reservation exceeds owned RSS budget")
    return value


def validate_shared_vae(value, *, local_memory):
    """History-source identity independent of optional single-instance measurements."""
    if (
        type(value) is not dict
        or set(value) != {"version", "history_source", "worker"}
        or value["version"] != "sim_shared_vae_b1_v1"
    ):
        raise ValueError("exact shared VAE history-source identity required")
    validate_single_latency(
        {
            "version": "stage3a_single_v1",
            "arm": "C",
            "history_source": value["history_source"],
            "worker": value["worker"],
        },
        local_memory=local_memory,
    )
    return value


def build_manifest(
    *,
    tasks,
    episodes,
    seed,
    horizons,
    experiment,
    formal=False,
    execution=None,
    local_memory="required",
    rendering=None,
):
    if local_memory not in ("off", "required"):
        raise ValueError("explicit off|required Local-TTT mode required")
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
            "schedule": ROUND_SCHEDULE,
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
            "local_memory": local_memory,
            "success_authority": "env._check_success",
            "media_participates_in_SR": False,
        },
    }
    if execution is not None:
        is_streaming = isinstance(execution, dict) and execution.get("scheduling") == STREAMING_SCHEDULE
        extra = {"scheduling", "resources", "max_rss_bytes"} if is_streaming else set()
        single = execution.get("single_latency") if isinstance(execution, dict) else None
        if single is not None:
            validate_single_latency(single, local_memory=local_memory)
            extra |= {"single_latency"}
        shared = execution.get("shared_vae") if isinstance(execution, dict) else None
        if isinstance(execution, dict) and "local_ttt_history_k" in execution:
            history_k = execution["local_ttt_history_k"]
            if (
                type(history_k) is not int
                or history_k not in (1, 2, 4, 8)
                or local_memory != "required"
                or not (shared or single and single["arm"] == "C")
            ):
                raise ValueError("sparse K1/2/4/8 requires frozen shared Local-TTT execution")
            extra.add("local_ttt_history_k")
        if shared is not None:
            if single is not None:
                raise ValueError("single measurements retain their original frozen identity")
            validate_shared_vae(shared, local_memory=local_memory)
            extra |= {"shared_vae"}
        if (
            not isinstance(execution, dict)
            or set(execution)
            != {"profile", "gpu_ids", "sims_per_gpu", "max_infer_batch_size", "batch_wait_ms", "noise"} | extra
            or execution["profile"] != "phase_b_native_batch_v1"
            or execution["noise"] != "literal_cfg_seed_per_sample_v1"
        ):
            raise ValueError("exact native batch execution identity required")
        integer(execution["sims_per_gpu"], 1, 32 if is_streaming else 8, "M")
        integer(execution["max_infer_batch_size"], 1, 32 if is_streaming else 8, "B")
        integer(
            execution["batch_wait_ms"], 0 if shared or (single and single["arm"] == "C") else 1, 1000, "batch_wait_ms"
        )
        if shared and execution["max_infer_batch_size"] > 8:
            raise ValueError("shared VAE uses Policy batch1..8 independently of Worker B1")
        if single and (
            execution["sims_per_gpu"] != 1
            or execution["max_infer_batch_size"] != 1
            or type(execution["gpu_ids"]) is not list
            or len(execution["gpu_ids"]) != 1
        ):
            raise ValueError("single-latency experiment requires exactly one Server and one Sim")
        if single and single["arm"] == "C" and execution["batch_wait_ms"] != 0:
            raise ValueError("Stage3A C requires zero Policy peer wait")
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
        if is_streaming:
            validate_streaming_resources(execution["resources"])
            integer(execution["max_rss_bytes"], 1, 2**63 - 1, "max_rss_bytes")
            body["protocol"].update(schedule=STREAMING_SCHEDULE, sampling=STREAMING_SAMPLING)
    if rendering is not None:
        if (
            not isinstance(rendering, dict)
            or set(rendering) != {"backend", "gpu_ids"}
            or rendering["backend"] not in ("egl", "osmesa")
            or not isinstance(rendering["gpu_ids"], list)
            or any(type(g) is not int or g < 0 for g in rendering["gpu_ids"])
            or (rendering["backend"] == "osmesa" and rendering["gpu_ids"])
            or (rendering["backend"] == "egl" and not rendering["gpu_ids"])
            or (
                rendering["backend"] == "egl"
                and execution is not None
                and len(rendering["gpu_ids"]) != len(execution["gpu_ids"])
            )
        ):
            raise ValueError("exact render backend/physical GPU policy required")
        body["rendering"] = rendering
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
        local_memory=value["protocol"]["local_memory"],
        rendering=value.get("rendering"),
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
        "sampling_mode": manifest["protocol"]["sampling"],
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
    if streaming(manifest) and "sampling" not in row:
        raise ValueError("streaming Core requires physical Slot sampling provenance")
    if "sampling" in row:
        sample = row["sampling"]
        if (
            not isinstance(sample, dict)
            or set(sample)
            != {"environment_id", "actual_seed", "environment_reset_index", "physical_slot_id", "protocol"}
            or sample["protocol"] != manifest["protocol"]["sampling"]
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
        row["sampling_mode"] != manifest["protocol"]["sampling"]
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
