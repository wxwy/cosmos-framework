"""Corrected exact-window RoboCasa Phase5 trainer 入口。"""

from __future__ import annotations

import argparse
import dataclasses
import fcntl
import functools
import hashlib
import json
import math
import os
import pickle
import re
import subprocess
import time
import uuid
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from examples.psm_wma_robocasa_effective_config import (
    describe_effective_data_source,
    synchronize_legacy_chunk_length,
    validate_effective_snapshot,
    validate_saved_yaml,
)

if TYPE_CHECKING:
    import torch
    from cosmos_framework.checkpoint.dcp import SafeLocalDistributedCheckpointer
    from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml
    from cosmos_framework.data.generator.action.datasets.robocasa_exact_window_cache import (
        RoboCasaExactWindowCacheCatalog,
    )
    from cosmos_framework.data.generator.action.datasets.robocasa_exact_window_cached_sft import (
        get_action_robocasa_exact_window_cached_sft_dataset,
    )
    from cosmos_framework.data.generator.action.datasets.robocasa_exact_window_policy import (
        CorrectedRoboCasaPolicyContract,
    )
    from cosmos_framework.data.generator.action.datasets.robocasa_verified_index import VerifiedExactWindowIndex
    from cosmos_framework.model.generator.mot.robocasa_exact_window_local import (
        ExactWindowLocalCatalog,
        ExactWindowRankPlanner,
        ExactWindowSegmentProducer,
    )
    from cosmos_framework.trainer.local_memory_grouped import GroupedLocalMemoryTrainer, collate_grouped_native_batch
    from cosmos_framework.utils import distributed
    from cosmos_framework.utils.context_managers import model_init
    from cosmos_framework.utils.generator.optimizer import OptimizersContainer
    from cosmos_framework.utils.lazy_config import instantiate
    from examples.psm_wma_robocasa_corrected_telemetry import GroupedPlanObserver
    from examples.psm_wma_robocasa_formal_monitor import TelemetryJournal
    from examples.psm_wma_robocasa_native import RECIPE, check_droid_dcp, check_edge_checkpoint
    from omegaconf import DictConfig, OmegaConf


def _load_runtime() -> None:
    """Keep read-only namespace rejection and CLI help usable without CUDA dependencies."""
    global torch
    global DictConfig
    global OmegaConf
    global load_experiment_from_toml
    global RoboCasaExactWindowCacheCatalog
    global get_action_robocasa_exact_window_cached_sft_dataset
    global CorrectedRoboCasaPolicyContract
    global VerifiedExactWindowIndex
    global ExactWindowLocalCatalog
    global ExactWindowRankPlanner
    global ExactWindowSegmentProducer
    global GroupedLocalMemoryTrainer
    global collate_grouped_native_batch
    global distributed
    global model_init
    global OptimizersContainer
    global instantiate
    global GroupedPlanObserver
    global TelemetryJournal
    global RECIPE
    global check_droid_dcp
    global check_edge_checkpoint
    global SafeLocalDistributedCheckpointer
    global _RUNTIME_READY
    if globals().get("_RUNTIME_READY", False):
        return
    import torch
    from cosmos_framework.checkpoint.dcp import SafeLocalDistributedCheckpointer
    from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml
    from cosmos_framework.data.generator.action.datasets.robocasa_exact_window_cache import (
        RoboCasaExactWindowCacheCatalog,
    )
    from cosmos_framework.data.generator.action.datasets.robocasa_exact_window_cached_sft import (
        get_action_robocasa_exact_window_cached_sft_dataset,
    )
    from cosmos_framework.data.generator.action.datasets.robocasa_exact_window_policy import (
        CorrectedRoboCasaPolicyContract,
    )
    from cosmos_framework.data.generator.action.datasets.robocasa_verified_index import VerifiedExactWindowIndex
    from cosmos_framework.model.generator.mot.robocasa_exact_window_local import (
        ExactWindowLocalCatalog,
        ExactWindowRankPlanner,
        ExactWindowSegmentProducer,
    )
    from cosmos_framework.trainer.local_memory_grouped import GroupedLocalMemoryTrainer, collate_grouped_native_batch
    from cosmos_framework.utils import distributed
    from cosmos_framework.utils.context_managers import model_init
    from cosmos_framework.utils.generator.optimizer import OptimizersContainer
    from cosmos_framework.utils.lazy_config import instantiate
    from examples.psm_wma_robocasa_corrected_telemetry import GroupedPlanObserver
    from examples.psm_wma_robocasa_formal_monitor import TelemetryJournal
    from examples.psm_wma_robocasa_native import RECIPE, check_droid_dcp, check_edge_checkpoint
    from omegaconf import DictConfig, OmegaConf

    _RUNTIME_READY = True


GENERATION_KEYS = (
    "moe_gen",
    "time_embedder",
    "vae2llm",
    "llm2vae",
    "action2llm",
    "llm2action",
    "action_modality_embed",
)
LOCAL_KEYS = (
    "local_memory_runtime.encoder.",
    "local_memory_runtime.core.",
    "local_memory2llm.",
    "local_memory_modality_embed",
)
OPTIMIZER_KEYS = (*GENERATION_KEYS, *LOCAL_KEYS)
ACTION_KEYS = ("action2llm", "llm2action", "action_modality_embed")
# This changes the gradient accumulation/RNG graph and must not silently
# same-job resume the legacy per-index native execution checkpoint.
NATIVE_BATCH_LAYOUT = "v2_a2_member_packed_slot_major_v1"
JOB_IDENTITY = "v3_job_identity.json"
G1_JOB = "packed128_droid_fresh_g1_20261019"
G1_FROZEN_ROOT = "e0121d34592d10323ede480eb3409468a07371ba"
G1_FROZEN_CHILD = "2d128ed5a07601cdd424974f99b67832eafeeb41"
G1_RUNTIME_BASE_CHILD = "35b0ea1ccee9ee26653ed6589c484f0997d1c586"
G1_CONFIG_YAML_SHA256 = "f87e2e49c6931ce7dc32b467aa3f8026269f5ad255216e1cdd8fafb1d7e7374e"
G1_CHECKPOINT = "iter_000008900"


def _owner_g1_cross_pair_resume(args: argparse.Namespace) -> bool:
    enabled = getattr(args, "owner_authorized_g1_cross_pair_resume", False)
    if type(enabled) is not bool:
        raise ValueError("Owner G1 override must be an explicit boolean flag")
    # Owner's standing G1 instruction: Resume is permitted after every fully
    # committed same-job checkpoint, without repinning the runtime Root HEAD.
    # The historical --owner-authorized flag is accepted for compatibility but
    # no longer needed when this *exact* G1 job is resumed.
    g1_resume = args.phase == "resume" and args.job_name == G1_JOB
    if enabled and not g1_resume:
        raise ValueError("G1 cross-pair flag is only valid for this named G1 Resume")
    if g1_resume and (
        getattr(args, "no_ttt", False)
        or tuple(getattr(args, key, None) for key in ("t", "b", "ga", "k", "world_size")) != (16, 8, 2, 4, 8)
        or (args.max_iter, args.warmup, args.save_iter) != (30000, 500, 100)
        or getattr(args, "stop_after_iter", None) is not None
    ):
        raise ValueError("G1 Resume requires unchanged formal T16/B8/GA2/K4/world8 and 30000/500/100")
    return g1_resume


def _semantic_drift_paths(old: Any, new: Any, path: str = "semantic") -> list[str]:
    if type(old) is not type(new):
        return [path]
    if isinstance(old, dict):
        return [
            item
            for key in sorted(old.keys() | new.keys())
            for item in (
                [f"{path}.{key}"]
                if key not in old or key not in new
                else _semantic_drift_paths(old[key], new[key], f"{path}.{key}")
            )
        ]
    if isinstance(old, list):
        if len(old) != len(new):
            return [path]
        return [
            item
            for index, (a, b) in enumerate(zip(old, new, strict=True))
            for item in _semantic_drift_paths(a, b, f"{path}[{index}]")
        ]
    return [] if old == new else [path]


def _json_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def job_namespace(args: argparse.Namespace) -> Path:
    """Reject traversal and symlink aliases instead of resolving into an old job."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.job_name) or args.job_name in {".", ".."}:
        raise ValueError("job-name must be one ordinary path component")
    if ".." in args.output_root.parts:
        raise ValueError("output-root traversal is forbidden")
    path = args.output_root.absolute() / "psm_wma_v3" / "corrected_phase5" / args.job_name
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("job namespace symlink alias is forbidden")
    return path


def check_job_namespace(args: argparse.Namespace, semantic: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Read-only checks; a new Fresh claim is still required after distributed agreement."""
    override = _owner_g1_cross_pair_resume(args)
    no_ttt = getattr(args, "no_ttt", False)
    if no_ttt and not args.job_name.startswith(("no_ttt_", "bounded_no_ttt_")):
        raise ValueError("Matched No-TTT requires a separate no_ttt_ job namespace")
    path = job_namespace(args)
    if args.phase == "fresh":
        if path.exists():
            raise FileExistsError(f"Fresh requires a previously nonexistent job namespace: {path}")
        return None
    if args.job_name.startswith("bounded_"):
        raise ValueError("bounded diagnostic checkpoint must not be resumed as formal training")
    if (path / JOB_IDENTITY).is_symlink():
        raise ValueError("job identity symlink is forbidden")
    record = json.loads((path / JOB_IDENTITY).read_text())
    if (
        record.get("schema") != "corrected_phase5_f0_v1"
        or record.get("namespace") != str(path)
        or record.get("root") != (G1_FROZEN_ROOT if override else args.expected_root)
        or record.get("child") != (G1_FROZEN_CHILD if override else args.expected_child)
        or record.get("native_batch_layout") != NATIVE_BATCH_LAYOUT
        or record.get("diagnostic_only_do_not_resume") is not False
        or record.get("semantic_digest") != _json_digest(record.get("semantic"))
        or (record.get("semantic", {}).get("training_profile") == "matched_no_ttt_v1") != no_ttt
    ):
        raise ValueError("same-job identity Root/Child/Layout/namespace mismatch")
    if override and (
        not isinstance(record["semantic"], dict)
        or (record["semantic"].get("root"), record["semantic"].get("child")) != (G1_FROZEN_ROOT, G1_FROZEN_CHILD)
        or record.get("canonical_config_sha256", {}).get("config.yaml") != G1_CONFIG_YAML_SHA256
    ):
        raise ValueError("frozen G1 source/semantic/config authority mismatch")
    if semantic is not None:
        if override:
            # G1 Root/Child are runtime provenance, never caller-supplied
            # expected-HEAD admission keys. Real Child/Gitlink integrity and
            # observed source identity are verified in _g1_resume_compatibility().
            old = {key: value for key, value in record["semantic"].items() if key not in {"root", "child"}}
            new = {key: value for key, value in semantic.items() if key not in {"root", "child"}}
            drift = _semantic_drift_paths(old, new)
            if drift:
                raise ValueError("G1 semantic drift outside root/child: " + ", ".join(drift))
        elif record["semantic"] != semantic:
            raise ValueError("same-job training semantic identity mismatch")
    if set(record.get("canonical_config_sha256", {})) != {"config.pkl", "config.yaml"}:
        raise ValueError("canonical config witnesses missing")
    for name, digest in record["canonical_config_sha256"].items():
        item = path / name
        if item.is_symlink() or _file_sha256(item) != digest:
            raise ValueError(f"canonical config changed: {name}")
    return record


def _g1_resume_compatibility(args: argparse.Namespace, report: dict[str, Any], phase5_digest: str) -> dict[str, Any]:
    """Read frozen authorities; never rewrite identity or relax actual state restoration."""
    if not _owner_g1_cross_pair_resume(args):
        raise ValueError("G1 compatibility requires explicit Owner authorization")
    if not isinstance(phase5_digest, str) or re.fullmatch(r"[0-9a-f]{64}", phase5_digest) is None:
        raise ValueError("G1 phase5 digest must be a full SHA256")
    record = check_job_namespace(args, report["training_identity"])
    checkpoint = _resume_checkpoint(args)
    if str(checkpoint) != report["checkpoint_load_path"]:
        raise ValueError("G1 checkpoint changed since preflight")
    expected = _json_digest({"phase5_digest": phase5_digest, "training_identity": record["semantic"]})
    path = job_namespace(args)
    launches = path / "launches"
    if launches.is_symlink():
        raise ValueError("G1 launches symlink is forbidden")
    frozen_launch = None
    for item in sorted(launches.glob("*/launch.json")):
        if item.is_symlink() or item.parent.is_symlink():
            raise ValueError("G1 launch symlink is forbidden")
        prior = json.loads(item.read_text())
        if prior.get("semantic_digest") != record["semantic_digest"]:
            continue
        if prior.get("config_digest") != expected:
            raise ValueError("G1 phase5/config digest differs from historical launch")
        if prior.get("phase") == "fresh":
            for name, digest in record["canonical_config_sha256"].items():
                snapshot = item.parent / name
                if snapshot.is_symlink() or _file_sha256(snapshot) != digest:
                    raise ValueError("G1 original launch config differs from frozen canonical config")
            frozen_launch = {"path": str(item), "sha256": _file_sha256(item)}
    if frozen_launch is None:
        raise ValueError("G1 original frozen-config launch authority is missing")
    witnesses = {}
    for rank in range(args.world_size):
        sidecar = checkpoint / "dataloader" / f"rank_{rank}.pkl"
        before = _file_sha256(sidecar)
        # Same trusted local G1 pickle format already used by the unchanged DCP loader.
        with sidecar.open("rb") as stream:
            state = pickle.load(stream)
            if stream.read(1):
                raise ValueError("G1 sidecar has trailing bytes")
        if _file_sha256(sidecar) != before:
            raise ValueError("G1 sidecar changed during read-only validation")
        if (
            not isinstance(state, dict)
            or state.get("format") != "psm_v3_corrected_grouped_local_v2"
            or type(state.get("iteration")) is not int
            or state["iteration"] != int(checkpoint.name[5:])
            or state.get("config_digest") != expected
            or type(state.get("profile")) is not tuple
            or state["profile"][:2] != (rank, args.world_size)
            or not isinstance(state.get("sidecar"), dict)
            or any(
                state.get(name) != report[name]
                for name in ("cache_manifest_sha256", "cache_corpus_digest", "source_binding_digest")
            )
        ):
            raise ValueError(f"G1 rank{rank} sidecar digest/iteration/source/profile mismatch")
        for saved in state["sidecar"].values():
            if type(saved) is not tuple or len(saved) != 3 or getattr(saved[1], "config_digest", None) != expected:
                raise ValueError(f"G1 rank{rank} SegmentProvenance config digest mismatch")
        witnesses[str(rank)] = {"sha256": before, "bytes": sidecar.stat().st_size}
    lock = report["lock"]
    # Runtime Root/Child are recorded from Git, not caller-supplied SHA pins.
    # Still require the actual Child commit to match the Root Gitlink.
    if lock["child"] != lock["gitlink"] or (
        report["training_identity"].get("root"),
        report["training_identity"].get("child"),
    ) != (lock["root"], lock["child"]):
        raise ValueError("G1 runtime Child/Gitlink/provenance mismatch")
    return {
        "schema": "g1_iter8900_owner_cross_pair_resume_v1",
        "owner_override": True,
        "owner_authorization": "standing_named_g1_same_job_resume",
        "owner_flag": (
            "--owner-authorized-g1-cross-pair-resume"
            if getattr(args, "owner_authorized_g1_cross_pair_resume", False)
            else None
        ),
        "frozen_source": {"root": record["root"], "child": record["child"]},
        "runtime_source": lock,
        "frozen_semantic_digest": record["semantic_digest"],
        "runtime_semantic_digest": _json_digest(report["training_identity"]),
        "canonical_config_sha256": record["canonical_config_sha256"],
        "phase5_digest": phase5_digest,
        "config_digest": expected,
        "checkpoint": str(checkpoint),
        "iteration": int(checkpoint.name[5:]),
        "frozen_launch": frozen_launch,
        "rank_sidecars": witnesses,
        "validation": {
            "non_source_semantics": "PASS",
            "full_component_presence": "PASS",
            "all_rank_digest_iteration_source_provenance": "PASS",
            "optimizer_scheduler_rng_cursor_fast_state_restoration": "PENDING_UNCHANGED_STRICT_RUNTIME_LOAD",
        },
    }


def _identity_value(value: Any) -> Any:
    """Serialize resolved training values without process addresses or unsafe pickle loads."""
    if value is None or type(value) in (str, int, bool):
        return value
    if isinstance(value, Enum):
        return _identity_value(value.value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("nonfinite config identity")
        return value
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, functools.partial):
        return {
            "callable": _identity_value(value.func),
            "args": _identity_value(value.args),
            "kwargs": _identity_value(value.keywords),
        }
    if isinstance(value, type) or callable(value):
        if "<lambda>" in value.__qualname__ or "<locals>" in value.__qualname__:
            raise ValueError("anonymous training config callable has no stable identity")
        return f"{value.__module__}.{value.__qualname__}"
    # OmegaConf is loaded only for the full config preflight, never namespace-only checks.
    if "OmegaConf" in globals() and OmegaConf.is_config(value):
        return _identity_value(OmegaConf.to_container(value, resolve=True))
    if isinstance(value, Mapping):
        return {str(key): _identity_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_identity_value(item) for item in value]
    fields = getattr(type(value), "__attrs_attrs__", ())
    if dataclasses.is_dataclass(value):
        fields = dataclasses.fields(value)
    if fields:
        return {field.name: _identity_value(getattr(value, field.name)) for field in fields}
    if type(value).__module__ == "torch" and type(value).__name__ in {"dtype", "memory_format", "device"}:
        return str(value)
    raise TypeError(f"unsupported config identity value: {type(value)}")


def training_identity(args: argparse.Namespace, config: Any, report: dict[str, Any]) -> dict[str, Any]:
    """Bind complete model/optimizer/seed semantics; exclude phase/load path and run budget."""
    return {
        **({"training_profile": "matched_no_ttt_v1"} if getattr(args, "no_ttt", False) else {}),
        "root": report["lock"]["root"] if _owner_g1_cross_pair_resume(args) else args.expected_root,
        "child": report["lock"]["child"] if _owner_g1_cross_pair_resume(args) else args.expected_child,
        "native_batch_layout": NATIVE_BATCH_LAYOUT,
        "training": {
            name: _identity_value(getattr(config, name))
            for name in ("model", "optimizer", "scheduler", "trainer", "model_parallel", "data_setting")
        },
        "checkpoint": {
            **{name: _identity_value(getattr(config.checkpoint, name)) for name in ("save_iter", "strict_resume")},
            "safety_profile": "psm_v3_sync_local_v1",
        },
        "source_paths": {
            name: str(getattr(args, name).resolve())
            for name in ("source_root", "cache_root", "dataset_index_root", "base_checkpoint", "edge", "vae")
        },
        "authority": {
            key: report[key]
            for key in (
                "cache_manifest_sha256",
                "cache_corpus_digest",
                "source_binding_digest",
                "model_witnesses",
                "transform_rng_mode",
            )
        },
        "contract": {
            "raw_action_dim": 15,
            "state_dim": 15,
            "h_pred": 16,
            "consumer_frames": 17,
            "t": args.t,
            "b": args.b,
            "ga": args.ga,
            "k": args.k,
            "world_size": args.world_size,
        },
        "rng": {"seed_by_rank": True, "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()},
        "index_receipt_sha256": _file_sha256(args.dataset_index_root / "receipt.json"),
    }


class JobLifecycle:
    """Rank0-only writer after all ranks have agreed on read-only preflight."""

    def __init__(self, args: argparse.Namespace, report: dict[str, Any]):
        self.args, self.report = args, report
        self.path = job_namespace(args)
        self.launch_id = uuid.uuid4().hex
        self._lease = None

    def close(self) -> None:
        if self._lease is not None:
            self._lease.close()
            self._lease = None

    def write_config(self, config: Any, serializer: Any) -> None:
        semantic = self.report["training_identity"]
        if Path(config.job.path_local).absolute() != self.path:
            raise ValueError("Trainer job path differs from claimed namespace")
        check_job_namespace(self.args, semantic)
        effective_data_source = self.report["effective_data_source"]
        validate_effective_snapshot(config, effective_data_source)
        if self.args.phase == "fresh":
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # mkdir(exist_ok=False) is the cross-process compare-and-create operation.
            # A failed/interrupted claim remains partial and can never be silently reused.
            self.path.mkdir()
        # Keep the rank0 descriptor alive throughout this launch. Never unlink the
        # lock inode: a second Resume must not write alongside the first one.
        lease_path = self.path / "launch.lock"
        if lease_path.is_symlink():
            raise ValueError("launch lock symlink is forbidden")
        self._lease = lease_path.open("xb" if self.args.phase == "fresh" else "r+b")
        fcntl.flock(self._lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if self.args.phase == "resume" and str(_resume_checkpoint(self.args)) != self.report["checkpoint_load_path"]:
            raise ValueError("latest checkpoint changed between preflight and exclusive launch")
        if _owner_g1_cross_pair_resume(self.args):
            current = {**self.report, "lock": verify_root_child_lock(self.args)}
            audit = _g1_resume_compatibility(
                self.args, current, self.report["owner_cross_pair_resume"]["phase5_digest"]
            )
            if (
                audit != self.report["owner_cross_pair_resume"]
                or audit["config_digest"] != self.report["config_digest"]
            ):
                raise ValueError("G1 compatibility authority changed before exclusive launch")
        if (self.path / "launches").is_symlink():
            raise ValueError("launch snapshots symlink is forbidden")
        launch = self.path / "launches" / self.launch_id
        launch.mkdir(parents=True)
        serializer.save_pkl(config, str(launch / "config.pkl"))
        serializer.save_yaml(config, str(launch / "config.yaml"))
        # Serializers may log failures instead of raising. Require both actual outputs.
        for name in ("config.pkl", "config.yaml"):
            if not (launch / name).is_file() or (launch / name).stat().st_size == 0:
                raise RuntimeError(f"resolved config snapshot missing: {name}")
        validate_saved_yaml(launch / "config.yaml", effective_data_source)
        record = {
            "phase": self.args.phase,
            "formal_execution_until": getattr(self.args, "formal_execution_until", None),
            "execution_stop_after_iter": self.report["execution_stop_after_iter"],
            "semantic_digest": _json_digest(semantic),
            "config_digest": self.report["config_digest"],
            "effective_data_source": effective_data_source,
        }
        if _owner_g1_cross_pair_resume(self.args):
            record["owner_cross_pair_resume"] = self.report["owner_cross_pair_resume"]
        (launch / "launch.json").write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
        if self.args.phase == "fresh":
            for name in ("config.pkl", "config.yaml"):
                with (self.path / name).open("xb") as out:
                    out.write((launch / name).read_bytes())
            identity = {
                "schema": "corrected_phase5_f0_v1",
                "namespace": str(self.path),
                "root": self.args.expected_root,
                "child": self.args.expected_child,
                "native_batch_layout": NATIVE_BATCH_LAYOUT,
                "semantic": semantic,
                "semantic_digest": _json_digest(semantic),
                "diagnostic_only_do_not_resume": self.args.stop_after_iter is not None,
                "canonical_config_sha256": {
                    name: _file_sha256(self.path / name) for name in ("config.pkl", "config.yaml")
                },
            }
            with (self.path / JOB_IDENTITY).open("x") as out:
                json.dump(identity, out, sort_keys=True, indent=2)
                out.write("\n")


def _agree_preflight(args: argparse.Namespace):
    """Collect errors before any rank can claim the namespace or construct Trainer."""
    result, error = None, None
    try:
        result = preflight(args)
        result[1].validate()
        result[1].freeze()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    errors = [None] * args.world_size
    torch.distributed.all_gather_object(errors, error)
    if any(errors):
        raise RuntimeError(f"all-rank read-only preflight failed: {errors}")
    # Also reject inconsistent per-rank resolved configs before a writer is installed.
    identities = [None] * args.world_size
    identity = result[0]["config_digest"]
    if _owner_g1_cross_pair_resume(args):
        identity = _json_digest({"config_digest": identity, "audit": result[0]["owner_cross_pair_resume"]})
    torch.distributed.all_gather_object(identities, identity)
    if len(set(identities)) != 1:
        raise ValueError("all-rank training config identities disagree")
    return result


class GroupedTriggerLoader:
    """每个 optimizer iteration 精确提供 active_ga 个空触发。"""

    def __init__(
        self, max_iter: int, active_ga: int, *, on_iteration_start: Callable[[int], None] | None = None
    ) -> None:
        if any(type(value) is not int or value <= 0 for value in (max_iter, active_ga)):
            raise ValueError("max_iter/active_ga 必须是正整数")
        self.max_iter, self.active_ga, self.start = max_iter, active_ga, 0
        self.on_iteration_start = on_iteration_start

    def set_start_iteration(self, fetched: int) -> None:
        if type(fetched) is not int or not 0 <= fetched <= self.max_iter * self.active_ga:
            raise ValueError("trigger resume 偏移越界")
        self.start = fetched

    def __iter__(self):
        for fetched in range(self.start, self.max_iter * self.active_ga):
            if fetched % self.active_ga == 0 and self.on_iteration_start is not None:
                self.on_iteration_start(fetched // self.active_ga)
            yield {}

    def __len__(self) -> int:
        return self.max_iter * self.active_ga - self.start


def _selected_name(name: str) -> bool:
    if not name.startswith("net."):
        return False
    if name.startswith("net.language_model."):
        return "_moe_gen" in name
    root = name.split(".", 2)[1]
    if root in GENERATION_KEYS:
        return True
    return (
        name.startswith("net.local_memory_runtime.encoder.")
        or name.startswith("net.local_memory_runtime.core.")
        or name.startswith("net.local_memory2llm.")
        or root == "local_memory_modality_embed"
    )


def _telemetry_parameter_group(name: str) -> str | None:
    """The existing optimizer allowlist is the sole telemetry parameter authority."""
    if not _selected_name(name):
        return None
    if (
        name.startswith("net.local_memory_runtime.")
        or name.startswith("net.local_memory2llm.")
        or name.startswith("net.local_memory_modality_embed")
    ):
        return "local"
    if name.split(".", 2)[1] in ACTION_KEYS:
        return "action"
    return "generation"


def validate_optimizer_inventory(model: torch.nn.Module, optimizer: Any, *, no_ttt: bool = False) -> dict[str, int]:
    """用实际模型身份和实际 optimizer 参数集合核对精确 allowlist。"""
    named = dict(model.named_parameters())
    if no_ttt:
        from cosmos_framework.trainer.matched_no_ttt import require_no_local_model

        require_no_local_model(model)
    expected = {name for name in named if _selected_name(name) and (not no_ttt or "local_memory" not in name)}
    generation = {name for name in expected if any(key in name for key in GENERATION_KEYS)}
    local = expected - generation
    if (
        not generation
        or (not no_ttt and not local)
        or any(not any(key in name for key in LOCAL_KEYS) for name in local)
    ):
        raise ValueError("generation/Local 参数集合缺失或重叠")
    if any(parameter.requires_grad != (name in expected) for name, parameter in named.items()):
        raise ValueError("非 generation/Local 参数必须冻结，选中参数必须可训练")
    inners = optimizer.optimizers if isinstance(optimizer, OptimizersContainer) else (optimizer,)
    selected = [parameter for inner in inners for group in inner.param_groups for parameter in group["params"]]
    by_id = {id(parameter): name for name, parameter in named.items()}
    if len(selected) != len({id(parameter) for parameter in selected}) or any(id(p) not in by_id for p in selected):
        raise ValueError("optimizer 参数重复或不属于模型")
    selected_names = {by_id[id(parameter)] for parameter in selected}
    if selected_names != expected:
        raise ValueError(
            f"optimizer inventory 不匹配：缺少={sorted(expected - selected_names)[:4]}，多出={sorted(selected_names - expected)[:4]}"
        )
    return {
        "generation_tensors": len(generation),
        "local_tensors": len(local),
        "generation_params": sum(named[name].numel() for name in generation),
        "local_params": sum(named[name].numel() for name in local),
    }


def install_optimizer_inventory_check(model: torch.nn.Module, report: dict[str, Any], *, no_ttt: bool = False) -> None:
    original = model.init_optimizer_scheduler

    def checked(optimizer_config, scheduler_config):
        optimizer, scheduler = original(optimizer_config, scheduler_config)
        report["optimizer_inventory"] = validate_optimizer_inventory(model, optimizer, no_ttt=no_ttt)
        return optimizer, scheduler

    model.init_optimizer_scheduler = checked


def install_checkpoint_save_telemetry(checkpointer: Any, observer: GroupedPlanObserver) -> None:
    """Wrap DCP save for timing only; preserve its return value and exceptions."""
    original_save = checkpointer.save

    def observed_save(*args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        result = original_save(*args, **kwargs)
        iteration = kwargs.get("iteration")
        if iteration is not None:
            observer.log_checkpoint(int(iteration), (time.perf_counter() - started) * 1000.0)
        return result

    checkpointer.save = observed_save


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_witnesses(edge: Path, base_checkpoint: Path) -> dict[str, str]:
    return {
        "edge_config_sha256": _file_sha256(edge / "config.json"),
        "base_model_metadata_sha256": _file_sha256(base_checkpoint / "model/.metadata"),
    }


def config_digest(
    catalog: ExactWindowLocalCatalog,
    config: Any,
    *,
    b_stream: int,
    active_ga: int,
    witnesses: dict[str, str],
    transform_rng_mode: str = "legacy_global",
) -> str:
    if transform_rng_mode not in {"legacy_global", "consumer_keyed_v1"}:
        raise ValueError("invalid transform RNG mode")
    model = config.model.config
    no_ttt = getattr(model, "local_memory_enabled", True) is False
    local = (
        None
        if no_ttt
        else {
            key: getattr(model, f"local_memory_{key}")
            for key in ("ttt_tbptt_steps", "k_local", "ttt_dim", "fast_hidden_dim", "dim", "evidence_dim", "inner_lr")
        }
    )
    authority = {
        **({"training_profile": "matched_no_ttt_v1"} if no_ttt else {}),
        "cache_manifest_sha256": catalog.manifest_digest,
        "cache_corpus_digest": catalog.cache_corpus_digest,
        "source_binding_digest": catalog.source_binding_digest,
        "model": "Cosmos3-Edge-Policy-DROID/corrected-exact-window-v1",
        "native_batch_layout": NATIVE_BATCH_LAYOUT,
        "model_witnesses": witnesses,
        "raw_action_dim": 15,
        "state_dim": 15,
        "h_pred": 16,
        "consumer_frames": 17,
        "max_action_dim": model.max_action_dim,
        "local": local,
        "b_stream": b_stream,
        "active_ga": active_ga,
        "optimizer": {
            "keys": list(config.optimizer.keys_to_select),
            "type": config.optimizer.optimizer_type,
            "lr": float(config.optimizer.lr),
            "action_lr_multipliers": dict(config.optimizer.lr_multipliers),
            "weight_decay": float(config.optimizer.weight_decay),
        },
        "mesh": {
            "world_size": model.parallelism.data_parallel_shard_degree
            * model.parallelism.data_parallel_replicate_degree,
            "shard": model.parallelism.data_parallel_shard_degree,
            "replicate": model.parallelism.data_parallel_replicate_degree,
            "context_parallel": getattr(getattr(config, "model_parallel", None), "context_parallel_size", 1),
        },
        "schedule": {
            "max_iter": config.trainer.max_iter,
            "cycle_lengths": list(config.scheduler.cycle_lengths),
            "warm_up_steps": list(config.scheduler.warm_up_steps),
            "save_iter": config.checkpoint.save_iter,
        },
    }
    # Legacy no-prefetch identity remains byte-identical; opt-in keyed RNG
    # changes data augmentation semantics and gets an explicit new digest.
    if transform_rng_mode != "legacy_global":
        authority["transform_rng_mode"] = transform_rng_mode
    return hashlib.sha256(json.dumps(authority, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _validate_runtime_tokenizer(contract: CorrectedRoboCasaPolicyContract, tokenizer: Any) -> None:
    plain = OmegaConf.to_container(tokenizer, resolve=False) if isinstance(tokenizer, DictConfig) else tokenizer
    contract.validate_tokenizer_config(plain)


def overlay_config(
    config: Any,
    args: argparse.Namespace,
    contract: CorrectedRoboCasaPolicyContract,
    *,
    resume_checkpoint: Path | None = None,
) -> None:
    from examples.psm_wma_robocasa_local_s1 import overlay_local_config

    overlay_local_config(config)
    if getattr(args, "no_ttt", False):
        # Config-only overlay above supplies the identical non-Local optimizer,
        # compile and EMA settings. Disable before any model is instantiated.
        config.model.config.local_memory_enabled = False
    synchronize_legacy_chunk_length(config, contract)
    model = config.model.config
    model.local_memory_ttt_tbptt_steps = args.t
    model.local_memory_k_local = args.k
    resolved_tokenizer = contract.resolve_tokenizer_config(model.tokenizer)
    contract.validate_tokenizer_config(resolved_tokenizer)
    model.tokenizer = resolved_tokenizer
    _validate_runtime_tokenizer(contract, model.tokenizer)
    model.parallelism.data_parallel_shard_degree = args.world_size
    model.parallelism.data_parallel_replicate_degree = 1
    config.trainer.type = GroupedLocalMemoryTrainer
    if getattr(args, "no_ttt", False):
        from cosmos_framework.trainer.matched_no_ttt import MatchedNoTTTTrainer

        config.trainer.type = MatchedNoTTTTrainer
    config.trainer.grad_accum_iter = args.ga
    config.trainer.max_iter = args.max_iter
    config.trainer.run_validation = False
    config.trainer.callbacks = {}
    config.trainer.save_zero_checkpoint = False
    config.optimizer.keys_to_select = list(GENERATION_KEYS if getattr(args, "no_ttt", False) else OPTIMIZER_KEYS)
    config.optimizer.lr_multipliers = {key: 5.0 for key in ACTION_KEYS}
    config.scheduler.cycle_lengths = [args.max_iter]
    config.scheduler.warm_up_steps = [args.warmup]
    config.checkpoint.save_iter = args.save_iter
    config.checkpoint.strict_resume = True
    config.checkpoint.dcp_async_mode_enabled = False
    config.checkpoint.type = {"_target_": SafeLocalDistributedCheckpointer}
    config.checkpoint.load_path = str(resume_checkpoint or args.base_checkpoint)
    config.checkpoint.load_training_state = resume_checkpoint is not None
    config.checkpoint.keys_to_skip_loading = ["net_ema."] if resume_checkpoint else ["net_ema.", "local_memory"]
    config.job.project = "psm_wma_v3"
    config.job.group = "corrected_phase5"
    config.job.name = args.job_name
    if model.max_action_dim != 64 or model.local_memory_action_dim != 15:
        raise ValueError("Edge raw15/64 合同不匹配")


def _load_config(args: argparse.Namespace):
    overrides = {
        "EDGE_POLICY_CHECKPOINT": str(args.edge),
        "WAN_VAE_PATH": str(args.vae),
        "ROBOCASA_ROOT": str(args.source_root),
        "BASE_CHECKPOINT_PATH": str(args.base_checkpoint),
    }
    previous = {key: os.environ.get(key) for key in overrides}
    try:
        os.environ.update(overrides)
        return load_experiment_from_toml(RECIPE)
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _resume_checkpoint(args: argparse.Namespace) -> Path | None:
    override = _owner_g1_cross_pair_resume(args)
    if args.phase == "fresh":
        return None
    latest = args.output_root / "psm_wma_v3/corrected_phase5" / args.job_name / "checkpoints/latest_checkpoint.txt"
    if latest.is_symlink() or latest.parent.is_symlink():
        raise ValueError("same-job checkpoint symlink alias is forbidden")
    name = latest.read_text().strip()
    if not name.startswith("iter_") or not name[5:].isdigit() or not 0 < int(name[5:]) < args.max_iter:
        raise ValueError("same-job latest checkpoint iteration 不合法")
    if override and int(name[5:]) < int(G1_CHECKPOINT[5:]):
        raise ValueError("G1 Resume latest checkpoint predates the original completed iter8900")
    checkpoint = latest.parent / name
    if checkpoint.is_symlink():
        raise ValueError("same-job checkpoint symlink alias is forbidden")
    for component in ("model", "optim", "scheduler", "trainer"):
        if (checkpoint / component).is_symlink() or (checkpoint / component / ".metadata").is_symlink():
            raise ValueError("same-job checkpoint component symlink is forbidden")
        if not (checkpoint / component / ".metadata").is_file():
            raise FileNotFoundError(f"same-job DCP 缺少 {component}")
    for rank in range(args.world_size):
        if (checkpoint / "dataloader").is_symlink() or (checkpoint / "dataloader" / f"rank_{rank}.pkl").is_symlink():
            raise ValueError("same-job Local sidecar symlink is forbidden")
        if not (checkpoint / "dataloader" / f"rank_{rank}.pkl").is_file():
            raise FileNotFoundError(f"same-job DCP 缺少 rank_{rank} Local 状态")
    return checkpoint


def _audit_root_noncode_changes(porcelain_z: str) -> tuple[str, ...]:
    """Permit known MM notes/Evidence, never production-code or unknown root changes."""
    if not porcelain_z:
        return ()
    records = porcelain_z.split("\0")
    if records[-1] != "":
        raise ValueError("Root git status porcelain -z 格式不完整")
    permitted = []
    for record in records[:-1]:
        if len(record) < 4 or record[2] != " ":
            raise ValueError("Root git status porcelain -z 记录格式非法")
        status, path = record[:2], record[3:]
        notes = status in {" M", "M ", "MM"} and path in {"SESSION.md", "TODO.md"}
        evidence_json = status == "??" and path.startswith("artifacts/g0/") and path.endswith(".json")
        evidence_report = (
            status == "??"
            and path.startswith("docs/collab/chatgpt/DS_PRO_")
            and path.endswith(".md")
            and "/" not in path[len("docs/collab/chatgpt/") :]
        )
        # Historical local-only H3-F visualization helper (created before this
        # Gate, never imported by the corrected training entrypoint).
        h3f_monitor = status == "??" and path == "scripts/plot_h3f_monitor.py"
        if not (notes or evidence_json or evidence_report or h3f_monitor):
            raise ValueError(f"Root 工作树含非授权改动：{status} {path}")
        permitted.append(f"{status} {path}")
    return tuple(permitted)


def verify_root_child_lock(args: argparse.Namespace) -> dict[str, Any]:
    override = _owner_g1_cross_pair_resume(args)
    child = Path(__file__).resolve().parents[1]

    def git(repo: Path, *command: str) -> str:
        return subprocess.run(
            ("git", "-C", str(repo), *command), check=True, capture_output=True, text=True
        ).stdout.strip()

    root = args.root_worktree
    actual_root = git(root, "rev-parse", "HEAD")
    actual_child = git(child, "rev-parse", "HEAD")
    gitlink = git(root, "ls-tree", "HEAD", "cosmos-framework").split()[2]
    if (
        (not override and (len(args.expected_root or "") != 40 or actual_root != args.expected_root))
        or (not override and (len(args.expected_child or "") != 40 or actual_child != args.expected_child))
        or actual_child != gitlink
        or git(child, "status", "--porcelain")
    ):
        raise ValueError("V3 root/child/Gitlink 或 child 工作树不匹配")
    # Gitlink and child source remain strictly clean and SHA-pinned. The root
    # tree contains MM-owned notes and DS-only reports that must be preserved.
    # Read porcelain -z without .strip(), which would destroy its leading XY status.
    root_status = subprocess.run(
        ("git", "-C", str(root), "status", "--porcelain=v1", "-z", "--untracked-files=all"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    noncode_changes = _audit_root_noncode_changes(root_status)
    lock = {
        "root": actual_root,
        "child": actual_child,
        "gitlink": gitlink,
        "allowed_root_noncode_changes": len(noncode_changes),
    }
    if override:
        # Reject unrelated Child histories while allowing G1 to continue
        # under later V3 revisions without repinning EXPECTED_CHILD.
        git(child, "merge-base", "--is-ancestor", G1_RUNTIME_BASE_CHILD, actual_child)
        lock["allowed_root_noncode_change_paths"] = list(noncode_changes)
        lock["runtime_base_child"] = G1_RUNTIME_BASE_CHILD
    return lock


def _execution_stop_iteration(args: argparse.Namespace) -> int:
    """Bound execution only; preserve the configured training/scheduler horizon."""
    stop_after = getattr(args, "stop_after_iter", None)
    formal_until = getattr(args, "formal_execution_until", None)
    if formal_until is not None:
        if (
            stop_after is not None
            or args.job_name.startswith("bounded_")
            or type(formal_until) is not int
            or not 1 <= formal_until <= args.max_iter
            or (args.max_iter, args.warmup, args.save_iter) != (30000, 500, 100)
            or getattr(args, "audit_missing_grads", False)
        ):
            raise ValueError("formal execution budget requires 1..30000 and schedule 30000/500/100; no diagnostics")
        return formal_until
    if getattr(args, "audit_missing_grads", False) and stop_after != 3:
        raise ValueError("--audit-missing-grads requires fresh bounded --stop-after-iter 3")
    if stop_after is None:
        if args.phase == "resume" and args.job_name.startswith("bounded_"):
            raise ValueError("bounded diagnostic checkpoint must not be resumed as formal training")
        return args.max_iter
    if (
        args.phase != "fresh"
        or type(stop_after) is not int
        or not 1 <= stop_after < args.max_iter
        or (args.max_iter, args.warmup, args.save_iter) != (30000, 500, 100)
        or not args.job_name.startswith("bounded_")
    ):
        raise ValueError(
            "--stop-after-iter is fresh-only and requires formal 30000/500/100 schedule "
            "and a distinct bounded_ job-name"
        )
    namespace = args.output_root / "psm_wma_v3" / "corrected_phase5" / args.job_name
    if namespace.exists():
        raise FileExistsError(f"bounded diagnostic requires a new isolated job namespace: {namespace}")
    return stop_after


def preflight(args: argparse.Namespace) -> tuple[dict[str, Any], Any, ExactWindowLocalCatalog, Any]:
    if getattr(args, "dataset_index_root", None) is None:
        raise ValueError("训练/Resume 必须提供已离线验收的 --dataset-index-root；禁止 cold build 与数据全量复检")
    check_job_namespace(args)
    if getattr(args, "no_ttt", False) and (
        (args.t, args.b, args.ga, args.k, args.world_size) != (16, 8, 2, 4, 8)
        or args.num_workers != 0
        or getattr(args, "full_segment_prefetch", False)
        or getattr(args, "transform_rng_mode", "legacy_global") != "legacy_global"
    ):
        raise ValueError("Matched No-TTT freezes T16/B8/GA2/world8 and synchronous legacy-global transforms")
    if any(
        type(value) is not int or value <= 0
        for value in (args.t, args.b, args.ga, args.k, args.world_size, args.max_iter, args.save_iter)
    ):
        raise ValueError("T/B/GA/K/world_size/max_iter/save_iter 必须是正整数")
    if args.warmup < 0 or args.warmup > args.max_iter:
        raise ValueError("warmup 越界")
    execution_stop = _execution_stop_iteration(args)
    if getattr(args, "full_segment_prefetch", False) and (
        args.num_workers != 1 or getattr(args, "transform_rng_mode", "legacy_global") != "consumer_keyed_v1"
    ):
        raise ValueError("full segment prefetch requires --num-workers 1 --transform-rng-mode consumer_keyed_v1")
    _load_runtime()
    lock = verify_root_child_lock(args)
    os.environ["HF_HUB_OFFLINE"] = "1"
    check_edge_checkpoint(str(args.edge))
    check_droid_dcp(args.base_checkpoint)
    if not args.vae.is_file():
        raise FileNotFoundError("WAN VAE 文件缺失")
    resume = _resume_checkpoint(args)
    if resume is not None and int(resume.name[5:]) >= execution_stop:
        raise ValueError("restored iteration must be below execution budget")
    config = _load_config(args)
    init_start = time.perf_counter()
    verify_start = time.perf_counter()
    index_root = getattr(args, "dataset_index_root", None)
    verified_index = VerifiedExactWindowIndex.open(
        index_root,
        cache_root=args.cache_root,
        source_root=args.source_root,
        verify_files=False,  # owner-approved offline acceptance: never rescan on train/resume
    )
    index_verify_ms = (time.perf_counter() - verify_start) * 1000
    stage = time.perf_counter()
    cache = RoboCasaExactWindowCacheCatalog(args.cache_root, verified_index=verified_index)
    cache_catalog_ms = (time.perf_counter() - stage) * 1000
    contract = CorrectedRoboCasaPolicyContract.from_cache_catalog(cache)
    overlay_config(config, args, contract, resume_checkpoint=resume)
    tokenizer = config.model.config.vlm_config.tokenizer
    stage = time.perf_counter()
    dataset = get_action_robocasa_exact_window_cached_sft_dataset(
        cache_root=args.cache_root,
        source_root=args.source_root,
        tokenizer_config=tokenizer,
        catalog=cache,
        verified_index=verified_index,
    )
    dataset_create_ms = (time.perf_counter() - stage) * 1000
    stage = time.perf_counter()
    catalog = ExactWindowLocalCatalog(dataset, ttt_tbptt_steps=args.t)
    local_catalog_ms = (time.perf_counter() - stage) * 1000
    dataset_init = {
        "verified_index_hit": True,
        "verified_index_mode": "trusted_offline_receipt_no_rescan",
        "index_verify_ms": index_verify_ms,  # historical telemetry key; now measures open only
        "index_open_ms": index_verify_ms,
        "cache_catalog_ms": cache_catalog_ms,
        "sft_dataset_create_ms": dataset_create_ms,
        "source_reader_stages": dataset._dataset.source_reader.init_timings_ms,
        "local_catalog_ms": local_catalog_ms,
        "total_ms": (time.perf_counter() - init_start) * 1000,
    }
    if (
        catalog.manifest_digest != cache.manifest_sha256
        or catalog.raw.contract.vae_encode_contract != contract.vae_encode_contract
    ):
        raise ValueError("预检期间 cache VAE authority 漂移")
    _validate_runtime_tokenizer(contract, config.model.config.tokenizer)
    witnesses = model_witnesses(args.edge, args.base_checkpoint)
    digest = config_digest(
        catalog,
        config,
        b_stream=args.b,
        active_ga=args.ga,
        witnesses=witnesses,
        transform_rng_mode=getattr(args, "transform_rng_mode", "legacy_global"),
    )
    planner = ExactWindowRankPlanner(catalog, rank=0, world_size=args.world_size, b_stream=args.b, active_ga=args.ga)
    plan = planner.plan_window(planner.initial_frontier())
    report = {
        "lock": lock,
        "config_digest": digest,
        "transform_rng_mode": getattr(args, "transform_rng_mode", "legacy_global"),
        "full_segment_prefetch": getattr(args, "full_segment_prefetch", False),
        "execution_stop_after_iter": execution_stop,
        "execution_scope": "bounded_diagnostic" if getattr(args, "stop_after_iter", None) is not None else "formal",
        "cache_manifest_sha256": catalog.manifest_digest,
        "cache_corpus_digest": catalog.cache_corpus_digest,
        "source_binding_digest": catalog.source_binding_digest,
        "model_witnesses": witnesses,
        "episodes": len(catalog.episodes),
        "dataset_init": dataset_init,
        "checkpoint_load_path": str(resume or args.base_checkpoint),
        "effective_data_source": describe_effective_data_source(
            config, catalog.raw.contract, ttt_steps=args.t, local_memory_enabled=not getattr(args, "no_ttt", False)
        ),
    }
    semantic = training_identity(args, config, report)
    check_job_namespace(args, semantic)
    report["training_identity"] = semantic
    # A new profile/schema cannot silently resume historical jobs with incomplete identity.
    report["config_digest"] = _json_digest({"phase5_digest": digest, "training_identity": semantic})
    if _owner_g1_cross_pair_resume(args):
        report["owner_cross_pair_resume"] = _g1_resume_compatibility(args, report, digest)
        report["config_digest"] = report["owner_cross_pair_resume"]["config_digest"]
    if args.snapshot10:
        producer = ExactWindowSegmentProducer(
            catalog,
            config_digest=report["config_digest"] if _owner_g1_cross_pair_resume(args) else digest,
            transform_rng_mode=getattr(args, "transform_rng_mode", "legacy_global"),
        )
        checked = 0
        while checked < 10:
            for member in plan.members:
                for request in member:
                    if checked == 10:
                        break
                    segment = producer.produce(request)
                    if segment.consumer_payload[0][0]["cached_latent_required"] is not True:
                        raise ValueError("snapshot10 缺少 cached latent")
                    collate_grouped_native_batch((segment.consumer_payload[0][0],))
                    checked += 1
            if checked < 10:
                plan = planner.plan_window(plan.candidate_frontier)
    return report, config, catalog, dataset


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--no-ttt", action="store_true", help="Independent matched Packed128 baseline; never a G1 Resume override"
    )
    result.add_argument("--phase", choices=("fresh", "resume"), required=True)
    result.add_argument(
        "--owner-authorized-g1-cross-pair-resume",
        action="store_true",
        help="Explicit Owner exception for G1 iter8900 only; preserve old digest and all non-source checks",
    )
    result.add_argument("--preflight", action="store_true")
    result.add_argument("--snapshot10", action="store_true")
    for name in ("output-root", "source-root", "cache-root", "edge", "vae", "base-checkpoint"):
        result.add_argument(f"--{name}", required=True, type=Path)
    result.add_argument("--root-worktree", required=True, type=Path)
    result.add_argument(
        "--formal-execution-until",
        type=int,
        default=None,
        help="Stop a formal job at a committed optimizer iteration without changing its 30000/500/100 schedule",
    )
    result.add_argument(
        "--dataset-index-root",
        type=Path,
        default=None,
        help="Prebuilt verified read-only index; missing/stale index fails closed, never rebuilds during training",
    )
    # Required only for non-G1 jobs; the standing named G1 Resume records
    # observed runtime SHAs and checks actual Child HEAD == Gitlink instead.
    result.add_argument("--expected-root")
    result.add_argument("--expected-child")
    result.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Per-rank bounded background RAW-READ threads; 0 retains synchronous semantics (not DataLoader workers)",
    )
    result.add_argument("--job-name", default="edge_local_exact_window")
    result.add_argument(
        "--transform-rng-mode",
        choices=("legacy_global", "consumer_keyed_v1"),
        default="legacy_global",
        help="Opt-in per-consumer keyed CFG augmentation for deterministic background transforms",
    )
    result.add_argument(
        "--full-segment-prefetch",
        action="store_true",
        help="One bounded background CPU member (Raw+Transform+Segment+Collate), requires num-workers=1 and keyed RNG",
    )
    for name, default in (
        ("t", 16),
        ("b", 8),
        ("ga", 2),
        ("k", 4),
        ("world-size", 8),
        ("max-iter", 30000),
        ("save-iter", 100),
        ("warmup", 500),
    ):
        result.add_argument(f"--{name}", type=int, default=default)
    result.add_argument(
        "--stop-after-iter",
        type=int,
        default=None,
        help="Fresh-only bounded optimizer steps; preserves max_iter=30000/warmup=500/save_iter=100",
    )
    result.add_argument(
        "--audit-missing-grads",
        action="store_true",
        help="Bounded 3-step only: emit rank-local grad=None parameter names after successful optimizer commits",
    )
    return result


def _telemetry_sample_interval(stop_after_iter: int | None) -> int:
    """Sample on the last bounded step; preserve every-100-step formal cadence."""
    return 100 if stop_after_iter is None else stop_after_iter


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    if type(args.num_workers) is not int or not 0 <= args.num_workers <= 16:
        raise ValueError("--num-workers must be 0..16")
    if args.preflight:
        report, _, _, _ = preflight(args)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return
    _load_runtime()
    if int(os.environ.get("WORLD_SIZE", "0")) != args.world_size or not torch.cuda.is_available():
        raise RuntimeError("训练要求匹配 world_size 的 CUDA torchrun")
    os.environ["IMAGINAIRE_OUTPUT_ROOT"] = str(args.output_root)
    distributed.init()
    report, config, catalog, _ = _agree_preflight(args)
    rank = int(os.environ["RANK"])
    if rank == 0:
        print("[CorrectedV3][dataset_init] " + json.dumps(report["dataset_init"], sort_keys=True))
    lifecycle = JobLifecycle(args, report)
    trainer_type = GroupedLocalMemoryTrainer
    if args.no_ttt:
        from cosmos_framework.trainer.matched_no_ttt import MatchedNoTTTTrainer

        trainer_type = MatchedNoTTTTrainer
    trainer = trainer_type(config, config_writer=lifecycle.write_config)
    trainer._execution_max_iter = report["execution_stop_after_iter"]
    trainer._require_committed_execution_boundary = True
    with model_init():
        model = instantiate(config.model)
    install_optimizer_inventory_check(model, report, no_ttt=args.no_ttt)
    planner = ExactWindowRankPlanner(catalog, rank=rank, world_size=args.world_size, b_stream=args.b, active_ga=args.ga)
    producer = ExactWindowSegmentProducer(
        catalog,
        config_digest=report["config_digest"],
        transform_rng_mode=report["transform_rng_mode"],
    )
    trainer.bind_grouped_stream(
        planner,
        producer,
        config_digest=report["config_digest"],
        num_workers=args.num_workers,
        full_segment_prefetch=report["full_segment_prefetch"],
    )
    sample_interval = _telemetry_sample_interval(getattr(args, "stop_after_iter", None))
    journal = (
        TelemetryJournal(args.output_root / "psm_wma_v3" / "corrected_phase5" / args.job_name, rank=0)
        if rank == 0
        else None
    )
    observer_type = GroupedPlanObserver
    if args.no_ttt:
        from examples.psm_wma_robocasa_matched_no_ttt import NoTTTPlanObserver

        observer_type = NoTTTPlanObserver
    observer = observer_type(
        rank=rank,
        parameter_group=_telemetry_parameter_group,
        cuda_sample_interval=sample_interval,
        parameter_norm_interval=sample_interval,
        audit_missing_gradients=args.audit_missing_grads,
        human_progress=rank == 0,
        max_iter=args.max_iter,
        emit=journal if journal is not None else print,
    )
    trainer.grouped_observer = observer
    install_checkpoint_save_telemetry(trainer.checkpointer, observer)
    try:
        trainer.train(
            model,
            GroupedTriggerLoader(args.max_iter, args.ga, on_iteration_start=observer.start_iteration),
            None,
        )
    finally:
        if journal is not None:
            journal.close()
    start_iteration = int(Path(report["checkpoint_load_path"]).name[5:]) if args.phase == "resume" else 0
    expected_iteration = report["execution_stop_after_iter"]
    if (
        trainer._grouped_completed_iteration != expected_iteration
        or observer.completed != expected_iteration - start_iteration
    ):
        raise RuntimeError("Phase5 完成的 optimizer iteration 不匹配")
    lifecycle.close()
    if rank == 0 and getattr(args, "stop_after_iter", None) is not None:
        print(
            "[CorrectedV3][bounded_stop] "
            + json.dumps(
                {
                    "completed_iteration": expected_iteration,
                    "configured_max_iter": args.max_iter,
                    "configured_warmup": args.warmup,
                    "config_digest": report["config_digest"],
                    "diagnostic_only_do_not_resume": True,
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
