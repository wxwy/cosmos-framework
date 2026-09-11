"""CPU/static contract for Local Memory config, selectors and slow checkpoints."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch
from torch import nn

_CONFIG_VERSION = 1
_PAYLOAD_VERSION = 1
_CONFIG_KEYS = frozenset({"version", "ttt_tbptt_steps", "ttt_inner_lr", "k_local", "local_evidence_feature_version", "local_fast_state_dtype", "local_runtime_resume_mode"})
_PAYLOAD_KEYS = frozenset({"version", "config", "feature_config", "base_identity", "parameters", "optimizer", "optimizer_identity", "scheduler", "scheduler_identity", "iteration"})
_FEATURE_CONFIG_SCHEMA = "canonical_native_local_ttt_config_v2"
_FEATURE_CONFIG_KEYS = frozenset(
    {
        "schema",
        "local_memory_enabled",
        "local_memory_dim",
        "local_history_enabled",
        "local_history_backend",
        "local_history_evidence_dim",
        "local_history_state_enabled",
        "local_ttt_enabled",
        "enable_input_bias",
        "ttt_tbptt_steps",
        "ttt_inner_lr",
        "k_local",
        "local_evidence_feature_version",
        "local_fast_state_dtype",
        "local_runtime_resume_mode",
    }
)
_BASE_IDENTITY_SCHEMA = "canonical_native_local_ttt_base_v1"
_BASE_IDENTITY_KEYS = frozenset(
    {
        "schema",
        "child_git_revision",
        "canonical_model_config_sha256",
        "checkpoint_source_fingerprint",
        "manifest_sha256",
        "source_sha256",
    }
)
_RUNTIME_KEY_FRAGMENTS = ("continualtttfaststate", "fast_state", "frontier", "pending", "scan", "native_forward", "commit", "retry", "suffix", "transaction", "receipt", "cursor", "queue", "rng", "grad")

SELECTORS = (
    "local_memory_runtime.evidence_encoder.",
    "local_memory_runtime.ttt_core.",
    "local_memory2llm.",
    "local_memory_modality_embed",
)


def _canonical_sha256(value: Mapping[str, object]) -> str:
    """Hash an exact identity mapping without allowing non-finite JSON values."""
    try:
        encoded = json.dumps(dict(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("identity is not canonical JSON") from error
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FeatureConfigIdentity:
    local_memory_enabled: bool = True
    local_memory_dim: int = 32
    local_history_enabled: bool = True
    local_history_backend: str = "ttt_fast_weight"
    local_history_evidence_dim: int = 96
    local_history_state_enabled: bool = False
    local_ttt_enabled: bool = True
    enable_input_bias: bool = True
    ttt_tbptt_steps: int = 16
    ttt_inner_lr: float = 0.1
    k_local: int = 1
    local_evidence_feature_version: str = "causal_visual96_executed_action10_v1"
    local_fast_state_dtype: str = "fp32"
    local_runtime_resume_mode: str = "slow_only_no_mid_episode_resume"

    def validate(self) -> None:
        if not all(isinstance(value, bool) for value in (self.local_memory_enabled, self.local_history_enabled, self.local_history_state_enabled, self.local_ttt_enabled, self.enable_input_bias)):
            raise ValueError("Local Memory feature booleans must be bool")
        if not self.local_memory_enabled or self.local_memory_dim != 32 or not self.local_history_enabled or self.local_history_backend != "ttt_fast_weight" or self.local_history_state_enabled or not self.local_ttt_enabled:
            raise ValueError("Local Memory feature identity is not active canonical TTT")
        if isinstance(self.local_history_evidence_dim, bool) or not isinstance(self.local_history_evidence_dim, int) or self.local_history_evidence_dim <= 0:
            raise ValueError("local_history_evidence_dim must be positive")
        LocalMemoryConfig(self.ttt_tbptt_steps, self.ttt_inner_lr, self.k_local, self.local_evidence_feature_version, self.local_fast_state_dtype, self.local_runtime_resume_mode).validate()

    def to_mapping(self) -> dict[str, object]:
        self.validate()
        return {"schema": _FEATURE_CONFIG_SCHEMA, **asdict(self)}

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "FeatureConfigIdentity":
        if not isinstance(value, Mapping) or set(value) != _FEATURE_CONFIG_KEYS or value.get("schema") != _FEATURE_CONFIG_SCHEMA:
            raise ValueError("FeatureConfigIdentity is not the exact 15-key mapping")
        result = cls(**{key: value[key] for key in _FEATURE_CONFIG_KEYS if key != "schema"})
        result.validate()
        return result


def build_base_identity(*, child_git_revision: str, feature_config: FeatureConfigIdentity, checkpoint_source_fingerprint: str, manifest_sha256: str, source_sha256: str) -> dict[str, object]:
    feature_digest = _canonical_sha256(feature_config.to_mapping())
    result = {
        "schema": _BASE_IDENTITY_SCHEMA,
        "child_git_revision": child_git_revision,
        "canonical_model_config_sha256": feature_digest,
        "checkpoint_source_fingerprint": checkpoint_source_fingerprint,
        "manifest_sha256": manifest_sha256,
        "source_sha256": source_sha256,
    }
    if set(result) != _BASE_IDENTITY_KEYS or any(not isinstance(value, str) or not value for value in result.values()):
        raise ValueError("base identity must contain immutable non-empty strings")
    return result


def _validate_base_identity(value: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != _BASE_IDENTITY_KEYS or value.get("schema") != _BASE_IDENTITY_SCHEMA:
        raise ValueError("base identity is not the canonical versioned mapping")
    result = dict(value)
    if any(not isinstance(item, str) or not item for item in result.values()):
        raise ValueError("base identity contains an empty or non-string field")
    return result


def _validate_feature_config_for_local(feature_config: FeatureConfigIdentity, config: "LocalMemoryConfig") -> None:
    """Keep the legacy local subset and the exact feature identity inseparable."""
    feature_config.validate()
    if (
        feature_config.ttt_tbptt_steps != config.ttt_tbptt_steps
        or feature_config.ttt_inner_lr != config.ttt_inner_lr
        or feature_config.k_local != config.k_local
        or feature_config.local_evidence_feature_version != config.local_evidence_feature_version
        or feature_config.local_fast_state_dtype != config.local_fast_state_dtype
        or feature_config.local_runtime_resume_mode != config.local_runtime_resume_mode
    ):
        raise ValueError("FeatureConfigIdentity and LocalMemoryConfig disagree")


def _validate_identity_binding(feature_config: FeatureConfigIdentity, base_identity: Mapping[str, object]) -> dict[str, object]:
    result = _validate_base_identity(base_identity)
    if result["canonical_model_config_sha256"] != _canonical_sha256(feature_config.to_mapping()):
        raise ValueError("base identity does not bind the exact feature config")
    return result


@dataclass(frozen=True)
class LocalMemoryConfig:
    ttt_tbptt_steps: int = 16
    ttt_inner_lr: float = 0.1
    k_local: int = 1
    local_evidence_feature_version: str = "causal_visual96_executed_action10_v1"
    local_fast_state_dtype: str = "fp32"
    local_runtime_resume_mode: str = "slow_only_no_mid_episode_resume"

    def validate(self) -> None:
        if isinstance(self.ttt_tbptt_steps, bool) or not isinstance(self.ttt_tbptt_steps, int) or self.ttt_tbptt_steps <= 0:
            raise ValueError("ttt_tbptt_steps must be a positive integer")
        if isinstance(self.ttt_inner_lr, bool) or not isinstance(self.ttt_inner_lr, (int, float)) or not math.isfinite(float(self.ttt_inner_lr)) or self.ttt_inner_lr <= 0:
            raise ValueError("ttt_inner_lr must be finite and positive")
        if isinstance(self.k_local, bool) or not isinstance(self.k_local, int) or self.k_local != 1:
            raise ValueError("k_local must be exactly 1")
        if self.local_evidence_feature_version != "causal_visual96_executed_action10_v1":
            raise ValueError("local_evidence_feature_version is not canonical")
        if self.local_fast_state_dtype != "fp32":
            raise ValueError("local_fast_state_dtype must be fp32")
        if self.local_runtime_resume_mode != "slow_only_no_mid_episode_resume":
            raise ValueError("local_runtime_resume_mode is not canonical")

    def to_mapping(self) -> dict[str, object]:
        self.validate()
        return {"version": _CONFIG_VERSION, **asdict(self)}

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "LocalMemoryConfig":
        if not isinstance(value, Mapping) or set(value) != _CONFIG_KEYS or value.get("version") != _CONFIG_VERSION:
            raise ValueError("Local Memory config identity is not the canonical versioned mapping")
        result = cls(**{key: value[key] for key in _CONFIG_KEYS if key != "version"})
        result.validate()
        return result


def _is_selector_match(name: str, selector: str) -> bool:
    return name == selector or name.startswith(selector)


def _validate_selector_cover(names: Mapping[str, torch.Tensor]) -> None:
    matches = {name: tuple(selector for selector in SELECTORS if _is_selector_match(name, selector)) for name in names}
    if any(len(value) != 1 for value in matches.values()) or any(not any(selector in match for match in matches.values()) for selector in SELECTORS):
        raise ValueError("Local Memory selectors do not exact-cover the slow inventory")
    if any(not isinstance(parameter, nn.Parameter) or not parameter.requires_grad for parameter in names.values()):
        raise ValueError("Local Memory slow inventory contains a non-trainable value")
    if len({id(value) for value in names.values()}) != len(names):
        raise ValueError("Local Memory slow inventory contains an alias")


def validate_slow_inventory(root: nn.Module, *, runtime_encoder: nn.Module, runtime_core: nn.Module, adapter: object | None = None) -> tuple[str, ...]:
    """Prove that the registered active TTT owner is the sole slow owner."""
    if (not isinstance(root, nn.Module) or getattr(root, "evidence_encoder", None) is not runtime_encoder or getattr(root, "ttt_core", None) is not runtime_core or tuple(root._modules) != ("evidence_encoder", "ttt_core")):
        raise ValueError("runtime must reference exactly the registered Local Memory owner objects")
    if adapter is not None and (getattr(adapter, "encoder", None) is not runtime_encoder or getattr(adapter, "core", None) is not runtime_core):
        raise ValueError("canonical adapter is not bound to the registered Local Memory owner")
    names = tuple(f"local_memory_runtime.{name}" for name, _ in root.named_parameters())
    if any("readout" in name or "recurrent" in name or "local_history_runtime" in name for name in names):
        raise ValueError("legacy Local Memory owner leaked into the active TTT inventory")
    return names


def canonical_slow_inventory(root: nn.Module, local_memory2llm: nn.Module, modality: nn.Parameter) -> dict[str, nn.Parameter]:
    if (
        not isinstance(local_memory2llm, nn.Linear)
        or local_memory2llm.in_features != 32
        or local_memory2llm.out_features != 2048
        or tuple(modality.shape) != (2048,)
    ):
        raise ValueError("Local Memory projector must be the canonical per-token 32 -> 2048 ABI")
    values = {f"local_memory_runtime.{name}": value for name, value in root.named_parameters()}
    values.update({f"local_memory2llm.{name}": value for name, value in local_memory2llm.named_parameters()})
    values["local_memory_modality_embed"] = modality
    _validate_selector_cover(values)
    return values


def validate_exact_optimizer_membership(candidate: Mapping[str, torch.Tensor], expected: Mapping[str, torch.Tensor]) -> None:
    if set(candidate) != set(expected) or any(candidate[name] is not expected[name] for name in expected) or len({id(value) for value in candidate.values()}) != len(candidate):
        raise ValueError("optimizer membership is not exact")


def validate_optimizer_membership(candidate: Mapping[str, torch.Tensor], expected: Mapping[str, torch.Tensor]) -> tuple[str, ...]:
    _validate_selector_cover(candidate)
    validate_exact_optimizer_membership(candidate, expected)
    return tuple(candidate)


def _clone_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, Mapping):
        return {key: _clone_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_clone_value(item) for item in value)
    if isinstance(value, list):
        return [_clone_value(item) for item in value]
    return value


def _fully_qualified_class(value: object) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _value_schema(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _value_schema(item) for key, item in value.items()}
    if isinstance(value, list):
        return ["list", [_value_schema(item) for item in value]]
    if isinstance(value, tuple):
        return ["tuple", [_value_schema(item) for item in value]]
    return _fully_qualified_class(value)


def _optimizer_identity(optimizer: torch.optim.Optimizer, expected: Mapping[str, torch.Tensor]) -> dict[str, object]:
    names_by_id = {id(parameter): name for name, parameter in expected.items()}
    groups: list[dict[str, object]] = []
    seen: set[int] = set()
    for group in optimizer.param_groups:
        parameters = group.get("params")
        if not isinstance(parameters, list):
            raise ValueError("optimizer group parameters must be a list")
        members: list[str] = []
        for parameter in parameters:
            name = names_by_id.get(id(parameter))
            if name is None or id(parameter) in seen:
                raise ValueError("checkpoint optimizer group members must exactly match slow inventory")
            seen.add(id(parameter))
            members.append(name)
        groups.append({"members": members, "hyperparameters": {key: _clone_value(value) for key, value in group.items() if key != "params"}})
    if tuple(name for group in groups for name in group["members"]) != tuple(expected) or len(seen) != len(expected):
        raise ValueError("checkpoint optimizer group member order must exactly match slow inventory")
    return {"class": _fully_qualified_class(optimizer), "groups": groups}


def _scheduler_identity(scheduler: object) -> dict[str, object]:
    state_dict = getattr(scheduler, "state_dict", None)
    if not callable(state_dict):
        raise ValueError("scheduler must provide state_dict")
    state = state_dict()
    if not isinstance(state, Mapping):
        raise ValueError("scheduler state must be a mapping")
    return {"class": _fully_qualified_class(scheduler), "state_schema": _value_schema(state)}


def _contains_runtime_key(names: Mapping[str, object]) -> bool:
    for name in names:
        lowered = name.lower()
        if name.startswith("local_memory_runtime.ttt_core.w0_fast_"):
            continue
        if any(fragment in lowered for fragment in _RUNTIME_KEY_FRAGMENTS):
            return True
    return False


def validate_pristine_progress(*, iteration: object, optimizer_state: object, scheduler_state: object, pristine_scheduler_state: object, optimizer_present: bool, scheduler_present: bool) -> None:
    """Validate the v0.3 no-step checkpoint progress predicate without mutation."""
    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration != 0:
        raise ValueError("checkpoint progress must be pristine iteration zero")
    if optimizer_present != scheduler_present:
        raise ValueError("checkpoint optimizer and scheduler must be jointly present or absent")
    if not optimizer_present:
        if optimizer_state is not None or scheduler_state is not None:
            raise ValueError("absent checkpoint optimizer and scheduler must be null")
        return
    if not isinstance(optimizer_state, Mapping) or optimizer_state.get("state") != {}:
        raise ValueError("checkpoint optimizer state must be pristine empty")
    if scheduler_state != pristine_scheduler_state:
        raise ValueError("checkpoint scheduler state is not pristine")


def slow_checkpoint_payload(named_parameters: Mapping[str, torch.Tensor], config: LocalMemoryConfig, *, feature_config: FeatureConfigIdentity, base_identity: Mapping[str, object], optimizer: torch.optim.Optimizer | None = None, scheduler: object | None = None, iteration: int = 0) -> dict[str, object]:
    """Create the only synthetic, in-memory slow-only checkpoint payload."""
    config.validate()
    if _contains_runtime_key(named_parameters):
        raise ValueError("fast runtime or sidecar state cannot enter checkpoint")
    if not isinstance(iteration, int) or isinstance(iteration, bool) or iteration < 0:
        raise ValueError("iteration must be a non-negative integer")
    _validate_feature_config_for_local(feature_config, config)
    base_identity = _validate_identity_binding(feature_config, base_identity)
    return {
        "version": _PAYLOAD_VERSION,
        "config": config.to_mapping(),
        "feature_config": feature_config.to_mapping(),
        "base_identity": _clone_value(base_identity),
        "parameters": {name: value.detach().clone() for name, value in named_parameters.items()},
        "optimizer": None if optimizer is None else _clone_value(optimizer.state_dict()),
        "optimizer_identity": None if optimizer is None else _optimizer_identity(optimizer, named_parameters),
        "scheduler": None if scheduler is None else _clone_value(scheduler.state_dict()),
        "scheduler_identity": None if scheduler is None else _scheduler_identity(scheduler),
        "iteration": iteration,
    }


def _stage_restore(payload: Mapping[str, object], expected: Mapping[str, torch.Tensor], config: LocalMemoryConfig, *, feature_config: FeatureConfigIdentity, base_identity: Mapping[str, object], optimizer: torch.optim.Optimizer | None, scheduler: object | None, iteration: int) -> dict[str, torch.Tensor]:
    config.validate()
    if not isinstance(payload, Mapping) or set(payload) != _PAYLOAD_KEYS or payload.get("version") != _PAYLOAD_VERSION:
        raise ValueError("checkpoint payload schema mismatch")
    _validate_feature_config_for_local(feature_config, config)
    base_identity = _validate_identity_binding(feature_config, base_identity)
    if (
        LocalMemoryConfig.from_mapping(payload["config"]) != config
        or FeatureConfigIdentity.from_mapping(payload["feature_config"]) != feature_config
        or payload.get("base_identity") != base_identity
    ):
        raise ValueError("checkpoint identity mismatch")
    if payload.get("iteration") != iteration:
        raise ValueError("checkpoint iteration identity mismatch")
    values = payload.get("parameters")
    if not isinstance(values, Mapping) or set(values) != set(expected) or _contains_runtime_key(values):
        raise ValueError("checkpoint slow inventory mismatch")
    if (
        (payload.get("optimizer") is None) != (optimizer is None)
        or (payload.get("optimizer_identity") is None) != (optimizer is None)
        or (payload.get("scheduler") is None) != (scheduler is None)
        or (payload.get("scheduler_identity") is None) != (scheduler is None)
    ):
        raise ValueError("checkpoint optimizer or scheduler presence mismatch")
    if optimizer is not None:
        candidate = payload["optimizer"]
        if payload.get("optimizer_identity") != _optimizer_identity(optimizer, expected):
            raise ValueError("checkpoint optimizer identity mismatch")
        live_groups = optimizer.state_dict().get("param_groups")
        live_parameters = tuple(parameter for group in optimizer.param_groups for parameter in group.get("params", ()))
        if (
            not isinstance(candidate, Mapping)
            or set(candidate) != {"state", "param_groups"}
            or not isinstance(candidate.get("state"), Mapping)
            or not isinstance(candidate.get("param_groups"), list)
            or not isinstance(live_groups, list)
            or len(candidate["param_groups"]) != len(live_groups)
            or len(live_parameters) != len(expected)
            or any(actual is not required for actual, required in zip(live_parameters, expected.values(), strict=True))
            or len({id(parameter) for parameter in live_parameters}) != len(live_parameters)
        ):
            raise ValueError("checkpoint optimizer group schema mismatch")
        for saved, live in zip(candidate["param_groups"], live_groups, strict=True):
            if (
                not isinstance(saved, Mapping)
                or set(saved) != set(live)
                or saved.get("params") != live.get("params")
            ):
                raise ValueError("checkpoint optimizer group schema mismatch")
        parameter_ids = {identifier for group in candidate["param_groups"] for identifier in group["params"]}
        if (
            len(parameter_ids) != len(live_parameters)
            or any(not isinstance(identifier, int) for identifier in parameter_ids)
            or not set(candidate["state"]).issubset(parameter_ids)
            or any(not isinstance(state, Mapping) for state in candidate["state"].values())
        ):
            raise ValueError("checkpoint optimizer state schema mismatch")
        try:
            shadow_optimizer = copy.deepcopy(optimizer)
            shadow_optimizer.load_state_dict(_clone_value(candidate))
        except (KeyError, TypeError, ValueError, RuntimeError) as error:
            raise ValueError("checkpoint optimizer state is not loadable") from error
    if scheduler is not None:
        candidate = payload["scheduler"]
        live = scheduler.state_dict()
        if payload.get("scheduler_identity") != _scheduler_identity(scheduler):
            raise ValueError("checkpoint scheduler identity mismatch")
        if not isinstance(candidate, Mapping) or set(candidate) != set(live):
            raise ValueError("checkpoint scheduler schema mismatch")
        try:
            shadow_scheduler = copy.deepcopy(scheduler)
            shadow_scheduler.load_state_dict(_clone_value(candidate))
        except (KeyError, TypeError, ValueError, RuntimeError) as error:
            raise ValueError("checkpoint scheduler state is not loadable") from error
    validate_pristine_progress(
        iteration=payload.get("iteration"),
        optimizer_state=payload.get("optimizer"),
        scheduler_state=payload.get("scheduler"),
        pristine_scheduler_state=None if scheduler is None else scheduler.state_dict(),
        optimizer_present=optimizer is not None,
        scheduler_present=scheduler is not None,
    )
    restored: dict[str, torch.Tensor] = {}
    for name, target in expected.items():
        value = values[name]
        if not isinstance(value, torch.Tensor) or value.shape != target.shape or value.dtype != target.dtype:
            raise ValueError("checkpoint tensor mismatch")
        restored[name] = value.detach().clone()
    return restored


def strict_restore(payload: Mapping[str, object], expected: Mapping[str, torch.Tensor], config: LocalMemoryConfig, *, feature_config: FeatureConfigIdentity, base_identity: Mapping[str, object]) -> dict[str, torch.Tensor]:
    """Stage tensors only; this never mutates a registered object."""
    return _stage_restore(payload, expected, config, feature_config=feature_config, base_identity=base_identity, optimizer=None, scheduler=None, iteration=0)


def validate_runtime_admission(*, adapter: object, scheduler: object, transaction: object | None = None) -> None:
    """Reject actual non-quiescent canonical runtime authorities before restore."""
    frontier = getattr(adapter, "frontier", None)
    if not isinstance(getattr(frontier, "_states", None), dict) or frontier._states:
        raise ValueError("restore requires an empty canonical fast-state frontier")
    for name in ("_scan_requests", "_scan_results", "_commit_capabilities", "_post_mutation_commits", "_native_forward_capabilities", "_retry_capabilities", "_suffix_recovery_capabilities", "_retryable_source_transient_capabilities", "_suffix_recovery_requests", "_active_suffix_recoveries", "_suffix_recovery_request_ids", "_suffix_recovery_scans", "_suffix_recovery_commits"):
        value = getattr(adapter, name, None)
        if not isinstance(value, (set, dict)) or value:
            raise ValueError(f"restore requires no pending canonical authority: {name}")
    frozen = getattr(scheduler, "_frozen_transitions", None)
    if not isinstance(frozen, list) or frozen:
        raise ValueError("restore requires no frozen scheduler transition")
    if transaction is not None:
        raise ValueError("restore rejects an open canonical transaction or recovery receipt")


def strict_restore_into(root: nn.Module, payload: Mapping[str, object], expected: Mapping[str, torch.Tensor], config: LocalMemoryConfig, *, runtime_encoder: nn.Module, runtime_core: nn.Module, local_memory2llm: nn.Module, modality: nn.Parameter, adapter: object, scheduler: object, transaction: object | None = None, optimizer: torch.optim.Optimizer | None = None, state_scheduler: object | None = None, iteration: int = 0, feature_config: FeatureConfigIdentity, base_identity: Mapping[str, object]) -> nn.Module:
    """Preflight every fallible contract before copying into existing objects once."""
    validate_slow_inventory(root, runtime_encoder=runtime_encoder, runtime_core=runtime_core, adapter=adapter)
    inventory = canonical_slow_inventory(root, local_memory2llm, modality)
    validate_exact_optimizer_membership(expected, inventory)
    staged = _stage_restore(payload, expected, config, feature_config=feature_config, base_identity=base_identity, optimizer=optimizer, scheduler=state_scheduler, iteration=iteration)
    validate_runtime_admission(adapter=adapter, scheduler=scheduler, transaction=transaction)
    with torch.no_grad():
        for name, target in expected.items():
            target.copy_(staged[name])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if state_scheduler is not None:
        state_scheduler.load_state_dict(payload["scheduler"])
    return root
