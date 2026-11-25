# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Project-owned G2 inference config: immutable training source, narrow asset relocation.

No monkeypatches. The native policy server calls this independently before loading
the actual DCP model. All scientific/model values must equal the training YAML.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import yaml

RUNTIME_ASSET_SCHEMA = "psm_g2_project_runtime_asset_contract_v1"
VAE_FIELD = ("model", "config", "tokenizer", "vae_path")
VLM_FIELD = ("model", "config", "vlm_config", "model_name")
TOKENIZER_FIELD = ("model", "config", "vlm_config", "tokenizer", "tokenizer_type")
APPROVED_RELOCATIONS = frozenset((VAE_FIELD, VLM_FIELD, TOKENIZER_FIELD))
REQUIRED_VLM_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "model.safetensors.index.json",
)
HEX64 = re.compile(r"[0-9a-f]{64}")


def _hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _yaml(path: Path) -> dict:
    data = yaml.safe_load(path.read_bytes())
    if type(data) is not dict:
        raise ValueError("native G2 training and effective YAML must be mappings")
    return data


def _entry(config: dict, key_path: tuple[str, ...]) -> Any:
    current = config
    for key in key_path:
        if type(current) is not dict or key not in current:
            raise ValueError("missing native G2 model field " + ".".join(key_path))
        current = current[key]
    return current


def _changes(original, runtime, prefix=()):
    # Preserve type, dictionary key set and array length; True != 1, etc.
    if type(original) is not type(runtime):
        return [(prefix, original, runtime)]
    if isinstance(original, dict):
        if set(original) != set(runtime):
            return [(prefix, sorted(original), sorted(runtime))]
        result = []
        for key in sorted(original):
            result.extend(_changes(original[key], runtime[key], prefix + (key,)))
        return result
    if isinstance(original, list):
        if len(original) != len(runtime):
            return [(prefix, len(original), len(runtime))]
        result = []
        for i, (x, y) in enumerate(zip(original, runtime)):
            result.extend(_changes(x, y, prefix + (i,)))
        return result
    return [(prefix, original, runtime)] if original != runtime else []


def _configs_root(path: Path) -> Path:
    for ancestor in path.resolve().parents:
        if ancestor.name == "configs":
            return ancestor
    raise ValueError("native G2 runtime config and manifest must live in project configs/")


def _regular_file(path: Path) -> None:
    if not path.is_file() or path.is_symlink():
        raise ValueError("native G2 asset must be an existing, non-symlink regular file: " + str(path))


def _asset_manifest(path: Path) -> dict:
    _regular_file(path)
    value = json.loads(path.read_bytes())
    if type(value) is not dict or set(value) != {
        "schema", "checkpoint_job", "canonical_config_sha256", "training_profile",
        "asset_sha256", "asset_relative_paths"
    }:
        raise ValueError("native G2 runtime asset manifest has unknown/missing fields")
    if value["schema"] != RUNTIME_ASSET_SCHEMA or value["training_profile"] not in (
        "g2_r1", "g2_r2", "g1_no_ttt", "g1_local_ttt"
    ):
        raise ValueError("native G2 runtime asset manifest version/profile invalid")
    relative = value["asset_relative_paths"]
    if type(relative) is not dict or set(relative) != {"vlm_dir", "vae_file"}:
        raise ValueError("native G2 runtime asset-relative path contract is incomplete")
    for part in relative.values():
        if (type(part) is not str or not part or part.startswith("/") or
            any(piece in ("", ".", "..") for piece in part.split("/"))):
            raise ValueError("native G2 asset-relative paths must be normalized and safe")
    asset = value["asset_sha256"]
    if type(asset) is not dict or set(asset) != {"vae_file", "vlm_files"}:
        raise ValueError("native G2 runtime asset hashes must name VAE and VLM files")
    if type(asset["vlm_files"]) is not dict or set(asset["vlm_files"]) != set(REQUIRED_VLM_FILES):
        raise ValueError("native G2 runtime VLM asset hash inventory is incomplete")
    for value_hash in [value["canonical_config_sha256"], asset["vae_file"], *asset["vlm_files"].values()]:
        if not isinstance(value_hash, str) or HEX64.fullmatch(value_hash) is None:
            raise ValueError("native G2 runtime asset hashes must be SHA256")
    return value


def validate_project_runtime_config(
    *,
    canonical_config: Path,
    runtime_config: Path,
    asset_manifest: Path | None,
    checkpoint_job: Path,
    training_profile: str,
    expected_tokenizer_sha256: str,
    legacy_g1_additions: dict | None = None,
    asset_identity_manifest: Path | None = None,
) -> dict:
    """Validate before DCP loading; return byte-bound identities and stat guard paths.

    A production relocated config must be under the same project's configs/
    as its versioned asset contract. The checkpoint job's canonical config is
    never substituted or mutated. All other training YAML entries are identical.
    """
    canonical_config, runtime_config = Path(canonical_config), Path(runtime_config)
    checkpoint_job = Path(checkpoint_job)
    _regular_file(canonical_config)
    _regular_file(runtime_config)
    canonical_sha, runtime_sha = _hash(canonical_config), _hash(runtime_config)
    original, effective = _yaml(canonical_config), _yaml(runtime_config)
    compare_effective = effective
    if legacy_g1_additions is not None:
        if training_profile not in ("g1_no_ttt", "g1_local_ttt") or (
            type(legacy_g1_additions) is not dict or
            set(legacy_g1_additions) != {
                "g2_text_enabled", "global_memory_config", "global_memory_enabled",
                "global_memory_latent_channels", "global_memory_top_variant",
            }
        ):
            raise ValueError("native G1 requires an exact reviewed five-field legacy schema")
        expected = _entry(effective, ("model", "config"))
        original_model = _entry(original, ("model", "config"))
        compare_effective = copy.deepcopy(effective)
        for key, value in legacy_g1_additions.items():
            if key in original_model or key not in expected or expected[key] != value or (
                type(expected[key]) is not type(value)
            ):
                raise ValueError("native G1 legacy schema delta differs from reviewed five additions: " + key)
            compare_effective["model"]["config"].pop(key)
    differences = _changes(original, compare_effective)
    wrong = [x for x, _, _ in differences if x not in APPROVED_RELOCATIONS]
    if wrong:
        raise ValueError("native G2 effective config changes frozen model/training semantics at " + repr(wrong[:5]))
    base = dict(
        canonical_config_sha256=canonical_sha,
        runtime_config_sha256=runtime_sha,
        approved_deployment_fields=[".".join(path) for path, _, _ in differences],
        approved_legacy_schema_fields=(
            sorted(legacy_g1_additions) if legacy_g1_additions is not None else []
        ),
        asset_manifest_sha256=None,
        asset_sha256={},
        protected_files=[canonical_config, runtime_config],
    )
    if runtime_config.resolve() == canonical_config.resolve():
        if legacy_g1_additions is not None:
            raise ValueError("native G1 reviewed five-field schema is required in effective config")
        if asset_manifest is not None:
            raise ValueError("native G2 canonical training config must not carry a relocation manifest")
        return base
    if asset_manifest is None:
        raise ValueError("native G2 relocated runtime config requires a project-owned asset manifest")
    manifest_path = Path(asset_manifest)
    _regular_file(manifest_path)
    configs_root = _configs_root(manifest_path)
    if (_configs_root(runtime_config) != configs_root or
        not runtime_config.resolve().is_relative_to(configs_root) or
        not manifest_path.resolve().is_relative_to(configs_root)):
        raise ValueError("native G2 effective config and manifest must share the launched project's configs/")
    manifest = _asset_manifest(manifest_path)
    if (manifest["checkpoint_job"] != checkpoint_job.name
        or manifest["canonical_config_sha256"] != canonical_sha
        or manifest["training_profile"] != training_profile):
        raise ValueError("native G2 asset manifest disagrees with checkpoint training identity")

    # Hash the actual relocated bytes, not merely path strings or directory
    # presence. VAE is large; only hash once at startup and stat-guard after.
    vae_value, model_value, tokenizer_value = (_entry(effective, key) for key in
                                                 (VAE_FIELD, VLM_FIELD, TOKENIZER_FIELD))
    if any(type(v) is not str or not v.startswith("/") for v in (vae_value, model_value, tokenizer_value)):
        raise ValueError("native G2 runtime asset paths must be concrete absolute paths")
    vae, model_dir, tokenizer_dir = Path(vae_value), Path(model_value), Path(tokenizer_value)
    if tokenizer_dir.resolve() != model_dir.resolve() or tokenizer_dir != model_dir:
        raise ValueError("native G2 model and tokenizer must use the same asset directory")
    if not model_dir.is_dir() or model_dir.is_symlink():
        raise ValueError("native G2 VLM asset directory must exist and must not be a symlink")
    relative = manifest["asset_relative_paths"]
    parts = Path(relative["vlm_dir"]).parts
    root = model_dir
    for _ in parts:
        root = root.parent
    if (root / relative["vlm_dir"] != model_dir or
        root / relative["vae_file"] != vae):
        raise ValueError("native G2 runtime assets do not match the project-relative layout")
    _regular_file(vae)
    witnesses = {"vae_file": _hash(vae)}
    if witnesses["vae_file"] != manifest["asset_sha256"]["vae_file"]:
        raise ValueError("native G2 VAE weight bytes do not match the project runtime contract")
    files = [vae]
    for filename in REQUIRED_VLM_FILES:
        p = model_dir / filename
        _regular_file(p)
        fingerprint = _hash(p)
        if fingerprint != manifest["asset_sha256"]["vlm_files"][filename]:
            raise ValueError("native G2 VLM asset bytes differ from the project runtime contract: " + filename)
        witnesses[filename] = fingerprint
        files.append(p)
    if witnesses["tokenizer.json"] != expected_tokenizer_sha256:
        raise ValueError("native G2 runtime tokenizer mismatches training text recipe")
    # The TEST-3 project-owned asset inventory is a strict extension of the
    # lightweight model manifest: hash all 14 actually loaded tokenizer and
    # processor resources, when this integration's config catalog exists.
    published_inventory = configs_root / "eval/robocasa365_v5/asset_identity.json"
    if asset_identity_manifest is None and published_inventory.is_file():
        asset_identity_manifest = published_inventory
    if asset_identity_manifest is not None:
        inventory_file = Path(asset_identity_manifest)
        _regular_file(inventory_file)
        if not inventory_file.resolve().is_relative_to(configs_root):
            raise ValueError("native G2/G1 extended assets must be project-owned configs/")
        inventory = json.loads(inventory_file.read_bytes())
        if (type(inventory) is not dict or inventory.get("schema") != "psm_robocasa365_v5_asset_identity_v1"
            or type(inventory.get("tokenizer_files")) is not dict
            or len(inventory["tokenizer_files"]) != 13
            or type(inventory.get("vae")) is not dict):
            raise ValueError("native G2/G1 extended tokenizer/processor witness inventory invalid")
        if inventory["vae"]["sha256"] != witnesses["vae_file"] or inventory["vae"]["bytes"] != vae.stat().st_size:
            raise ValueError("native G2/G1 extended VAE hash/size mismatch")
        for relative, expected in inventory["tokenizer_files"].items():
            path = Path(relative)
            if path.is_absolute() or ".." in path.parts or "." in path.parts or str(path) != relative:
                raise ValueError("native G2/G1 extended tokenizer path invalid")
            current = model_dir / path
            _regular_file(current)
            if _hash(current) != expected.get("sha256") or current.stat().st_size != expected.get("bytes"):
                raise ValueError("native G2/G1 extended tokenizer or processor hash differs: " + relative)
            files.append(current)
            witnesses["tokenizer_files/" + relative] = expected["sha256"]
        files.append(inventory_file)
        witnesses["full_asset_manifest"] = _hash(inventory_file)
    base.update(asset_manifest_sha256=_hash(manifest_path), asset_sha256=witnesses,
                protected_files=list(dict.fromkeys([*base["protected_files"], manifest_path, *files])))
    return base
