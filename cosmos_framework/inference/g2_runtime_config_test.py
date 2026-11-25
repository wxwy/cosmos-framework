# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""No-GPU, no-checkpoint model-configuration and asset provenance regression."""
import copy
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from cosmos_framework.inference.g2_runtime_config import (
    APPROVED_RELOCATIONS,
    RUNTIME_ASSET_SCHEMA,
    validate_project_runtime_config,
)


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _put(x, path, value):
    for part in path[:-1]:
        x = x[part]
    x[path[-1]] = value


@pytest.fixture()
def layout(tmp_path):
    job = tmp_path / "training_job"
    job.mkdir()
    canonical = job / "config.yaml"
    base = {
        "model": {"config": {
            "tokenizer": {"vae_path": "/original/vae.pth", "vae_scale_factor": 8},
            "vlm_config": {
                "model_name": "/original/vlm",
                "tokenizer": {"tokenizer_type": "/original/vlm", "vocab_size": 262144},
                "g2_text_enabled": True,
                "local_memory_enabled": True,
                "global_memory_enabled": False,
            },
            "action_chunk_size": 16,
            "use_state": True,
            "state_dim": 15,
            "base_encoding": "raw",
            "diffusion_steps": 30,
        }},
        "dataloader_train": {"cfg_dropout_rate": 0.1, "canvas": [17,20,192,320]},
        "optimizer": {"schedule": {"lr": 0.0001}},
    }
    canonical.write_text(yaml.safe_dump(base, sort_keys=False))
    manifest = tmp_path / "project/configs/eval/model_assets/assets.json"
    manifest.parent.mkdir(parents=True)
    runtime = tmp_path / "project/configs/eval/.runtime/runtime.yaml"
    runtime.parent.mkdir(parents=True)
    def assets(root):
        model = root / "Cosmos3-Edge-Policy-DROID"
        model.mkdir(parents=True)
        witnesses = {}
        for name in ("tokenizer.json", "config.json", "tokenizer_config.json", "model.safetensors.index.json"):
            p = model / name
            p.write_bytes((name + ":unchanged contents").encode())
            witnesses[name] = _sha(p)
        vae = root / "Wan2.2-TI2V-5B/Wan2.2_VAE.pth"
        vae.parent.mkdir()
        vae.write_bytes(b"weights are test-only; never load torch here")
        return model,vae,witnesses
    root = tmp_path / "asset_mount_A"
    model, vae, files = assets(root)
    contract = {
        "schema": RUNTIME_ASSET_SCHEMA,
        "checkpoint_job": job.name,
        "training_profile": "g2_r1",
        "canonical_config_sha256": _sha(canonical),
        "asset_relative_paths": {
            "vlm_dir": "Cosmos3-Edge-Policy-DROID",
            "vae_file": "Wan2.2-TI2V-5B/Wan2.2_VAE.pth",
        },
        "asset_sha256": {"vae_file": _sha(vae), "vlm_files": files},
    }
    manifest.write_text(json.dumps(contract))
    def eff(model_dir=model, vae_file=vae):
        cfg = copy.deepcopy(base)
        _put(cfg, ("model","config","vlm_config","model_name"), str(model_dir))
        _put(cfg, ("model","config","vlm_config","tokenizer","tokenizer_type"), str(model_dir))
        _put(cfg, ("model","config","tokenizer","vae_path"), str(vae_file))
        return cfg
    runtime.write_text(yaml.safe_dump(eff(), sort_keys=False))
    def validate(path=runtime, manifest_path=manifest, expected=files["tokenizer.json"]):
        return validate_project_runtime_config(
            canonical_config=canonical, runtime_config=path,
            asset_manifest=manifest_path, checkpoint_job=job,
            training_profile="g2_r1", expected_tokenizer_sha256=expected,
        )
    return locals()


def test_exact_three_relocations_with_real_content_hashes(layout):
    x = layout["validate"]()
    assert set(x["approved_deployment_fields"]) == {".".join(t) for t in APPROVED_RELOCATIONS}
    assert x["asset_sha256"]["vae_file"] == layout["contract"]["asset_sha256"]["vae_file"]
    assert x["asset_sha256"]["tokenizer.json"] == layout["contract"]["asset_sha256"]["vlm_files"]["tokenizer.json"]
    assert len(x["protected_files"]) == 8  # canonical, effective, contract, VAE + 4 VLM files


def test_canonical_without_manifest_still_preserves_old_native_contract(layout):
    assert layout["validate"](layout["canonical"], None)["approved_deployment_fields"] == []


@pytest.mark.parametrize("modification", [
    ("model","config","action_chunk_size",17),
    ("model","config","use_state",False),
    ("model","config","state_dim",20),
    ("model","config","vlm_config","g2_text_enabled",False),
    ("model","config","vlm_config","global_memory_enabled",True),
    ("dataloader_train","cfg_dropout_rate",0.2),
    ("optimizer","schedule","lr",0.0002),
])
def test_arbitrary_scientific_changes_fail_closed(layout, modification):
    config = yaml.safe_load(layout["runtime"].read_bytes())
    _put(config, modification[:-1], modification[-1])
    layout["runtime"].write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="frozen model/training semantics"):
        layout["validate"]()


def test_relocated_config_needs_project_manifest(layout):
    with pytest.raises(ValueError, match="requires a project-owned asset manifest"):
        layout["validate"](layout["runtime"], None)


def test_manifest_cannot_be_outside_project_configs(layout):
    foreign = layout["job"] / "manifest.json"
    foreign.write_bytes(layout["manifest"].read_bytes())
    with pytest.raises(ValueError, match="configs/"):
        layout["validate"](layout["runtime"], foreign)


def test_bad_canonical_sha_rejected(layout):
    manifest = layout["manifest"]
    x = json.loads(manifest.read_text())
    x["canonical_config_sha256"] = "0"*64
    manifest.write_text(json.dumps(x))
    with pytest.raises(ValueError, match="training identity"):
        layout["validate"]()


def test_unknown_manifest_key_rejected(layout):
    manifest = layout["manifest"]
    x = json.loads(manifest.read_text())
    x["bypass"] = True
    manifest.write_text(json.dumps(x))
    with pytest.raises(ValueError, match="unknown/missing"):
        layout["validate"]()


def test_wrong_checkpoint_training_profile_rejected(layout):
    with pytest.raises(ValueError, match="training identity"):
        validate_project_runtime_config(
            canonical_config=layout["canonical"], runtime_config=layout["runtime"],
            asset_manifest=layout["manifest"], checkpoint_job=layout["job"],
            training_profile="g2_r2",
            expected_tokenizer_sha256=layout["contract"]["asset_sha256"]["vlm_files"]["tokenizer.json"],
        )


def test_mutated_tokenizer_sha_or_content_rejected(layout):
    with pytest.raises(ValueError, match="tokenizer mismatches"):
        layout["validate"](expected="0"*64)
    (layout["model"] / "tokenizer.json").write_bytes(b"malicious")
    with pytest.raises(ValueError, match="VLM asset bytes"):
        layout["validate"]()


def test_vae_content_mutation_is_not_a_path_only_exemption(layout):
    layout["vae"].write_bytes(b"poisoned asset")
    with pytest.raises(ValueError, match="VAE weight bytes"):
        layout["validate"]()


def test_asset_dir_or_vae_symlink_rejected(layout):
    vae = layout["vae"]
    target = vae.with_name("backing.pth")
    vae.rename(target)
    vae.symlink_to(target)
    with pytest.raises(ValueError, match="non-symlink"):
        layout["validate"]()


def test_unreviewed_vlm_and_tokenizer_directory_split_rejected(layout):
    config = yaml.safe_load(layout["runtime"].read_bytes())
    _put(config, ("model","config","vlm_config","tokenizer","tokenizer_type"), str(layout["model"].parent))
    layout["runtime"].write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="same asset directory"):
        layout["validate"]()


def test_relative_deployment_path_rejected(layout):
    config = yaml.safe_load(layout["runtime"].read_bytes())
    _put(config, ("model","config","tokenizer","vae_path"), "./weights.pth")
    layout["runtime"].write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="concrete absolute"):
        layout["validate"]()


def test_byte_identical_data_and_bms_style_asset_roots(layout):
    new_root = layout["job"].parent / "asset_mount_B"
    model, vae, _ = layout["assets"](new_root)
    config = layout["eff"](model,vae)
    layout["runtime"].write_text(yaml.safe_dump(config))
    result = layout["validate"]()
    assert result["asset_sha256"]["vae_file"] == layout["contract"]["asset_sha256"]["vae_file"]
    assert result["runtime_config_sha256"] != layout["contract"]["canonical_config_sha256"]
    assert len(result["approved_deployment_fields"]) == 3
