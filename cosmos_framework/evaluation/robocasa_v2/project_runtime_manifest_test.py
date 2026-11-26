"""Native EvalV2 project-model identity: explicit G1/G2 opt-in, no legacy drift.

CPU-only, no model/torch import, GPU, Env, hardware or Episode.
"""
from __future__ import annotations

import copy

import pytest

from cosmos_framework.evaluation.robocasa_v2.contract import build_manifest, validate_manifest


def _experiment():
    return {
        "model_digest": "a" * 64,
        "config_sha256": "b" * 64,
        "runtime_digest": "c" * 64,
        "root_sha": "d" * 40,
        "child_sha": "e" * 40,
        "task_metadata": {"SyntheticTask": "f" * 64},
    }


def _project(profile):
    return {
        "profile": profile,
        "canonical_training_config_sha256": "1" * 64,
        "effective_runtime_config_sha256": "b" * 64,
        "runtime_asset_manifest_sha256": "2" * 64,
    }


def _g2(profile="g2_r1"):
    return {
        "profile": profile,
        "run_recipe_sha256": "3" * 64,
        "checkpoint_metadata_sha256": "4" * 64,
        "actor_options_sha256": "5" * 64,
    }


def _execution():
    return {
        "profile": "phase_b_native_batch_v1",
        "gpu_ids": [0],
        "sims_per_gpu": 1,
        "max_infer_batch_size": 1,
        "batch_wait_ms": 120,
        "noise": "literal_cfg_seed_per_sample_v1",
    }


def _make(experiment, *, local="required", execution=None):
    return build_manifest(
        tasks=["SyntheticTask"],
        episodes=4,
        seed=0,
        horizons={"SyntheticTask": 450},
        experiment=experiment,
        local_memory=local,
        execution=execution,
    )


def test_old_g1_manifest_byte_identity_is_unchanged():
    x = _make(_experiment())
    assert validate_manifest(x) == x
    assert set(x["experiment"]) == set(_experiment())
    assert "project_model_runtime" not in x["experiment"]


def test_old_g2_manifest_byte_identity_is_unchanged():
    exp = dict(_experiment(), g2=_g2())
    x = _make(exp, execution=_execution())
    assert validate_manifest(x) == x
    assert set(x["experiment"]["g2"]) == set(_g2())


@pytest.mark.parametrize("profile,local", [("g1_no_ttt", "off"), ("g1_local_ttt", "required")])
def test_new_g1_project_model_identity_is_exact_and_preserved(profile, local):
    exp = dict(_experiment(), project_model_runtime=_project(profile))
    m = _make(exp, local=local, execution=_execution())
    assert validate_manifest(m) == m
    assert m["experiment"]["project_model_runtime"]["profile"] == profile
    assert m["experiment"]["config_sha256"] == m["experiment"]["project_model_runtime"]["effective_runtime_config_sha256"]


@pytest.mark.parametrize("profile", ["g2_r1", "g2_r2"])
def test_new_g2_top_and_nested_sha_witnesses_agree(profile):
    runtime = _project(profile)
    nested = _g2(profile)
    nested.update({key: runtime[key] for key in (
        "canonical_training_config_sha256",
        "effective_runtime_config_sha256",
        "runtime_asset_manifest_sha256",
    )})
    if profile == "g2_r2":
        nested["view_recipe_id"] = "6" * 64
    exp = dict(_experiment(), g2=nested, project_model_runtime=runtime)
    m = _make(exp, execution=_execution())
    assert validate_manifest(m) == m


@pytest.mark.parametrize("mutation", [
    lambda x: x["project_model_runtime"].update(effective_runtime_config_sha256="0" * 64),
    lambda x: x["project_model_runtime"].update(canonical_training_config_sha256="bad"),
    lambda x: x["project_model_runtime"].update(runtime_asset_manifest_sha256=None),
    lambda x: x["project_model_runtime"].update(profile="unknown"),
    lambda x: x["project_model_runtime"].update(unreviewed_parameter="true"),
    lambda x: x["project_model_runtime"].pop("runtime_asset_manifest_sha256"),
    lambda x: x.update(extra="x"),
])
def test_project_model_schema_fail_closed_on_unknown_or_mismatched_values(mutation):
    exp = dict(_experiment(), project_model_runtime=_project("g1_local_ttt"))
    exp = copy.deepcopy(exp)
    mutation(exp)
    with pytest.raises(ValueError, match="identity"):
        _make(exp, execution=_execution())


def test_project_model_g1_does_not_claim_native_g2_or_wrong_local_mode():
    exp = dict(_experiment(), project_model_runtime=_project("g1_no_ttt"))
    with pytest.raises(ValueError, match="G1 model/Local profile mismatch"):
        _make(exp, local="required", execution=_execution())
    exp["g2"] = _g2()
    with pytest.raises(ValueError, match="G1 model/Local profile mismatch"):
        _make(exp, local="off", execution=_execution())


def test_project_model_g2_requires_native_g2_and_correct_profile():
    exp = dict(_experiment(), project_model_runtime=_project("g2_r1"))
    with pytest.raises(ValueError, match="requires native G2 experiment"):
        _make(exp, execution=_execution())
    exp["g2"] = _g2("g2_r2")
    exp["g2"].update({k:exp["project_model_runtime"][k] for k in (
        "canonical_training_config_sha256","effective_runtime_config_sha256","runtime_asset_manifest_sha256")})
    exp["g2"]["view_recipe_id"] = "7"*64
    with pytest.raises(ValueError, match="must match"):
        _make(exp,execution=_execution())


@pytest.mark.parametrize("field", [
    "canonical_training_config_sha256",
    "effective_runtime_config_sha256",
    "runtime_asset_manifest_sha256",
])
def test_new_g2_rejects_inconsistent_top_and_nested_sha(field):
    runtime = _project("g2_r1")
    nested = _g2()
    nested.update({k:runtime[k] for k in (
        "canonical_training_config_sha256","effective_runtime_config_sha256","runtime_asset_manifest_sha256")})
    nested[field] = "9" * 64
    with pytest.raises(ValueError, match="must match"):
        _make(dict(_experiment(), project_model_runtime=runtime, g2=nested),execution=_execution())


def test_rejects_g2_extra_nested_fields_without_project_opt_in():
    nested = _g2()
    nested["canonical_training_config_sha256"] = "1" * 64
    with pytest.raises(ValueError, match="invalid G2"):
        _make(dict(_experiment(), g2=nested), execution=_execution())


def test_rejects_project_g2_with_missing_nested_identity_fields():
    exp = dict(_experiment(), project_model_runtime=_project("g2_r1"),g2=_g2())
    with pytest.raises(ValueError, match="invalid G2"):
        _make(exp, execution=_execution())


def test_rejects_implicit_project_model_opt_in_none():
    exp = dict(_experiment(),project_model_runtime=None)
    with pytest.raises(ValueError, match="invalid project-owned"):
        _make(exp)


def test_new_manifest_digest_rejected_if_modified_after_publication():
    runtime = _project("g1_local_ttt")
    manifest = _make(dict(_experiment(), project_model_runtime=runtime),execution=_execution())
    manifest["experiment"]["project_model_runtime"]["canonical_training_config_sha256"] = "3"*64
    with pytest.raises(ValueError, match="immutable manifest or protocol"):
        validate_manifest(manifest)
