"""Future-job config truth: real overlay, lifecycle, serializers, cache and CPU tensors.

Heavy model imports are isolated; pinned PolicyContract and Child chunk resolver
execute their original source bodies, not a copied version of their logic.
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import json
import pickle
import sys
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import attrs
import pytest
import torch
import yaml
from omegaconf import DictConfig, OmegaConf

from cosmos_framework.data.generator.action.datasets.robocasa_exact_window_cache import (
    RoboCasaExactWindowCacheCatalog,
    RoboCasaExactWindowEpisodeReader,
)
from cosmos_framework.data.generator.action.datasets.robocasa_exact_window_cache_test import _manifest, _payload, _write
from cosmos_framework.utils.lazy_config import LazyConfig
from examples import psm_wma_robocasa_corrected_phase5 as phase5
from examples.psm_wma_robocasa_effective_config import (
    describe_effective_data_source,
    synchronize_legacy_chunk_length,
    validate_effective_snapshot,
)

CHILD = Path(__file__).resolve().parents[1]
DATASET_PATH = "dataloader_train.dataloader.datasets.robocasa.dataset"


def _policy_contract_class():
    source = CHILD / "cosmos_framework/data/generator/action/datasets/robocasa_exact_window_policy.py"
    tree = ast.parse(source.read_text())
    nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
        and node.name in {"CorrectedRoboCasaPolicyContract", "_mapping", "_exact"}
    ]
    module = ModuleType("config_truth_policy_cpu")
    module.__dict__.update(
        dataclass=dataclasses.dataclass,
        field=dataclasses.field,
        Mapping=Mapping,
        deepcopy=deepcopy,
        DictConfig=DictConfig,
        OmegaConf=OmegaConf,
        RoboCasaExactWindowCacheCatalog=RoboCasaExactWindowCacheCatalog,
        EDGE_MODEL_CONFIG={"tokenizer": {"encode_chunk_frames": {"256": 68, "480": 24}}},
    )
    sys.modules[module.__name__] = module
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    code = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    exec(compile(code, str(source), "exec"), module.__dict__)
    return module.CorrectedRoboCasaPolicyContract


PolicyContract = _policy_contract_class()


def _server_chunk_parser():
    source = CHILD / "cosmos_framework/scripts/action_policy_server_robocasa.py"
    tree = ast.parse(source.read_text())
    parser = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_extract_chunk_length_from_config"
    )
    namespace = {"Any": object, "DictConfig": DictConfig}
    exec(compile(ast.Module(body=[parser], type_ignores=[]), str(source), "exec"), namespace)
    return namespace[parser.name]


server_chunk = _server_chunk_parser()


def _sequence_plan_builder():
    sequence_source = CHILD / "cosmos_framework/data/generator/sequence_packing/sequence.py"
    plan = next(
        node
        for node in ast.parse(sequence_source.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "SequencePlan"
    )
    transforms = CHILD / "cosmos_framework/data/generator/action/utils/transforms.py"
    builder = next(
        node
        for node in ast.parse(transforms.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "build_sequence_plan_from_mode"
    )
    module = ModuleType("config_truth_sequence_cpu")
    module.__dict__.update(dataclass=dataclasses.dataclass, field=dataclasses.field)
    sys.modules[module.__name__] = module
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    code = ast.fix_missing_locations(ast.Module(body=[future, plan, builder], type_ignores=[]))
    exec(compile(code, str(transforms), "exec"), module.__dict__)
    return module.build_sequence_plan_from_mode


build_sequence_plan = _sequence_plan_builder()


@attrs.define
class SnapshotConfig:
    dataloader_train: DictConfig
    model: DictConfig
    trainer: DictConfig
    optimizer: DictConfig
    scheduler: DictConfig
    checkpoint: DictConfig
    job: DictConfig


class SyntheticTrainer:
    pass


def _args(tmp_path):
    return SimpleNamespace(
        phase="fresh",
        output_root=tmp_path,
        job_name="future_truth",
        expected_root="a" * 40,
        expected_child="b" * 40,
        t=8,
        k=4,
        world_size=8,
        ga=2,
        max_iter=30000,
        warmup=500,
        save_iter=100,
        base_checkpoint=tmp_path / "official_droid",
        stop_after_iter=None,
        formal_execution_until=100,
    )


def _config(args, chunk=32):
    dataset = dict(
        chunk_length=chunk,
        fps=20,
        camera_set="left_wrist",
        use_state=True,
        use_base_action=True,
        base_encoding="raw",
        action_normalization=None,
        mode="wam",
        max_action_dim=64,
    )
    create = lambda value: OmegaConf.create(value, flags={"allow_objects": True})  # noqa: E731
    return SnapshotConfig(
        dataloader_train=create({"dataloader": {"datasets": {"robocasa": {"dataset": dataset}}}}),
        model=create(
            {
                "config": {
                    "tokenizer": {"encode_exact_durations": [33], "encode_chunk_frames": {"256": 68, "480": 24}},
                    "parallelism": {},
                    "max_action_dim": 64,
                    "local_memory_action_dim": 15,
                }
            }
        ),
        trainer=create({}),
        optimizer=create({}),
        scheduler=create({}),
        checkpoint=create({}),
        job=create({"path_local": str(phase5.job_namespace(args))}),
    )


def _overlay(config, args, contract, resume_checkpoint=None):
    local = ModuleType("examples.psm_wma_robocasa_local_s1")
    local.overlay_local_config = lambda _: None
    with (
        patch.dict(sys.modules, {local.__name__: local}),
        patch.multiple(
            phase5,
            DictConfig=DictConfig,
            OmegaConf=OmegaConf,
            GroupedLocalMemoryTrainer=SyntheticTrainer,
            SafeLocalDistributedCheckpointer=SyntheticTrainer,
            create=True,
        ),
    ):
        phase5.overlay_config(config, args, contract, resume_checkpoint=resume_checkpoint)


@pytest.fixture
def geometry(tmp_path):
    root = tmp_path / "cache"
    _write(root, _manifest(root), _payload())
    cache = RoboCasaExactWindowCacheCatalog(root)
    return cache, PolicyContract.from_cache_catalog(cache)


def test_inherited32_overlay_saved_yaml_pickle_and_child_parser_are16(tmp_path, geometry):
    cache, contract = geometry
    args = _args(tmp_path)
    config = _config(args)
    assert server_chunk(config) == 32
    _overlay(config, args, contract)
    assert server_chunk(config) == 16
    metadata = describe_effective_data_source(config, contract, ttt_steps=args.t)
    assert metadata["action_horizon"] == metadata["chunk_length"] == cache.stats.chunk_length == 16
    assert metadata["observation_frames"] == 17
    assert metadata["local_ttt_tbptt_steps"] == 8  # T is independent of prediction H.
    assert metadata["raw_action_dim"] == metadata["state_dim"] == 15
    assert "not_training_producer" in metadata["legacy_dataloader_role"]
    for name in ("config.yaml", "config.pkl"):
        getattr(LazyConfig, "save_yaml" if name.endswith("yaml") else "save_pkl")(config, str(tmp_path / name))
    saved_yaml = yaml.safe_load((tmp_path / "config.yaml").read_text())
    with (tmp_path / "config.pkl").open("rb") as stream:
        saved_pickle = pickle.load(stream)  # Only this test's just-written config, never an external asset.
    for snapshot in (saved_yaml, saved_pickle):
        assert server_chunk(snapshot) == 16
        validate_effective_snapshot(snapshot, metadata)
    reader = RoboCasaExactWindowEpisodeReader(cache)
    window = reader.read_identity(cache.episodes[0].key, 0)
    assert len(window.window_frame_indices) == contract.observation_frames == 17
    action = torch.zeros(contract.action_horizon, contract.action_dim)
    state_and_action = torch.cat((torch.zeros(1, contract.state_dim), action))
    assert action.shape == (16, 15) and state_and_action.shape == (17, 15)
    plan = build_sequence_plan(contract.mode, contract.observation_frames, len(state_and_action))
    assert plan.condition_frame_indexes_vision == [0]
    assert plan.condition_frame_indexes_action == [0]
    assert plan.action_start_frame_offset == 0
    assert len(state_and_action) - len(plan.condition_frame_indexes_action) == contract.action_horizon
    print(
        "CONFIG_TRUTH_WITNESS "
        + json.dumps(
            {
                "before": 32,
                "yaml": server_chunk(saved_yaml),
                "pickle": server_chunk(saved_pickle),
                "cache": cache.stats.chunk_length,
                "action_shape": list(action.shape),
                "sequence_plan": plan.as_dict(),
                "observation_frames": len(window.window_frame_indices),
                "provenance": metadata,
            },
            sort_keys=True,
        )
    )


@pytest.mark.parametrize(
    "kind", ["alias_conflict", "other_dataset", "list_conflict", "string", "bool", "zero", "unsupported", "missing"]
)
def test_bad_or_ambiguous_fields_fail_before_mutation(tmp_path, geometry, kind):
    _, contract = geometry
    config = _config(_args(tmp_path))
    dataset = OmegaConf.select(config.dataloader_train, "dataloader.datasets.robocasa.dataset")
    if kind == "alias_conflict":
        dataset.num_action_per_chunk = 16
    elif kind == "other_dataset":
        config.dataloader_train.dataloader.datasets.other = {"dataset": {"chunk_length": 32}}
    elif kind == "list_conflict":
        config.dataloader_train.extra = [{"chunk_length": 16}]
    elif kind == "missing":
        del dataset.chunk_length
    else:
        dataset.chunk_length = {"string": "32", "bool": True, "zero": 0, "unsupported": 24}[kind]
    before = OmegaConf.to_container(config.dataloader_train, resolve=True)
    with pytest.raises(ValueError):
        synchronize_legacy_chunk_length(config, contract)
    assert OmegaConf.to_container(config.dataloader_train, resolve=True) == before


def test_matching_aliases_are_mirrored_and_drift_before_snapshot_is_rejected(tmp_path, geometry):
    _, contract = geometry
    config = _config(_args(tmp_path))
    dataset = OmegaConf.select(config.dataloader_train, "dataloader.datasets.robocasa.dataset")
    dataset.num_action_per_chunk = 32
    synchronize_legacy_chunk_length(config, contract)
    assert dataset.chunk_length == dataset.num_action_per_chunk == 16
    metadata = describe_effective_data_source(config, contract, ttt_steps=16)
    dataset.chunk_length = 32
    with pytest.raises(ValueError):
        validate_effective_snapshot(config, metadata)


def _report(config, args, contract):
    return {
        "training_identity": {"root": args.expected_root, "h_pred": 16},
        "config_digest": "d" * 64,
        "execution_stop_after_iter": 100,
        "checkpoint_load_path": "unused",
        "effective_data_source": describe_effective_data_source(config, contract, ttt_steps=args.t),
    }


def test_lifecycle_persists_provenance_and_same_job_resume_preserves_canonical(tmp_path, geometry):
    _, contract = geometry
    args = _args(tmp_path)
    config = _config(args)
    _overlay(config, args, contract)
    report = _report(config, args, contract)
    lifecycle = phase5.JobLifecycle(args, report)
    lifecycle.write_config(config, LazyConfig)
    lifecycle.close()
    path = phase5.job_namespace(args)
    canonical = {name: (path / name).read_bytes() for name in ("config.yaml", "config.pkl", phase5.JOB_IDENTITY)}
    launch = json.loads(next((path / "launches").glob("*/launch.json")).read_text())
    assert launch["effective_data_source"] == report["effective_data_source"]
    args.phase, args.formal_execution_until = "resume", 500
    restored = tmp_path / "synthetic_iteration100_not_a_dcp"
    report["checkpoint_load_path"], report["execution_stop_after_iter"] = str(restored), 500
    resumed_config = _config(args)
    _overlay(resumed_config, args, contract, resume_checkpoint=restored)
    assert phase5.check_job_namespace(args, report["training_identity"])["schema"] == "corrected_phase5_f0_v1"
    resumed = phase5.JobLifecycle(args, report)
    with patch.object(phase5, "_resume_checkpoint", return_value=restored):
        resumed.write_config(resumed_config, LazyConfig)
    resumed.close()
    assert {name: (path / name).read_bytes() for name in canonical} == canonical
    launches = [json.loads(p.read_text()) for p in (path / "launches").glob("*/launch.json")]
    assert len(launches) == 2
    assert {p["phase"] for p in launches} == {"fresh", "resume"}
    assert all(p["effective_data_source"] == report["effective_data_source"] for p in launches)
    print(
        "RESUME_CONFIG_WITNESS "
        + json.dumps(
            {
                "canonical_sha256": {name: hashlib.sha256(value).hexdigest() for name, value in canonical.items()},
                "launches": launches,
            },
            sort_keys=True,
        )
    )


@pytest.mark.parametrize("fault", ["live_drift", "yaml_drift", "yaml_conflict", "missing_provenance"])
def test_snapshot_drift_blocks_canonical_publication(tmp_path, geometry, fault):
    _, contract = geometry
    args = _args(tmp_path)
    config = _config(args)
    _overlay(config, args, contract)
    report = _report(config, args, contract)
    serializer = LazyConfig
    if fault == "live_drift":
        config.dataloader_train.dataloader.datasets.robocasa.dataset.chunk_length = 32
    elif fault == "missing_provenance":
        del report["effective_data_source"]
    else:

        def save_yaml(value, path):
            LazyConfig.save_yaml(value, path)
            payload = yaml.safe_load(Path(path).read_text())
            dataset = payload["dataloader_train"]["dataloader"]["datasets"]["robocasa"]["dataset"]
            dataset["chunk_length" if fault == "yaml_drift" else "num_action_per_chunk"] = 32
            Path(path).write_text(yaml.safe_dump(payload))

        serializer = SimpleNamespace(save_pkl=LazyConfig.save_pkl, save_yaml=save_yaml)
    lifecycle = phase5.JobLifecycle(args, report)
    try:
        with pytest.raises((ValueError, KeyError)):
            lifecycle.write_config(config, serializer)
    finally:
        lifecycle.close()
    path = phase5.job_namespace(args)
    assert not any((path / name).exists() for name in ("config.yaml", "config.pkl", phase5.JOB_IDENTITY))
    if fault in ("live_drift", "missing_provenance"):
        assert not path.exists()


def test_legacy_job_assets_and_other_recipe_remain_unchanged(tmp_path, geometry):
    _, contract = geometry
    args = _args(tmp_path)
    path = phase5.job_namespace(args)
    path.mkdir(parents=True)
    old_config = _config(args)
    LazyConfig.save_yaml(old_config, str(path / "config.yaml"))
    LazyConfig.save_pkl(old_config, str(path / "config.pkl"))
    (path / phase5.JOB_IDENTITY).write_text(json.dumps({"schema": "corrected_phase5_f0_v1", "root": "old_root"}))
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in path.iterdir()}
    candidate = deepcopy(old_config)
    synchronize_legacy_chunk_length(candidate, contract)
    assert server_chunk(candidate) == 16 and server_chunk(old_config) == 32
    args.phase = "resume"
    with pytest.raises(ValueError, match="identity"):
        phase5.check_job_namespace(args)
    assert {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in path.iterdir()} == before


@pytest.mark.parametrize("field,value", [("chunk_length", 32), ("fps", 19.0)])
def test_actual_cache_contract_rejects_wrong_geometry(tmp_path, field, value):
    root = tmp_path / "bad_cache"
    manifest = _manifest(root)
    manifest[field] = value
    _write(root, manifest)
    with pytest.raises(ValueError):
        PolicyContract.from_cache_catalog(RoboCasaExactWindowCacheCatalog(root))


def test_producer_provenance_names_match_actual_preflight_and_main_wiring():
    source = ast.parse((CHILD / "examples/psm_wma_robocasa_corrected_phase5.py").read_text())
    for function, required in (
        ("preflight", {"get_action_robocasa_exact_window_cached_sft_dataset", "describe_effective_data_source"}),
        ("main", {"ExactWindowSegmentProducer", "bind_grouped_stream"}),
    ):
        tree = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == function)
        calls = {
            node.func.id if isinstance(node.func, ast.Name) else node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, (ast.Name, ast.Attribute))
        }
        assert required <= calls
