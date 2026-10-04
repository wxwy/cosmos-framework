# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Synthetic CPU contract tests for official exact-window RoboCasa policy rows."""

from __future__ import annotations

import ast
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch
from omegaconf import OmegaConf

from cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_robocasa_edge import (
    action_policy_robocasa_edge,
)
from cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_robocasa_nano import (
    action_policy_robocasa_nano,
)
from cosmos_framework.data.generator.action.datasets import robocasa_exact_window_policy as policy
from cosmos_framework.data.generator.action.datasets.robocasa_exact_window_cache import (
    ExactWindowEpisodeKey,
    RoboCasaExactWindowCacheCatalog,
)
from cosmos_framework.data.generator.action.datasets.robocasa_exact_window_cache_test import _manifest, _payload, _write
from cosmos_framework.data.generator.action.datasets.robocasa_exact_window_source import ExactWindowRawSourceWindow
from cosmos_framework.data.generator.action.datasets.robocasa_lerobot_dataset import RoboCasaLeRobotDataset
from cosmos_framework.data.generator.action.utils.action_processing import ActionProcessor
from cosmos_framework.data.generator.action.utils.pose_utils import convert_rotation
from cosmos_framework.data.generator.action.utils.transforms import build_sequence_plan_from_mode
from cosmos_framework.data.generator.sequence_packing.sequence import PackedSequenceBuilder
from cosmos_framework.model.generator.algorithm.loss.flow_matching import compute_flow_matching_loss


def _contract() -> policy.CorrectedRoboCasaPolicyContract:
    return policy.CorrectedRoboCasaPolicyContract(
        vae_encode_contract={
            "compute_dtype": "torch.bfloat16",
            "encode_exact_durations": [17, 61],
            "encode_chunk_frames": {"256": 68},
        }
    )


def _source_window() -> ExactWindowRawSourceWindow:
    action = torch.zeros(16, 12)
    action[:, :4] = torch.tensor([0.11, -0.22, 0.33, -0.44])
    action[:, 4] = torch.tensor([1.0, -1.0] * 8)
    action[:, 5:8] = torch.tensor([0.15, -0.25, 0.35])
    action[:, 8:11] = torch.tensor([0.31, -0.42, 0.23])
    action[:, 11] = 0.57
    state = torch.zeros(17, 16)
    state[:, :7] = torch.tensor([4.0, -5.0, 6.0, 0.0, 0.0, 0.0, 1.0])
    state[0, 7:10] = torch.tensor([-0.21, 0.32, 0.43])
    state[0, 10:14] = torch.tensor([0.0, 0.0, 0.38268343, 0.92387953])
    state[0, 14:16] = torch.tensor([0.17, -0.06])
    state[1:, 7:10] = 9.0
    state[1:, 10:14] = torch.tensor([0.0, 0.0, 0.0, 1.0])
    state[1:, 14:16] = torch.tensor([0.9, -0.9])
    return ExactWindowRawSourceWindow(
        key=ExactWindowEpisodeKey("Pick Mug", "Pick_Mug", 7),
        start_frame=3,
        global_row_indices=tuple(range(103, 120)),
        action12=action,
        state16=state,
        ai_caption="pick up the mug",
        task_index=4,
        task_class="Pick Mug",
        source_binding_digest="source-binding-test",
        source_data_file="data/chunk-000/file-000.parquet",
    )


def _catalog(tmp_path: Path, *, mutate=None) -> RoboCasaExactWindowCacheCatalog:
    manifest = _manifest(tmp_path)
    if mutate is not None:
        mutate(manifest)
    _write(tmp_path, manifest, _payload())
    return RoboCasaExactWindowCacheCatalog(tmp_path)


@pytest.mark.level(0)
def test_official_raw15_state15_values_and_identity() -> None:
    source = _source_window()
    raw_action = source.action12.clone()
    raw_state = source.state16.clone()
    output = policy.OfficialRoboCasaPolicyAdapter(_contract()).convert(source)

    assert output.action15.shape == (16, 15)
    assert output.state15.shape == (15,)
    assert output.action_with_state15.shape == (17, 15)
    for value in (output.action15, output.state15, output.action_with_state15):
        assert value.dtype == torch.float32 and value.is_contiguous() and torch.isfinite(value).all()
    torch.testing.assert_close(output.action15[:, :5], raw_action[:, :5], rtol=0, atol=0)
    torch.testing.assert_close(output.action15[:, 5:8], raw_action[:, 5:8], rtol=0, atol=0)
    torch.testing.assert_close(output.action15[:, 14], raw_action[:, 11], rtol=0, atol=0)
    torch.testing.assert_close(output.state15[:5], torch.zeros(5), rtol=0, atol=0)
    torch.testing.assert_close(output.state15[5:8], raw_state[0, 7:10], rtol=0, atol=0)
    torch.testing.assert_close(output.state15[14], raw_state[0, 14] - raw_state[0, 15], rtol=0, atol=0)
    torch.testing.assert_close(output.action_with_state15[0], output.state15, rtol=0, atol=0)
    torch.testing.assert_close(output.action_with_state15[1:], output.action15, rtol=0, atol=0)
    torch.testing.assert_close(source.action12, raw_action, rtol=0, atol=0)
    torch.testing.assert_close(source.state16, raw_state, rtol=0, atol=0)
    assert (output.key, output.start_frame, output.global_row_indices) == (
        source.key,
        source.start_frame,
        source.global_row_indices,
    )
    assert (output.ai_caption, output.task_index, output.task_class) == (
        source.ai_caption,
        source.task_index,
        source.task_class,
    )
    assert (output.source_binding_digest, output.source_data_file) == (
        source.source_binding_digest,
        source.source_data_file,
    )


@pytest.mark.level(0)
def test_rotation_matrix_semantics_and_current_state_only() -> None:
    source = _source_window()
    output = policy.OfficialRoboCasaPolicyAdapter(_contract()).convert(source)
    expected_action_matrix = convert_rotation(source.action12[:, 8:11], "axisangle", "matrix")
    actual_action_matrix = convert_rotation(output.action15[:, 8:14], "rot6d", "matrix")
    torch.testing.assert_close(actual_action_matrix, expected_action_matrix, rtol=1e-5, atol=1e-5)
    expected_state_matrix = convert_rotation(source.state16[0, 10:14].unsqueeze(0), "quat_xyzw", "matrix")
    actual_state_matrix = convert_rotation(output.state15[8:14].unsqueeze(0), "rot6d", "matrix")
    torch.testing.assert_close(actual_state_matrix, expected_state_matrix, rtol=1e-5, atol=1e-5)
    changed = deepcopy(source)
    changed.state16[1:] = torch.randn_like(changed.state16[1:])
    other = policy.OfficialRoboCasaPolicyAdapter(_contract()).convert(changed)
    torch.testing.assert_close(other.state15, output.state15, rtol=0, atol=0)


@pytest.mark.level(0)
@pytest.mark.parametrize(
    "field,invalid",
    [
        ("action12", torch.zeros(32, 12)),
        ("action12", torch.zeros(16, 12, dtype=torch.int64)),
        ("action12", torch.full((16, 12), float("nan"))),
        ("state16", torch.zeros(16, 16)),
        ("state16", torch.zeros(17, 16, dtype=torch.int64)),
        ("state16", torch.full((17, 16), float("inf"))),
    ],
)
def test_invalid_source_fails_closed(field: str, invalid: torch.Tensor) -> None:
    source = _source_window()
    object.__setattr__(source, field, invalid)
    with pytest.raises(ValueError, match=field):
        policy.OfficialRoboCasaPolicyAdapter(_contract()).convert(source)


@pytest.mark.level(0)
def test_official_private_helpers_called_and_errors_propagate() -> None:
    action_helper = RoboCasaLeRobotDataset._build_frame_wise_action
    state_helper = RoboCasaLeRobotDataset._build_initial_state
    action_spy = Mock(wraps=action_helper)
    state_spy = Mock(wraps=state_helper)
    with (
        patch.object(RoboCasaLeRobotDataset, "_build_frame_wise_action", action_spy),
        patch.object(RoboCasaLeRobotDataset, "_build_initial_state", state_spy),
    ):
        policy.OfficialRoboCasaPolicyAdapter(_contract()).convert(_source_window())
    assert action_spy.call_count == 1 and state_spy.call_count == 1
    assert isinstance(action_spy.call_args.args[0], RoboCasaLeRobotDataset)
    assert action_spy.call_args.args[0]._chunk_length == 16
    with patch.object(RoboCasaLeRobotDataset, "_build_frame_wise_action", side_effect=RuntimeError("official drift")):
        with pytest.raises(RuntimeError, match="official drift"):
            policy.OfficialRoboCasaPolicyAdapter(_contract()).convert(_source_window())
    with patch.object(RoboCasaLeRobotDataset, "_build_initial_state", side_effect=RuntimeError("official drift")):
        with pytest.raises(RuntimeError, match="official drift"):
            policy.OfficialRoboCasaPolicyAdapter(_contract()).convert(_source_window())


@pytest.mark.level(0)
def test_policy_module_uses_no_copied_rotation_math_or_dataset_constructor() -> None:
    tree = ast.parse(Path(policy.__file__).read_text(encoding="utf-8"))
    imported = {
        alias.name for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names
    }
    called = {node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert "convert_rotation" not in imported | called
    assert "RoboCasaLeRobotDataset" not in called
    assert not any(isinstance(node, ast.FunctionDef) and "rot" in node.name.lower() for node in ast.walk(tree))


@pytest.mark.level(0)
def test_official_wam_packing_masks_noise_and_flow_loss() -> None:
    plan = build_sequence_plan_from_mode("wam", video_length=17, action_length=17)
    assert plan.condition_frame_indexes_vision == [0]
    assert plan.condition_frame_indexes_action == [0]
    action = policy.OfficialRoboCasaPolicyAdapter(_contract()).convert(_source_window()).action_with_state15
    builder = PackedSequenceBuilder()
    builder.pack_action_tokens(
        action,
        plan.condition_frame_indexes_action,
        input_timestep=0.5,
        action_start_frame_offset=plan.action_start_frame_offset,
    )
    assert builder.action is not None
    mask = builder.action.condition_mask[0]
    torch.testing.assert_close(mask[:, 0], torch.tensor([1.0] + [0.0] * 16), rtol=0, atol=0)
    torch.testing.assert_close(builder.action.noisy_frame_indexes[0], torch.arange(1, 17))
    assert builder.action.mse_loss_indexes == list(range(1, 17))
    sigma = torch.full((17, 1), 0.5) * (1 - mask)
    assert sigma[0].item() == 0 and bool(torch.all(sigma[1:] > 0))

    class UnitWeightFlow:
        def train_time_weight(self, timesteps, tensor_kwargs):
            return torch.ones_like(timesteps, **tensor_kwargs)

    target = torch.zeros(17, 15)
    only_state_error = target.clone()
    only_state_error[0] = 10
    only_action_error = target.clone()
    only_action_error[1:] = 1
    common = dict(
        target=[target],
        condition_mask=[mask],
        timesteps=torch.zeros(1, 1),
        has_valid_tokens=True,
        rectified_flow=UnitWeightFlow(),
        tensor_kwargs_fp32={"dtype": torch.float32},
        raw_action_dim=[torch.tensor(15)],
    )
    state_loss, _ = compute_flow_matching_loss(pred=[only_state_error], **common)
    action_loss, _ = compute_flow_matching_loss(pred=[only_action_error], **common)
    assert state_loss.item() == 0
    assert action_loss.item() > 0


@pytest.mark.level(0)
def test_official_action_processor_no_normalization_and_pad64() -> None:
    action = policy.OfficialRoboCasaPolicyAdapter(_contract()).convert(_source_window()).action_with_state15
    processed = ActionProcessor(max_action_dim=64).preprocess_action({}, action, action_normalizer=None)
    torch.testing.assert_close(processed["action_raw"], action, rtol=0, atol=0)
    assert processed["raw_action_dim"].item() == 15
    assert processed["action"].shape == (17, 64)
    torch.testing.assert_close(processed["action"][:, :15], action, rtol=0, atol=0)
    torch.testing.assert_close(processed["action"][:, 15:], torch.zeros(17, 49), rtol=0, atol=0)
    assert processed["action_processing_record"].action_normalizer is None


@pytest.mark.level(0)
def test_manifest_contract_preserves_full_authority_and_resolves_edge(tmp_path: Path) -> None:
    catalog = _catalog(
        tmp_path, mutate=lambda manifest: manifest["vae_encode_contract"].update({"vae_id": "test-only"})
    )
    contract = policy.CorrectedRoboCasaPolicyContract.from_cache_catalog(catalog)
    assert (contract.fps, contract.action_horizon, contract.chunk_length, contract.observation_frames) == (
        20.0,
        16,
        16,
        17,
    )
    assert (contract.source_action_dim, contract.action_dim, contract.source_state_dim, contract.state_dim) == (
        12,
        15,
        16,
        15,
    )
    assert (contract.camera_set, contract.use_state, contract.use_base_action, contract.base_encoding) == (
        "left_wrist",
        True,
        True,
        "raw",
    )
    assert (contract.action_normalization, contract.mode, contract.max_action_dim, contract.replan_default) == (
        None,
        "wam",
        64,
        16,
    )
    assert contract.vae_encode_contract == catalog.vae_encode_contract
    candidate = action_policy_robocasa_edge["model"]["config"]["tokenizer"]
    original = OmegaConf.to_container(candidate, resolve=False)
    assert original["vae_path"] == "${oc.env:WAN_VAE_PATH}"
    assert original["encode_exact_durations"] == [33]
    resolved = contract.resolve_tokenizer_config(candidate)
    assert resolved["encode_exact_durations"] == catalog.vae_encode_contract["encode_exact_durations"]
    assert OmegaConf.to_container(candidate, resolve=False) == original
    assert resolved is not candidate
    assert resolved["encode_chunk_frames"] is not original["encode_chunk_frames"]
    assert {key: value for key, value in resolved.items() if key != "encode_exact_durations"} == {
        key: value for key, value in original.items() if key != "encode_exact_durations"
    }
    contract.validate_tokenizer_config(resolved)
    with pytest.raises(TypeError):
        contract.resolve_tokenizer_config()
    contract.validate_replan_steps(1)
    contract.validate_replan_steps(16)
    with pytest.raises(ValueError, match="replan_steps"):
        contract.validate_replan_steps(17)


@pytest.mark.level(0)
@pytest.mark.parametrize(
    "change,match",
    [
        (lambda m: m["vae_encode_contract"].update(encode_exact_durations=[33]), "含17"),
        (lambda m: m["vae_encode_contract"].update(compute_dtype="torch.float32"), "compute_dtype"),
        (lambda m: m["vae_encode_contract"].update(encode_chunk_frames={"999": 8}), "fixed Edge"),
        (lambda m: m["vae_encode_contract"].update(encode_chunk_frames={"256": 64}), "fixed Edge"),
    ],
)
def test_manifest_contract_rejects_incompatible_cache(tmp_path: Path, change, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        policy.CorrectedRoboCasaPolicyContract.from_cache_catalog(_catalog(tmp_path, mutate=change))


@pytest.mark.level(0)
def test_fixed_edge_extra_capability_key_allowed(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path, mutate=lambda m: m["vae_encode_contract"].update(encode_chunk_frames={"256": 68}))
    contract = policy.CorrectedRoboCasaPolicyContract.from_cache_catalog(catalog)
    assert contract.vae_encode_contract["encode_chunk_frames"] == {"256": 68}
    resolved = contract.resolve_tokenizer_config(action_policy_robocasa_edge["model"]["config"]["tokenizer"])
    assert "480" in resolved["encode_chunk_frames"]
    contract.validate_tokenizer_config(resolved)


@pytest.mark.level(0)
def test_manifest_duration_authority_survives_public_and_source_mutation() -> None:
    source = {
        "compute_dtype": "torch.bfloat16",
        "encode_exact_durations": [17, 61],
        "encode_chunk_frames": {"256": 68},
    }
    contract = policy.CorrectedRoboCasaPolicyContract(vae_encode_contract=source)
    source["encode_exact_durations"][:] = [33]
    source["compute_dtype"] = "torch.float32"
    contract.vae_encode_contract["encode_exact_durations"][:] = [33]
    contract.vae_encode_contract["compute_dtype"] = "torch.float32"
    candidate = action_policy_robocasa_edge["model"]["config"]["tokenizer"]
    original_candidate = OmegaConf.to_container(candidate, resolve=False)
    resolved = contract.resolve_tokenizer_config(candidate)
    assert type(resolved["encode_exact_durations"]) is list
    assert resolved["encode_exact_durations"] == [17, 61]
    assert OmegaConf.to_container(candidate, resolve=False) == original_candidate
    assert {key: value for key, value in resolved.items() if key != "encode_exact_durations"} == {
        key: value for key, value in original_candidate.items() if key != "encode_exact_durations"
    }
    contract.validate_tokenizer_config(resolved)
    with pytest.raises(ValueError, match="encode_exact_durations"):
        contract.validate_tokenizer_config({**resolved, "encode_exact_durations": [33]})
    with pytest.raises(ValueError, match="compute_dtype"):
        contract.validate_tokenizer_config({**resolved, "compute_dtype": "torch.float32"})


@pytest.mark.level(0)
def test_manifest_chunk_authority_survives_public_and_source_mutation() -> None:
    source = {
        "compute_dtype": "torch.bfloat16",
        "encode_exact_durations": [17, 61],
        "encode_chunk_frames": {"256": 68},
    }
    contract = policy.CorrectedRoboCasaPolicyContract(vae_encode_contract=source)
    source["encode_chunk_frames"].clear()
    source["encode_chunk_frames"]["480"] = 24
    contract.vae_encode_contract["encode_chunk_frames"].clear()
    contract.vae_encode_contract["encode_chunk_frames"]["480"] = 24
    candidate = action_policy_robocasa_edge["model"]["config"]["tokenizer"]
    resolved = contract.resolve_tokenizer_config(candidate)
    assert resolved["encode_chunk_frames"]["256"] == 68
    contract.validate_tokenizer_config(resolved)
    missing_required = deepcopy(resolved)
    del missing_required["encode_chunk_frames"]["256"]
    with pytest.raises(ValueError, match="manifest"):
        contract.validate_tokenizer_config(missing_required)
    with pytest.raises(ValueError, match="manifest"):
        contract.resolve_tokenizer_config(missing_required)


@pytest.mark.level(0)
def test_legacy_nano_edge_and_runtime32_rejected() -> None:
    contract = _contract()
    dataset = action_policy_robocasa_nano["dataloader_train"]["dataloader"]["datasets"]["robocasa"]["dataset"]
    with pytest.raises(ValueError, match="chunk_length"):
        contract.validate_dataset_config(dataset)
    edge_dataset = action_policy_robocasa_edge["dataloader_train"]["dataloader"]["datasets"]["robocasa"]["dataset"]
    with pytest.raises(ValueError, match="chunk_length"):
        contract.validate_dataset_config(edge_dataset)
    edge_tokenizer = action_policy_robocasa_edge["model"]["config"]["tokenizer"]
    with pytest.raises(ValueError, match="encode_exact_durations"):
        contract.validate_tokenizer_config(edge_tokenizer)
    with pytest.raises(ValueError, match="action_horizon"):
        contract.validate_runtime_config(
            {"action_horizon": 32, "chunk_length": 32, "observation_frames": 33, "replan_steps": 16}
        )
    with pytest.raises(ValueError, match="chunk_length"):
        contract.validate_runtime_config(
            {"action_horizon": 16, "chunk_length": 32, "observation_frames": 17, "replan_steps": 16}
        )
    with pytest.raises(ValueError, match="encode_exact_durations"):
        contract.validate_tokenizer_config(
            {**contract.resolve_tokenizer_config(edge_tokenizer), "encode_exact_durations": [17]}
        )


@pytest.mark.level(0)
def test_resolved_dataset_runtime_and_tokenizer_must_match_manifest() -> None:
    contract = _contract()
    dataset = {
        key: getattr(contract, key)
        for key in (
            "fps",
            "chunk_length",
            "camera_set",
            "use_state",
            "use_base_action",
            "base_encoding",
            "action_normalization",
            "mode",
            "max_action_dim",
        )
    }
    contract.validate_dataset_config(dataset)
    contract.validate_dataset_config({**dataset, "fps": 20})
    contract.validate_runtime_config(
        {"action_horizon": 16, "chunk_length": 16, "observation_frames": 17, "replan_steps": 8}
    )
    candidate = action_policy_robocasa_edge["model"]["config"]["tokenizer"]
    resolved = contract.resolve_tokenizer_config(candidate)
    resolved["encode_chunk_frames"]["256"] = 64
    with pytest.raises(ValueError, match="encode_chunk_frames"):
        contract.validate_tokenizer_config(resolved)
    resolved = contract.resolve_tokenizer_config(candidate)
    resolved["encode_chunk_frames"]["480"] = 28
    with pytest.raises(ValueError, match="fixed Edge"):
        contract.validate_tokenizer_config(resolved)
    resolved = contract.resolve_tokenizer_config(candidate)
    resolved["compute_dtype"] = "torch.float32"
    with pytest.raises(ValueError, match="compute_dtype"):
        contract.validate_tokenizer_config(resolved)
    invalid_candidate = OmegaConf.to_container(candidate, resolve=False)
    invalid_candidate["encode_chunk_frames"] = {"480": 28}
    with pytest.raises(ValueError, match="fixed Edge"):
        contract.resolve_tokenizer_config(invalid_candidate)
