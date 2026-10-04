# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Official RoboCasa action/state bridge for cache-driven exact windows."""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Mapping

import torch
from omegaconf import DictConfig, OmegaConf

from cosmos_framework.configs.base.experiment.sft.models.edge_model_config import EDGE_MODEL_CONFIG
from cosmos_framework.data.generator.action.datasets.robocasa_exact_window_cache import (
    ExactWindowEpisodeKey,
    RoboCasaExactWindowCacheCatalog,
)
from cosmos_framework.data.generator.action.datasets.robocasa_exact_window_source import ExactWindowRawSourceWindow
from cosmos_framework.data.generator.action.datasets.robocasa_lerobot_dataset import RoboCasaLeRobotDataset


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} 必须是 mapping")
    return value


def _exact(config: Mapping[str, Any], name: str, expected: object) -> None:
    value = config.get(name)
    if type(value) is not type(expected) or value != expected:
        raise ValueError(f"{name} 不匹配：{value!r} != {expected!r}")


def _float_tensor(value: object, shape: tuple[int, int], label: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or not value.is_floating_point():
        raise ValueError(f"{label} 必须是浮点 tensor {shape}")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{label} 包含非有限值")
    return value.to(dtype=torch.float32).contiguous()


@dataclass(frozen=True)
class ExactWindowPolicyActionWindow:
    key: ExactWindowEpisodeKey
    start_frame: int
    global_row_indices: tuple[int, ...]
    action15: torch.Tensor
    state15: torch.Tensor
    action_with_state15: torch.Tensor
    ai_caption: str
    task_index: int
    task_class: str
    source_binding_digest: str
    source_data_file: str


@dataclass(frozen=True)
class CorrectedRoboCasaPolicyContract:
    """Phase2 resolved geometry and manifest VAE authority; T is independent."""

    vae_encode_contract: Mapping[str, Any]
    _vae_compute_dtype: str = field(init=False, repr=False)
    _vae_exact_durations: tuple[int, ...] = field(init=False, repr=False)
    _vae_chunk_frames: tuple[tuple[str, int], ...] = field(init=False, repr=False)
    fps: float = 20.0
    action_horizon: int = 16
    chunk_length: int = 16
    observation_frames: int = 17
    source_action_dim: int = 12
    action_dim: int = 15
    source_state_dim: int = 16
    state_dim: int = 15
    max_action_dim: int = 64
    camera_set: str = "left_wrist"
    use_state: bool = True
    use_base_action: bool = True
    base_encoding: str = "raw"
    action_normalization: None = None
    mode: str = "wam"
    replan_default: int = 16

    def __post_init__(self) -> None:
        for name, expected in (
            ("fps", 20.0),
            ("action_horizon", 16),
            ("chunk_length", 16),
            ("observation_frames", 17),
            ("source_action_dim", 12),
            ("action_dim", 15),
            ("source_state_dim", 16),
            ("state_dim", 15),
            ("max_action_dim", 64),
            ("camera_set", "left_wrist"),
            ("use_state", True),
            ("use_base_action", True),
            ("base_encoding", "raw"),
            ("action_normalization", None),
            ("mode", "wam"),
            ("replan_default", 16),
        ):
            _exact(vars(self), name, expected)
        contract = _mapping(self.vae_encode_contract, "vae_encode_contract")
        _exact(contract, "compute_dtype", "torch.bfloat16")
        durations = contract.get("encode_exact_durations")
        if (
            not isinstance(durations, list)
            or not durations
            or any(type(value) is not int or value <= 0 for value in durations)
            or len(set(durations)) != len(durations)
            or 17 not in durations
        ):
            raise ValueError("encode_exact_durations 必须是含17的无重复正整数列表")
        chunks = _mapping(contract.get("encode_chunk_frames"), "encode_chunk_frames")
        if not chunks or any(
            type(key) is not str or not key or type(value) is not int or value <= 0 for key, value in chunks.items()
        ):
            raise ValueError("encode_chunk_frames 无效")
        fixed = _mapping(EDGE_MODEL_CONFIG["tokenizer"]["encode_chunk_frames"], "fixed Edge encode_chunk_frames")
        for key, value in chunks.items():
            if key not in fixed or fixed[key] != value:
                raise ValueError(f"fixed Edge tokenizer 不支持 encode_chunk_frames[{key!r}]={value}")
        object.__setattr__(self, "_vae_compute_dtype", contract["compute_dtype"])
        object.__setattr__(self, "_vae_exact_durations", tuple(durations))
        object.__setattr__(self, "_vae_chunk_frames", tuple(chunks.items()))
        object.__setattr__(self, "vae_encode_contract", deepcopy(dict(contract)))

    @classmethod
    def from_cache_catalog(cls, catalog: RoboCasaExactWindowCacheCatalog) -> CorrectedRoboCasaPolicyContract:
        if not isinstance(catalog, RoboCasaExactWindowCacheCatalog):
            raise TypeError("catalog 必须是 RoboCasaExactWindowCacheCatalog")
        if catalog.stats.chunk_length != 16 or catalog.stats.fps != 20.0:
            raise ValueError("cache chunk_length/fps 不符合 H_pred16/20fps")
        return cls(vae_encode_contract=catalog.vae_encode_contract)

    def validate_replan_steps(self, steps: int) -> None:
        if type(steps) is not int or not 1 <= steps <= self.action_horizon:
            raise ValueError(f"replan_steps 必须在 1..{self.action_horizon}：{steps!r}")

    def validate_dataset_config(self, config: Mapping[str, Any]) -> None:
        config = _mapping(config, "dataset config")
        fps = config.get("fps")
        if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps != self.fps:
            raise ValueError(f"fps 不匹配：{fps!r} != {self.fps!r}")
        for name in (
            "chunk_length",
            "camera_set",
            "use_state",
            "use_base_action",
            "base_encoding",
            "action_normalization",
            "mode",
            "max_action_dim",
        ):
            _exact(config, name, getattr(self, name))

    def validate_runtime_config(self, config: Mapping[str, Any]) -> None:
        config = _mapping(config, "runtime config")
        for name in ("action_horizon", "chunk_length", "observation_frames"):
            _exact(config, name, getattr(self, name))
        self.validate_replan_steps(config.get("replan_steps"))

    def resolve_tokenizer_config(self, candidate_config: Mapping[str, Any]) -> dict[str, Any]:
        candidate = deepcopy(_mapping(candidate_config, "candidate tokenizer config"))
        tokenizer = (
            OmegaConf.to_container(candidate, resolve=False) if isinstance(candidate, DictConfig) else dict(candidate)
        )
        tokenizer["encode_exact_durations"] = list(self._vae_exact_durations)
        self.validate_tokenizer_config(tokenizer)
        return tokenizer

    def validate_tokenizer_config(self, config: Mapping[str, Any]) -> None:
        config = _mapping(config, "tokenizer config")
        durations = config.get("encode_exact_durations")
        if type(durations) is not list or durations != list(self._vae_exact_durations):
            raise ValueError("resolved tokenizer encode_exact_durations 与 manifest 完整列表不一致")
        chunks = _mapping(config.get("encode_chunk_frames"), "resolved tokenizer encode_chunk_frames")
        fixed = EDGE_MODEL_CONFIG["tokenizer"]["encode_chunk_frames"]
        for key, value in chunks.items():
            if key not in fixed or type(value) is not type(fixed[key]) or value != fixed[key]:
                raise ValueError(f"resolved tokenizer encode_chunk_frames[{key!r}] 不属于 fixed Edge 能力")
        for key, value in self._vae_chunk_frames:
            if key not in chunks or chunks[key] != value:
                raise ValueError(f"resolved tokenizer encode_chunk_frames[{key!r}] 与 manifest 不一致")
        if "compute_dtype" in config:
            _exact(config, "compute_dtype", self._vae_compute_dtype)


class OfficialRoboCasaPolicyAdapter:
    """Invoke the official private conversion helpers without constructing a dataset."""

    def __init__(self, contract: CorrectedRoboCasaPolicyContract) -> None:
        self.contract = contract

    def convert(self, window: ExactWindowRawSourceWindow) -> ExactWindowPolicyActionWindow:
        if not isinstance(window, ExactWindowRawSourceWindow):
            raise TypeError("window 必须是 ExactWindowRawSourceWindow")
        action12 = _float_tensor(
            window.action12, (self.contract.chunk_length, self.contract.source_action_dim), "action12"
        )
        state16 = _float_tensor(
            window.state16, (self.contract.observation_frames, self.contract.source_state_dim), "state16"
        )
        proxy = object.__new__(RoboCasaLeRobotDataset)
        proxy._chunk_length = self.contract.chunk_length
        arm10 = _float_tensor(
            RoboCasaLeRobotDataset._build_frame_wise_action(proxy, action12.clone()),
            (self.contract.chunk_length, 10),
            "official arm10",
        )
        state10 = RoboCasaLeRobotDataset._build_initial_state(proxy, state16.clone())
        if not isinstance(state10, torch.Tensor) or state10.shape != (10,) or not state10.is_floating_point():
            raise ValueError("official state10 shape/dtype 无效")
        if not bool(torch.isfinite(state10).all()):
            raise ValueError("official state10 包含非有限值")
        action15 = torch.cat((action12[:, :5], arm10), dim=-1).contiguous()
        state15 = torch.cat((torch.zeros(5, dtype=torch.float32, device=state10.device), state10.float())).contiguous()
        action_with_state15 = torch.cat((state15.unsqueeze(0), action15), dim=0).contiguous()
        return ExactWindowPolicyActionWindow(
            key=window.key,
            start_frame=window.start_frame,
            global_row_indices=window.global_row_indices,
            action15=action15,
            state15=state15,
            action_with_state15=action_with_state15,
            ai_caption=window.ai_caption,
            task_index=window.task_index,
            task_class=window.task_class,
            source_binding_digest=window.source_binding_digest,
            source_data_file=window.source_data_file,
        )
