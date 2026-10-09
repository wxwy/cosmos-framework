# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Cache-driven RoboCasa exact-window samples for the native Action SFT path."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from cosmos_framework.configs.base.experiment.sft.models.edge_model_config import EDGE_MODEL_CONFIG
from cosmos_framework.data.generator.action.datasets.action_sft_dataset import ActionSFTDataset
from cosmos_framework.data.generator.action.datasets.robocasa_exact_window_cache import (
    RoboCasaExactWindowCacheCatalog,
    RoboCasaExactWindowEpisodeReader,
)
from cosmos_framework.data.generator.action.datasets.robocasa_exact_window_policy import (
    CorrectedRoboCasaPolicyContract,
    OfficialRoboCasaPolicyAdapter,
)
from cosmos_framework.data.generator.action.datasets.robocasa_exact_window_source import RoboCasaExactWindowSourceReader
from cosmos_framework.data.generator.action.datasets.robocasa_lerobot_dataset import RoboCasaLeRobotDataset
from cosmos_framework.data.generator.action.datasets.robocasa_verified_index import VerifiedExactWindowIndex
from cosmos_framework.data.generator.action.utils.domain_utils import get_domain_id
from cosmos_framework.data.generator.action.utils.transforms import ActionTransformPipeline, VideoResize

_COMPOSITE_SHAPE = (3, 17, 256, 512)
_VIEW_DESCRIPTION = (
    "The left half is a third-person view of the scene. The right half is from the wrist-mounted camera."
)


class RoboCasaExactWindowCachedDataset(Dataset):
    """The cache defines membership; official source and policy helpers fill non-visual fields."""

    def __init__(
        self,
        catalog: RoboCasaExactWindowCacheCatalog,
        source_reader: RoboCasaExactWindowSourceReader,
        contract: CorrectedRoboCasaPolicyContract,
        *,
        cache_reader: RoboCasaExactWindowEpisodeReader | None = None,
    ) -> None:
        if source_reader.catalog is not catalog or source_reader.index.catalog is not catalog:
            raise ValueError("source reader 与 cache catalog 身份不一致")
        if contract.vae_encode_contract != catalog.vae_encode_contract:
            raise ValueError("policy VAE contract 与 cache manifest 不一致")
        self.catalog = catalog
        self.source_reader = source_reader
        self.cache_reader = cache_reader or RoboCasaExactWindowEpisodeReader(catalog)
        if self.cache_reader.catalog is not catalog:
            raise ValueError("cache reader 与 cache catalog 身份不一致")
        self.contract = contract
        self.policy_adapter = OfficialRoboCasaPolicyAdapter(contract)
        proxy = object.__new__(RoboCasaLeRobotDataset)
        proxy._use_base_action = True
        proxy._base_encoding = "raw"
        proxy._pose_convention = "backward_framewise"
        proxy._fps = contract.fps
        self._idle_proxy = proxy
        geometry = VideoResize()(dict(video=torch.zeros((3, 1, 256, 512), dtype=torch.uint8)), None)
        self._canvas_geometry = [int(value) for value in geometry["image_size"].reshape(-1).tolist()]
        spatial_factor = EDGE_MODEL_CONFIG["tokenizer"]["spatial_compression_factor"]
        target_h, target_w = self._canvas_geometry[:2]
        if (
            type(spatial_factor) is not int
            or spatial_factor <= 0
            or target_h % spatial_factor
            or target_w % spatial_factor
            or self.catalog.latent_shape[-2:] != (target_h // spatial_factor, target_w // spatial_factor)
        ):
            raise ValueError("cache latent spatial shape 与 placeholder padded canvas/tokenizer 不一致")

    def __len__(self) -> int:
        return self.catalog.stats.exact_window_count

    def get_shuffle_blocks(self) -> list[tuple[int, int]]:
        blocks = []
        offset = 0
        for episode in self.catalog.episodes:
            blocks.append((offset, episode.window_count))
            offset += episode.window_count
        if offset != len(self):
            raise ValueError("cache episode block 与 flat window count 不一致")
        return blocks

    def __getitem__(self, index: int) -> dict[str, Any]:
        key, start = self.source_reader.index[index]
        source = self.source_reader.read_at(index)
        policy = self.policy_adapter.convert(source)
        latent = self.cache_reader.read_window(key, start)
        identity = self.cache_reader.read_identity(key, start)
        if (
            source.key != key
            or source.start_frame != start
            or policy.key != key
            or policy.start_frame != start
            or identity.key != key
            or identity.start_frame != start
            or identity.global_row_indices != source.global_row_indices
            or identity.global_row_indices != policy.global_row_indices
            or policy.source_binding_digest != self.source_reader.source_binding_digest
            or source.source_binding_digest != self.source_reader.source_binding_digest
            or policy.task_class != key.task_class
            or source.task_class != key.task_class
            or policy.ai_caption != source.ai_caption
        ):
            raise ValueError(f"cache/source/policy exact window identity 不一致：{key}/{start}")
        if (
            tuple(latent.shape) != self.catalog.latent_shape
            or latent.dtype != torch.float32
            or not bool(torch.isfinite(latent).all())
            or tuple(policy.action_with_state15.shape) != (17, 15)
            or policy.action_with_state15.dtype != torch.float32
            or not bool(torch.isfinite(policy.action_with_state15).all())
        ):
            raise ValueError(f"cache latent 或 raw15 action 合同无效：{key}/{start}")
        idle_frames = RoboCasaLeRobotDataset._compute_idle_frames(self._idle_proxy, policy.action15)
        return {
            "ai_caption": policy.ai_caption,
            "video": torch.zeros(_COMPOSITE_SHAPE, dtype=torch.uint8),
            "video_latent": latent.contiguous(),
            "cached_latent_required": True,
            "action": policy.action_with_state15.contiguous(),
            "conditioning_fps": torch.tensor(self.contract.fps, dtype=torch.long),
            "mode": self.contract.mode,
            "domain_id": torch.tensor(get_domain_id("robocasa"), dtype=torch.long),
            "viewpoint": "concat_view",
            "additional_view_description": _VIEW_DESCRIPTION,
            "idle_frames": idle_frames,
            "task_class": key.task_class,
            "episode_index": key.episode_index,
            "start_frame": start,
            "global_row_indices": torch.tensor(identity.global_row_indices, dtype=torch.long),
            "window_frame_indices": torch.tensor(identity.window_frame_indices, dtype=torch.long),
            "latent_source_frame_indices": torch.tensor(identity.latent_source_frame_indices, dtype=torch.long),
            "cache_corpus_digest": self.catalog.corpus_digest,
            "source_binding_digest": self.source_reader.source_binding_digest,
        }

    def summary(self) -> dict[str, Any]:
        return {
            **self.catalog.stats.as_dict(),
            **self.source_reader.summary(),
            "H_pred": self.contract.action_horizon,
            "raw_action_dim": self.contract.action_dim,
            "state_dim": self.contract.state_dim,
            "vae_encode_contract": self.catalog.vae_encode_contract,
            "cached_latent_required": True,
            "online_vae_fallback": False,
            "placeholder_geometry": list(_COMPOSITE_SHAPE),
            "inner_collate_video_latent_abi": "Tensor[B,5,48,H,W]",
            "model_video_latent_abi": "list[B] of Tensor[1,5,48,H,W]",
            "transformed_canvas_geometry": list(self._canvas_geometry),
            "model_cache_hit_required": True,
        }


def get_action_robocasa_exact_window_cached_sft_dataset(
    *,
    cache_root: str | Path,
    source_root: str | Path,
    tokenizer_config: dict,
    cfg_dropout_rate: float = 0.1,
    iterable_shuffle: bool = False,
    episode_shuffle_seed: int = 42,
    catalog: RoboCasaExactWindowCacheCatalog | None = None,
    verified_index: VerifiedExactWindowIndex | None = None,
) -> Dataset:
    """Create the strict offline source/cache dataset and reuse the official SFT transform."""
    if tokenizer_config is None:
        raise ValueError("Corrected cached SFT requires a VLM text tokenizer config")
    if catalog is None:
        catalog = RoboCasaExactWindowCacheCatalog(cache_root, verified_index=verified_index)
    elif catalog.cache_root.resolve() != Path(cache_root).resolve():
        raise ValueError("shared cache catalog root 不匹配")
    if verified_index is not None:
        verified_index.check_catalog(catalog)
    contract = CorrectedRoboCasaPolicyContract.from_cache_catalog(catalog)
    source_reader = RoboCasaExactWindowSourceReader(catalog, source_root, verified_index=verified_index)
    raw = RoboCasaExactWindowCachedDataset(catalog, source_reader, contract)
    transform = ActionTransformPipeline(
        tokenizer_config=tokenizer_config,
        cfg_dropout_rate=cfg_dropout_rate,
        max_action_dim=contract.max_action_dim,
        append_viewpoint_info=True,
        append_duration_fps_timestamps=True,
        append_resolution_info=True,
        append_idle_frames=True,
        format_prompt_as_json=True,
    )
    sft = ActionSFTDataset(raw, transform, resolution=None)
    sft.summary = raw.summary
    if iterable_shuffle:
        from cosmos_framework.data.generator.action.datasets.action_sft_dataset import ActionIterableShuffleDataset

        iterable = ActionIterableShuffleDataset(sft, seed=episode_shuffle_seed)
        iterable.summary = raw.summary
        return iterable
    return sft
