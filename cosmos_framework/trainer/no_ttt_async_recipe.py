# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Offline worker-owned Source/Cache/Transform recipe for the No-TTT pipeline.

The worker prepares the deterministic conditioned branch and the same tokenizer's
empty-caption tokens. Only the trainer consumes the original CFG random draw,
at the original consumer position. There is no reseeding, RNG prediction, model,
frontier owner or instantiated tokenizer in the serialized recipe.
"""

from __future__ import annotations

import importlib
import math
import os
import pickle
import time
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any

import torch


def _known_pipeline(producer, *, g2=False):
    from cosmos_framework.data.generator.action.utils.cached_pixel_geometry import CachedGeometryResize
    from cosmos_framework.data.generator.action.utils.json_formatter import ActionPromptJsonFormatter
    from cosmos_framework.data.generator.action.utils.transforms import ActionTransformPipeline, VideoResize
    from cosmos_framework.data.generator.augmentors.text_tokenizer import TextTokenizerTransform

    wrapped = producer.catalog.wrapped_sft
    transform = wrapped._transform
    tokenizer = getattr(transform, "text_tokenizer", None)
    kwargs = getattr(wrapped, "_no_ttt_transform_recipe", None)
    if (
        type(transform) is not ActionTransformPipeline
        or type(tokenizer) is not TextTokenizerTransform
        or tokenizer.tokenize_separately
        or type(transform.prompt_json_formatter) is not ActionPromptJsonFormatter
        or transform.idle_frames_augmentor is not None
        or wrapped._resolution is not None
        or producer.transform_rng_mode != "legacy_global"
        or not isinstance(kwargs, dict)
        or kwargs.get("format_prompt_as_json") is not True
        or tokenizer.input_keys != [transform.caption_key]
        or len(tokenizer.output_keys) != 1
        or tokenizer.use_system_prompt
        or (transform.static_task_text_recipe is not None) != g2
    ):
        raise ValueError("async No-TTT requires the known JSON ActionTransformPipeline and legacy CFG RNG")
    rate = tokenizer.cfg_dropout_rate
    if (
        type(rate) not in (float, int)
        or not math.isfinite(rate)
        or not 0 <= rate <= 1
        or kwargs.get("cfg_dropout_rate") != rate
        or kwargs.get("caption_key", "ai_caption") != transform.caption_key
        or kwargs.get("text_token_key", "text_token_ids") != tokenizer.output_keys[0]
        or kwargs.get("max_action_dim") != transform.max_action_dim
        or kwargs.get("tokenizer_config") != tokenizer.args["tokenizer_config"]
    ):
        raise ValueError("async No-TTT transform reconstruction metadata changed")
    # Construct only the cheap processor-free deterministic skeleton. Compare
    # every constructor-owned field rather than trusting copied kwargs alone.
    expected_kwargs = deepcopy(kwargs)
    expected_kwargs["tokenizer_config"] = None
    static_recipe = expected_kwargs.pop("static_task_text_recipe", None)
    if g2:
        from cosmos_framework.data.generator.action.utils.static_task_text import StaticTaskTextRecipe

        if (
            not isinstance(static_recipe, dict)
            or StaticTaskTextRecipe(**static_recipe) != transform.static_task_text_recipe
            or transform.caption_key != "ai_caption"
            or tokenizer.output_keys != ["text_token_ids"]
        ):
            raise ValueError("async G2 static text reconstruction differs from the parent")
    expected = ActionTransformPipeline(**expected_kwargs)
    for name in (
        "caption_key",
        "video_temporal_downsample",
        "max_action_dim",
        "action_channel_masking",
        "enable_mode_specific_prompt",
        "append_action_caption_semantics",
        "pad_keys",
        "keep_aspect_ratio",
    ):
        if getattr(transform, name) != getattr(expected, name):
            raise ValueError(f"async No-TTT transform reconstruction field changed: {name}")
    for name in ("action_processor", "prompt_json_formatter"):
        actual_value, expected_value = getattr(transform, name), getattr(expected, name)
        if type(actual_value) is not type(expected_value) or vars(actual_value) != vars(expected_value):
            raise ValueError(f"async No-TTT transform reconstruction field changed: {name}")
    if any(
        getattr(transform, name) is not None
        for name in ("viewpoint_augmentor", "duration_fps_augmentor", "resolution_info_augmentor")
    ):
        raise ValueError("async No-TTT JSON pipeline gained a stochastic/plain-caption augmentor")
    resize = transform.video_resize
    if producer.catalog.raw.compact_cached_video:
        if (
            type(resize) is not CachedGeometryResize
            or resize.canvas != tuple(producer.catalog.raw._canvas_geometry)
            or type(resize.original) is not VideoResize
        ):
            raise ValueError("async No-TTT compact geometry reconstruction changed")
        resize = resize.original
    if type(resize) is not VideoResize or vars(resize) != vars(expected.video_resize):
        raise ValueError("async No-TTT resize reconstruction changed")
    return wrapped, transform, tokenizer, kwargs


@dataclass(frozen=True)
class G2WorkerBinding:
    factory: str
    kwargs: dict[str, Any]
    recipe: Any
    split_identity: str
    snapshot_identity: str

    @classmethod
    def from_producer(cls, producer, factory, kwargs):
        from cosmos_framework.data.generator.action.datasets.g2_training_adapter import G2TrainingDataAdapter

        adapter = getattr(producer.catalog.raw, "g2_adapter", None)
        if not isinstance(adapter, G2TrainingDataAdapter) or (
            type(factory) is not str or factory.count(":") != 1 or type(kwargs) is not dict
        ):
            raise ValueError("async G2 requires verified provider reconstruction metadata")
        provider = adapter.provider
        if provider.evidence_scope != "verified_data_c" or not all(
            isinstance(value, str) and value for value in (provider.split_identity, provider.snapshot_identity)
        ):
            raise ValueError("async G2 requires a sealed provider snapshot and split")
        return cls(factory, deepcopy(kwargs), adapter.recipe, provider.split_identity, provider.snapshot_identity)

    def build(self):
        from cosmos_framework.data.generator.action.datasets.g2_training_adapter import G2TrainingDataAdapter

        module, name = self.factory.split(":")
        factory = getattr(importlib.import_module(module), name)
        provider = factory(**self.kwargs, global_memory_enabled=self.recipe.global_config.enabled)
        if (
            provider.evidence_scope != "verified_data_c"
            or provider.split_identity != self.split_identity
            or provider.snapshot_identity != self.snapshot_identity
        ):
            raise ValueError("async G2 worker reconstructed a different provider snapshot/split")
        return G2TrainingDataAdapter(provider, self.recipe)


def _conditioned_static(transform, sample, resolution):
    if "text_dropout_applied" in sample:
        raise ValueError("async G2 worker received a preselected CFG branch")
    sample["text_dropout_applied"] = False
    return transform(sample, resolution)


@dataclass(frozen=True)
class AsyncRecipe:
    cache_root: str
    source_root: str
    index_root: str
    trusted_training: bool
    tokenizer_config: dict[str, Any]
    transform_kwargs: dict[str, Any]
    config_digest: str
    source_digest: str
    cache_digest: str
    manifest_digest: str
    ttt_steps: int
    compact: bool
    workers: int
    max_cache_bytes: int
    g2: G2WorkerBinding | None = None

    @property
    def rate(self):
        return self.transform_kwargs["cfg_dropout_rate"]

    @property
    def caption_key(self):
        return self.transform_kwargs.get("caption_key", "ai_caption")

    @property
    def token_key(self):
        return self.transform_kwargs.get("text_token_key", "text_token_ids")

    @classmethod
    def from_producer(cls, producer, profile, *, g2_provider_factory=None, g2_provider_kwargs=None):
        from cosmos_framework.data.generator.action.datasets.robocasa_verified_index import VerifiedExactWindowIndex
        from cosmos_framework.data.generator.processors.nemotron3densevl_processor import Nemotron3DenseVLProcessor

        if profile.workers not in (1, 2, 4, 8):
            raise ValueError("async recipe requires positive No-TTT CPU workers")
        raw = producer.catalog.raw
        uses_g2 = getattr(raw, "g2_adapter", None) is not None
        if uses_g2:
            g2 = G2WorkerBinding.from_producer(producer, g2_provider_factory, g2_provider_kwargs)
        elif g2_provider_factory is not None or g2_provider_kwargs is not None:
            raise ValueError("async G2 provider metadata without an actual G2 adapter")
        else:
            g2 = None
        wrapped, _, tokenizer, kwargs = _known_pipeline(producer, g2=uses_g2)
        source = raw.source_reader
        index = getattr(source, "verified_index", None)
        if type(index) is not VerifiedExactWindowIndex:
            raise ValueError("async No-TTT requires the existing verified offline index")
        # Reopen the already resolved local processor artifact. Repository lazy
        # configs can invoke an HF CLI subprocess; that must not run in workers.
        processor_path = getattr(tokenizer._processor, "name", None)
        if (
            type(tokenizer._processor) is not Nemotron3DenseVLProcessor
            or not isinstance(processor_path, str)
            or not Path(processor_path).is_dir()
        ):
            raise ValueError("async No-TTT requires an already staged local tokenizer directory")
        original_config = tokenizer.args["tokenizer_config"]
        target = original_config.get("_target_")
        target = (
            target
            if isinstance(target, str)
            else f"{getattr(target, '__module__', '')}.{getattr(target, '__qualname__', '')}"
        )
        if target not in {
            "cosmos_framework.data.generator.processors.build_processor_lazy",
            "cosmos_framework.data.generator.processors.build_processor",
        }:
            raise ValueError("async No-TTT cannot reconstruct an unknown tokenizer factory")
        native_edge = original_config.get("use_native_edge_processor", False)
        if type(native_edge) is not bool:
            raise ValueError("invalid native Edge processor option")
        local_config = {
            "_target_": "cosmos_framework.data.generator.processors.build_processor_lazy",
            "tokenizer_type": str(Path(processor_path).resolve()),
            "use_native_edge_processor": native_edge,
        }
        copied = deepcopy(kwargs)
        copied["tokenizer_config"] = local_config
        result = cls(
            str(Path(raw.catalog.cache_root).resolve()),
            str(Path(source.source_root).resolve()),
            str(Path(index.root).resolve()),
            index.trusted_training,
            local_config,
            copied,
            producer.config_digest,
            producer.catalog.source_binding_digest,
            producer.catalog.cache_corpus_digest,
            producer.catalog.manifest_digest,
            producer.catalog.ttt_tbptt_steps,
            raw.compact_cached_video,
            profile.workers,
            profile.max_cache_bytes,
            g2,
        )
        if len(pickle.dumps(result, protocol=5)) > 1024**2:
            raise ValueError("async CPU recipe exceeds 1MiB metadata bound")
        return result

    def build(self):
        from cosmos_framework.data.generator.action.datasets.action_sft_dataset import ActionSFTDataset
        from cosmos_framework.data.generator.action.datasets.robocasa_exact_window_cache import (
            RoboCasaExactWindowCacheCatalog,
        )
        from cosmos_framework.data.generator.action.datasets.robocasa_exact_window_cached_sft import (
            RoboCasaExactWindowCachedDataset,
        )
        from cosmos_framework.data.generator.action.datasets.robocasa_exact_window_policy import (
            CorrectedRoboCasaPolicyContract,
        )
        from cosmos_framework.data.generator.action.datasets.robocasa_exact_window_source import (
            RoboCasaExactWindowSourceReader,
        )
        from cosmos_framework.data.generator.action.datasets.robocasa_verified_index import VerifiedExactWindowIndex
        from cosmos_framework.data.generator.action.utils.cached_pixel_geometry import CachedGeometryResize
        from cosmos_framework.data.generator.action.utils.transforms import ActionTransformPipeline
        from cosmos_framework.model.generator.mot.robocasa_exact_window_local import (
            ExactWindowLocalCatalog,
            ExactWindowSegmentProducer,
        )
        from cosmos_framework.trainer.no_ttt_cpu_loader import MappedEpisodeReader
        from cosmos_framework.trainer.no_ttt_worker_source import VerifiedWorkerLeRobotDataset

        if torch.cuda.is_initialized() or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise RuntimeError("async worker reconstruction requires CPU-only startup")
        os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
        index = VerifiedExactWindowIndex.open(
            self.index_root,
            cache_root=self.cache_root,
            source_root=self.source_root,
            verify_files=not self.trusted_training,
        )
        catalog = RoboCasaExactWindowCacheCatalog(self.cache_root, verified_index=index)
        source = RoboCasaExactWindowSourceReader(
            catalog,
            self.source_root,
            verified_index=index,
            dataset_factory=partial(
                VerifiedWorkerLeRobotDataset,
                verified_index=index,
                max_episodes=max(1, 8 // self.workers),
                max_bytes=self.max_cache_bytes,
            ),
        )
        if (
            catalog.corpus_digest != self.cache_digest
            or catalog.manifest_sha256 != self.manifest_digest
            or source.source_binding_digest != self.source_digest
        ):
            raise ValueError("async worker source/cache/index identity mismatch")
        # max_cache_bytes retains the existing loader's per-worker meaning.
        # Dividing it by worker count would reject previously valid large
        # Episodes when switching to nw8. Total rank mapping is still bounded
        # by workers * max_cache_bytes and at most eight cached Episodes.
        reader = MappedEpisodeReader(catalog, max(1, 8 // self.workers), self.max_cache_bytes)
        source.cache_reader = reader
        adapter = self.g2.build() if self.g2 is not None else None
        raw = RoboCasaExactWindowCachedDataset(
            catalog,
            source,
            CorrectedRoboCasaPolicyContract.from_cache_catalog(catalog),
            cache_reader=reader,
            g2_adapter=adapter,
        )
        raw.compact_cached_video = self.compact
        kwargs = deepcopy(self.transform_kwargs)
        rate = kwargs["cfg_dropout_rate"]
        if self.g2 is None:
            kwargs["cfg_dropout_rate"] = 0.0
        transform = ActionTransformPipeline(**kwargs)
        if self.compact:
            transform.video_resize = CachedGeometryResize(transform.video_resize, tuple(raw._canvas_geometry))
        wrapped = ActionSFTDataset(
            raw, transform if self.g2 is None else partial(_conditioned_static, transform), resolution=None
        )
        local = ExactWindowLocalCatalog(
            wrapped,
            ttt_tbptt_steps=self.ttt_steps,
            allowed_episode_keys=None if adapter is None else adapter.provider.train_episode_keys,
        )
        producer = ExactWindowSegmentProducer(local, config_digest=self.config_digest)
        return AsyncWorkerState(producer, rate=rate, cache_reader=reader, transform=transform)


class AsyncWorkerState:
    """One worker's private deterministic reader/transform and bounded mmap cache."""

    def __init__(self, producer, *, rate, cache_reader, transform=None):
        from cosmos_framework.data.generator.augmentors.text_tokenizer import _MAX_NUM_TOKENS
        from cosmos_framework.trainer.no_ttt_cpu_loader import _require_same_rng, _rng_snapshot

        self.producer, self.rate, self.cache_reader = producer, rate, cache_reader
        self.g2_source_versions = ()
        if transform is None:
            transform = producer.catalog.wrapped_sft._transform
        tokenizer = transform.text_tokenizer
        recipe = getattr(transform, "static_task_text_recipe", None)
        self.g2_provider = producer.catalog.raw.g2_adapter.provider if recipe is not None else None
        if recipe is None and tokenizer.cfg_dropout_rate != 0:
            raise ValueError("worker must construct only the deterministic conditioned CFG branch")
        self.caption_key, self.token_key = tokenizer.input_keys[0], tokenizer.output_keys[0]
        before = _rng_snapshot()
        if recipe is None:
            tokens = tokenizer._processor.tokenize_text(
                "", is_video=False, use_system_prompt=tokenizer.use_system_prompt
            )[:_MAX_NUM_TOKENS]
            self.empty_tokens = torch.tensor(tokens)
        else:
            from cosmos_framework.data.generator.action.utils.static_task_text import (
                prepare_empty_static_task_text,
            )

            # The dropped branch depends on the fixed recipe/tokenizer only.
            # Preserve the actual consumer's task identity during selection.
            self.empty_tokens = prepare_empty_static_task_text(recipe, tokenizer._processor)
        _require_same_rng(before)

    def prepare(self, request):
        from cosmos_framework.model.generator.mot.robocasa_exact_window_local import ExactWindowSegmentRequest
        from cosmos_framework.trainer.no_ttt_cpu_loader import _require_same_rng, _rng_snapshot

        if type(request) is not ExactWindowSegmentRequest:
            raise TypeError("async worker request type mismatch")
        episode = self.producer.catalog.by_uid.get(request.episode.uid)
        if episode is None or episode != request.episode:
            raise ValueError("async worker episode metadata/capability mismatch")
        local_request = replace(request, episode=episode)
        events = []
        before = _rng_snapshot()
        started = time.monotonic()
        capture = self.g2_provider.capture_prefetch_witnesses() if self.g2_provider is not None else nullcontext({})
        with capture as witnesses:
            prepared = self.producer.prepare(local_request)
        self.g2_source_versions = tuple(
            (str(path), *token) for path, token in sorted(witnesses.items(), key=lambda row: str(row[0]))
        )
        events.append({"stage": "cpu_source", "start": started, "end": time.monotonic(), "pid": os.getpid()})
        _require_same_rng(before)
        started = time.monotonic()
        segment = self.producer.materialize(prepared)
        events.append({"stage": "cpu_transform", "start": started, "end": time.monotonic(), "pid": os.getpid()})
        _require_same_rng(before)
        if torch.cuda.is_initialized():
            raise RuntimeError("async worker unexpectedly initialized CUDA")
        return segment, events
