# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Read-only startup authority for the native, single-rank G2 R1 Policy Server."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

import torch

from cosmos_framework.checkpoint.safe_local_witness import (
    SAFE_LOCAL_COMPLETE,
    SAFE_LOCAL_PROFILE,
    component_witness,
    file_witness,
)
from cosmos_framework.data.generator.action.utils.static_task_text import StaticTaskTextRecipe
from cosmos_framework.inference.g2_model_hooks import G2ModelHooks
from cosmos_framework.model.generator.mot.g2_contracts import GlobalMemoryConfig, g2_run_recipe_identity


def _digest(value) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _record(path: Path) -> dict:
    file_witness(path)
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"expected an object in {path}")
    return value


def _file_generation(path: Path) -> tuple:
    stat = path.stat(follow_symlinks=False)
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _sha256(value) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None


@dataclass(frozen=True)
class NativeG2R1Authority:
    model_path: Path
    config_path: Path
    text_recipe: StaticTaskTextRecipe
    state15_recipe_id: str
    alignment_version: str
    top_variant: str
    versions: dict[str, str]
    files: dict[Path, tuple]

    def check_unchanged(self) -> None:
        if any(_file_generation(path) != witness for path, witness in self.files.items()):
            raise ValueError("native G2 model/config/source changed during startup")

    @classmethod
    def open(cls, *, run_recipe: Path, model_path: Path, config_path: Path, metadata_sha256: str):
        if not _sha256(metadata_sha256):
            raise ValueError("native G2 requires an explicit model/.metadata SHA256")
        model_path, config_path = Path(model_path), Path(config_path)
        step = model_path.parent
        if (
            model_path.name != "model"
            or model_path.is_symlink()
            or step.is_symlink()
            or step.parent.name != "checkpoints"
            or step.parent.is_symlink()
            or not re.fullmatch(r"iter_[0-9]{9}", step.name)
            or int(step.name[5:]) < 1
        ):
            raise ValueError("native G2 requires a published SafeLocal training checkpoint iteration")
        job = step.parent.parent
        if config_path.is_symlink() or config_path.resolve() != (job / "config.yaml").resolve():
            raise ValueError("native G2 requires the checkpoint job's canonical config.yaml")

        run_recipe = Path(run_recipe)
        job_path = job / "v3_job_identity.json"
        marker_path = step / SAFE_LOCAL_COMPLETE
        metadata_path = model_path / ".metadata"
        files = {
            path: _file_generation(path) for path in (run_recipe, job_path, marker_path, config_path, metadata_path)
        }
        recipe = _record(run_recipe)
        try:
            training = recipe["training"]
            text = StaticTaskTextRecipe(**recipe["text"])
            global_memory = GlobalMemoryConfig(**recipe["global_memory"])
            state_recipe = training["state15_recipe_identity"]
            alignment = recipe["alignment_version"]
        except (KeyError, TypeError) as error:
            raise ValueError("native G2 requires a complete materialized training recipe") from error
        if (
            training.get("profile") != "R1"
            or training.get("text_recipe_identity") != text.identity
            or global_memory.enabled
            or recipe.get("view_recipe_id") is not None
            or text.cfg_dropout_rate != 0.1
            or (text.planned_video_frames, text.fps) != (17, 20)
            or not _sha256(state_recipe)
            or not _sha256(text.tokenizer_identity)
            or not isinstance(alignment, str)
            or not alignment
        ):
            raise ValueError("native G2 requires the frozen R1 text, State15 and Raw15/H16 contract")
        recipe_id = g2_run_recipe_identity(recipe)
        receipt = _record(job_path)
        semantic = receipt.get("semantic") or {}
        contract = semantic.get("contract") or {}
        if (
            receipt.get("schema") != "corrected_phase5_f0_v1"
            or receipt.get("diagnostic_only_do_not_resume") is not False
            or receipt.get("semantic_digest") != _digest(semantic)
            or semantic.get("training_profile") != "g2_R1"
            or semantic.get("g2_recipe_identity") != recipe_id
            or semantic.get("checkpoint", {}).get("safety_profile") != SAFE_LOCAL_PROFILE
            or any(
                contract.get(k) != v
                for k, v in {
                    "raw_action_dim": 15,
                    "state_dim": 15,
                    "h_pred": 16,
                    "consumer_frames": 17,
                }.items()
            )
        ):
            raise ValueError("native G2 recipe or Raw15/State15/H16 disagrees with the formal training job")
        config_sha = file_witness(config_path)["sha256"]
        if config_sha != receipt.get("canonical_config_sha256", {}).get("config.yaml"):
            raise ValueError("native G2 config differs from the canonical training manifest")
        marker = _record(marker_path)
        model_receipt = marker.get("components", {}).get("model") or {}
        expected_metadata = model_receipt.get("metadata") or {}
        if (
            marker.get("schema") != SAFE_LOCAL_PROFILE
            or marker.get("complete") is not True
            or marker.get("checkpoint") != step.name
            or marker.get("iteration") != int(step.name[5:])
            or marker.get("world_size") != training.get("world_size")
            or expected_metadata.get("sha256") != metadata_sha256
            or not model_receipt.get("files")
        ):
            raise ValueError("native G2 checkpoint publication does not match the pinned model metadata or recipe")
        for name in model_receipt["files"]:
            if Path(name).name != name or not name.endswith(".distcp"):
                raise ValueError("invalid native G2 model shard path")
            path = model_path / name
            files[path] = _file_generation(path)
        if component_witness(model_path, expected_metadata=expected_metadata) != model_receipt:
            raise ValueError("native G2 model shards disagree with the SafeLocal publication")
        versions = {
            "run_recipe_id": recipe_id,
            "run_recipe_sha256": file_witness(run_recipe)["sha256"],
            "checkpoint_id": _digest({"job": receipt["semantic_digest"], "publication": marker}),
            "checkpoint_metadata_sha256": metadata_sha256,
            "config_sha256": config_sha,
            "state15_recipe_id": state_recipe,
            "alignment_version": alignment,
            "tokenizer_identity": text.tokenizer_identity,
        }
        authority = cls(
            model_path, config_path, text, state_recipe, alignment, global_memory.top_view_variant, versions, files
        )
        authority.check_unchanged()
        return authority

    @staticmethod
    def validate_state(state) -> None:
        finite_limit = torch.finfo(torch.float32).max
        if (
            not isinstance(state, list)
            or len(state) != 15
            or any(type(value) not in (int, float) or not -finite_limit <= value <= finite_limit for value in state)
        ):
            raise ValueError("native G2 State15 must be a flat list of 15 finite float32 numbers")

    def validate_sample(self, identity, *, frames, fps, height, width) -> None:
        recipe = self.text_recipe
        if (frames, fps, height, width) != (recipe.planned_video_frames, recipe.fps, recipe.height, recipe.width):
            raise ValueError("native G2 canvas/FPS differs from the frozen training text recipe")
        if (identity.state15_recipe_id, identity.alignment_version) != (self.state15_recipe_id, self.alignment_version):
            raise ValueError("native G2 sample differs from the frozen State15/alignment recipe")

    def hooks(self, model):
        self.check_unchanged()
        config, net_config = model.config, model.net.config
        use_float_positions = config.diffusion_expert_config.enable_fps_modulation
        if (
            model.training
            or type(use_float_positions) is not bool
            or getattr(config, "g2_text_enabled", None) is not True
            or getattr(net_config, "g2_text_enabled", None) is not True
            or getattr(net_config, "local_memory_enabled", None) is not True
            or getattr(config, "global_memory_enabled", None) is not False
            or getattr(net_config, "global_memory_enabled", None) is not False
            or getattr(model.net, "global_memory_reader", None) is not None
            or getattr(model.net, "global_memory_projector", None) is not None
            or not callable(getattr(model, "prepare_g2_text", None))
            or not callable(getattr(model, "prepare_g2_gen_text_kv", None))
        ):
            raise ValueError("native R1 model must have fixed-weight G2 text and Local Memory with Global disabled")
        processor = model.vlm_processor
        tokenizer_path = Path(processor.name) / "tokenizer.json"
        if (
            processor.tokenizer is not model.vlm_tokenizer
            or not callable(getattr(processor, "tokenize_text", None))
            or file_witness(tokenizer_path)["sha256"] != self.text_recipe.tokenizer_identity
        ):
            raise ValueError("native G2 tokenizer does not match the materialized training recipe")
        checkpoint_id = self.versions["checkpoint_id"]
        return G2ModelHooks(
            model,
            versions={
                "checkpoint_id": checkpoint_id,
                "inference_config_id": _digest(
                    {"canonical": self.versions["config_sha256"], "text": asdict(self.text_recipe)}
                ),
                "tokenizer_identity": self.text_recipe.tokenizer_identity,
                "und_version": checkpoint_id,
                "gen_version": checkpoint_id,
                "reader_version": "disabled:r1",
                "projector_version": "disabled:r1",
                "geometry_encoding_version": "disabled:r1",
                "mask_version": "disabled:r1",
                "position_rule": "native_mrope_float" if use_float_positions else "native_mrope_long",
                "dtype": str(model.precision),
            },
            use_float_positions=use_float_positions,
            top_variant=self.top_variant,
        )
