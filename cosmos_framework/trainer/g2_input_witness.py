# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Read-only, post-forward G2 witness of the *actual* native training inputs.

The DATA_C provider checks bound file identities and source SHA before reading.
This witness checks the *transformed payload* and the *G2ConditionBatch* used
by the real FSDP training_step. It does not own data, RNG, model, checkpoint,
or optimizer state, and is emitted only after a native forward returned.

No project scripts or model data outside the checked-out repository are loaded.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from cosmos_framework.model.generator.mot.g2_contracts import (
    EffectiveTextBatch,
    G2ConditionBatch,
    G2SampleIdentity,
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def inspect_native_g2_inputs(
    payloads: Sequence[Mapping[str, Any]],
    condition: G2ConditionBatch,
    *,
    profile: str,
    iteration: int,
    rank: int,
) -> dict[str, Any]:
    """Validate source-backed native inputs after model forward, without copying latents.

    Only non-dropped captions carry the augmented instruction: 10% CFG dropout
    intentionally replaces individual captions with the true empty prompt.
    R1 must never supply Global views, while R2 must supply real latent grids.
    """
    if profile not in ("R1", "R2") or type(iteration) is not int or iteration <= 0:
        raise ValueError("invalid formal G2 witness profile/iteration")
    if type(rank) is not int or rank < 0:
        raise ValueError("invalid formal G2 witness rank")
    if not isinstance(condition, G2ConditionBatch) or not isinstance(condition.text, EffectiveTextBatch):
        raise ValueError("native G2ConditionBatch is missing")
    if not payloads or len(payloads) != len(condition.text.lengths):
        raise ValueError("native payload and effective UND text sizes differ")
    if len(condition.views) != len(payloads):
        raise ValueError("native Global view sample count differs")
    if condition.text_features is not None or condition.global_tokens is not None:
        raise ValueError("train input witness must not use cached inference projections")
    if condition.gen_text_kv is not None or condition.global_kv is not None:
        raise ValueError("train input witness forbids inference KV snapshots")

    rows = []
    augmented = dropped = n_views = 0
    for index, (payload, is_dropped, token_length, views) in enumerate(
        zip(payloads, condition.text.dropped, condition.text.lengths, condition.views, strict=True)
    ):
        identity = payload.get("g2_sample_identity")
        if (
            not isinstance(identity, G2SampleIdentity)
            or payload.get("g2_provider_scope") != "verified_data_c"
            or identity.decision_frame != payload.get("start_frame")
            or payload.get("text_task_identity") is None
        ):
            raise ValueError("native G2 sample is not sourced from certified DATA_C")
        raw = payload.get("action_raw")
        padded = payload.get("action")
        if (
            not isinstance(raw, torch.Tensor)
            or raw.shape != (17, 15)
            or raw.device.type != "cpu"
            or not isinstance(padded, torch.Tensor)
            or padded.shape != (17, 64)
            or padded.device.type != "cpu"
            or not bool(torch.isfinite(raw).all())
            or not torch.equal(raw, padded[:, :15])
        ):
            raise ValueError("native Action input does not carry the actual State15 row")
        pose5 = raw[0, :5]
        sine, cosine = float(pose5[3]), float(pose5[4])
        if not math.isfinite(sine * sine + cosine * cosine) or abs(sine * sine + cosine * cosine - 1) > 1e-3:
            raise ValueError("State15 Pose5 yaw was not normalized")
        tokens = payload.get("text_token_ids")
        caption = payload.get("ai_caption")
        if (
            not isinstance(tokens, torch.Tensor)
            or tokens.ndim != 1
            or tokens.dtype != torch.long
            or tokens.numel() == 0
            or not isinstance(caption, str)
            or type(payload.get("text_dropout_applied")) is not bool
            or payload["text_dropout_applied"] != is_dropped
            or token_length < tokens.numel()
            or payload.get("text_recipe_identity") != condition.text.formatter_id
        ):
            raise ValueError("actual native effective Text input disagrees with per-consumer tokenization")
        if is_dropped:
            if caption != "":
                raise ValueError("CFG-dropped G2 prompt must be genuinely empty")
            dropped += 1
            has_initial_context = False
            description_size = 0
        else:
            try:
                decoded = json.loads(caption)
                actions = decoded.get("actions", ())
                description = actions[0]["description"]
            except (ValueError, IndexError, KeyError, TypeError, AttributeError) as exc:
                raise ValueError("enhanced CAP2 did not reach the native Prompt formatter") from exc
            if not isinstance(description, str) or not description.strip():
                raise ValueError("native non-dropout enhanced instruction is empty")
            has_initial_context = "Initial spatial context" in description
            if not has_initial_context:
                raise ValueError("CAP2 enhanced spatial context was absent from the native Prompt")
            augmented += 1
            description_size = len(description)

        if profile == "R1" and views is not None:
            raise ValueError("R1 must have no Global view latent")
        if profile == "R2" and (not isinstance(views, tuple) or not views):
            raise ValueError("R2 training must pass actual spatial latent views")
        view_list = []
        if views is not None:
            n_views += len(views)
            for view in views:
                z = view.latent
                valid = view.valid_mask
                if (
                    not isinstance(z, torch.Tensor)
                    or z.ndim != 3
                    or not z.is_floating_point()
                    or not isinstance(valid, torch.Tensor)
                    or valid.dtype != torch.bool
                    or valid.shape != z.shape[1:]
                    or z.device != valid.device
                    or not bool(torch.isfinite(z).all())
                    or not hasattr(view.identity, "recipe_id")
                    or view.identity.image_owner_frame > identity.logical_state_frame
                ):
                    raise ValueError("R2 native Global view latent/mask/source is invalid")
                view_list.append(
                    {
                        "view_id": view.identity.view_id,
                        "image_owner_frame": view.identity.image_owner_frame,
                        "latent_shape": list(z.shape),
                        "valid_pixels": int(valid.sum().item()),
                        "latent_abs_mean": float(z.detach().abs().float().mean().item()),
                        "vae_recipe_id": view.identity.recipe_id,
                        "top_variant": view.top_variant,
                    }
                )
            tops = [entry for entry in view_list if entry["top_variant"] is not None]
            if len(tops) != 1 or tops[0]["top_variant"] != "with_robot":
                raise ValueError("R2 native condition lacks its selected top-with-robot latent")
        rows.append(
            {
                "sample": index,
                "episode_id": identity.episode_id,
                "decision_frame": identity.decision_frame,
                "logical_state_frame": identity.logical_state_frame,
                "state15_recipe_id": identity.state15_recipe_id,
                "data_identity": identity.data_version,
                "pose5_source": "observation.state[0:7]_source_base_world_not_wheeled_base_body",
                "pose5_actual": [round(float(v), 7) for v in pose5],
                "enhanced_prompt_sha256": _sha(caption),
                "native_text_tokens": int(tokens.numel()),
                "effective_und_tokens": int(token_length),
                "text_dropout_applied": is_dropped,
                "enhanced_description_length": description_size,
                "has_initial_spatial_context": has_initial_context,
                "selected_views": view_list,
            }
        )
    if profile == "R2" and n_views < len(payloads):
        raise ValueError("R2 selected latent count cannot be below the sample count")
    return {
        "schema": "psm_g2_real_native_forward_input_v1",
        "meaning": "one actual transformed GA member accepted by native model forward",
        "rank": rank,
        "profile": profile,
        "optimizer_iteration": iteration,
        "samples": len(rows),
        "enhanced_active_samples": augmented,
        "cfg_dropout_samples": dropped,
        "selected_view_latents": n_views,
        "state15_pose5_source": "MuJoCo observation.state base-world position+quaternion",
        "rows": rows,
    }


def append_g2_input_witness(directory: Path, report: Mapping[str, Any], *, rank: int) -> Path:
    """Append a per-rank independent witness. Never modify the training DCP."""
    if type(rank) is not int or rank < 0:
        raise ValueError("invalid rank")
    if report.get("rank") != rank or report.get("schema") != "psm_g2_real_native_forward_input_v1":
        raise ValueError("runtime witness rank/schema mismatch")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("unsafe training witness directory")
    dest = directory / f"g2_native_inputs_rank{rank}.jsonl"
    if dest.is_symlink():
        raise ValueError("unsafe training witness path")
    content = (json.dumps(report, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    fd = os.open(dest, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        os.write(fd, content)
        os.fsync(fd)
    finally:
        os.close(fd)
    return dest
