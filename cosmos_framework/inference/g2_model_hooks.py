"""Thin service adapter to B's public Omni methods and shared G2 carrier.

No UND/Reader forward is implemented here. Versions come from the validated
checkpoint/config loader, never a request or a guessed checkpoint path.
"""

from __future__ import annotations

import torch

from cosmos_framework.inference.g2_session_cache import content_key
from cosmos_framework.model.generator.mot.g2_contracts import (
    GlobalCacheSignature,
    TextCacheSignature,
    stack_g2_prepared,
)

VERSION_FIELDS = {
    "checkpoint_id",
    "und_version",
    "gen_version",
    "reader_version",
    "projector_version",
    "geometry_encoding_version",
    "mask_version",
    "position_rule",
    "dtype",
    "tokenizer_identity",
    "inference_config_id",
}


class G2ModelHooks:
    """Construct in the explicitly configured service factory using actual identities.

    Host parameter generations supplement the loader's weight/config identity:
    replacement/in-place changes invalidate derived caches. Unversioned inference
    tensors conservatively force a new generation every request. Object references
    are only mutation guards; content keys never contain addresses, and text/view
    content equality still requires the complete public signatures.

    The factory must choose the native packer's position mode and configured top
    variant; R1 does not import/instantiate a Reader through this adapter.
    """

    def __init__(self, model, *, versions: dict[str, str], use_float_positions: bool, top_variant="with_robot"):
        if set(versions) != VERSION_FIELDS or any(
            not isinstance(v, str) or not v or len(v) > 512 for v in versions.values()
        ):
            raise ValueError("validated complete G2 model/config/tokenizer version identities required")
        if type(use_float_positions) is not bool or top_variant not in ("with_robot", "without_robot"):
            raise ValueError("actual native position mode and one configured top variant required")
        self.model, self.versions = model, dict(versions)
        self.use_float_positions, self.top_variant = use_float_positions, top_variant
        self._parameters = None
        self._generation = 0
        self.refresh_revision()

    @property
    def training(self):
        return self.model.training

    @property
    def tokenizer_identity(self):
        return self.versions["tokenizer_identity"]

    def refresh_revision(self):
        current = []
        untracked = False
        for name, parameter in self.model.named_parameters():
            try:
                version = parameter._version
            except RuntimeError:
                version = None
                untracked = True
            current.append((name, parameter, version, tuple(parameter.shape), parameter.dtype, parameter.device))
        changed = self._parameters is None or len(current) != len(self._parameters)
        if not changed:
            changed = any(
                old[0] != new[0] or old[1] is not new[1] or old[2:] != new[2:]
                for old, new in zip(self._parameters, current, strict=True)
            )
        if changed or untracked:
            self._generation += 1
        self._parameters = current
        self.cache_revision = content_key(self.versions, self.use_float_positions, self.top_variant, self._generation)

    def text_signature(self, session_id, text):
        return TextCacheSignature(
            session_id=session_id,
            checkpoint_id=self.cache_revision,
            und_version=self.versions["und_version"],
            gen_version=self.versions["gen_version"],
            position_rule=self.versions["position_rule"],
            dtype=self.versions["dtype"],
            text=text,
        )

    def global_signature(self, text, variant):
        if variant != self.top_variant:
            raise ValueError(
                "requested top variant differs from the loaded model configuration; explicit reconfigure required"
            )
        return GlobalCacheSignature(
            text=text,
            reader_version=self.versions["reader_version"],
            projector_version=self.versions["projector_version"],
            geometry_encoding_version=self.versions["geometry_encoding_version"],
            top_variant=variant,
            views=(),
            projections=(),
            rasters=(),
            view_types=(),
            mask_version=self.versions["mask_version"],
        )

    def tokenize_static_prompt(self, _serialized):
        raise ValueError("service must supply C-prepared EffectiveTextBatch including native special tokens")

    def prepare_text(self, text):
        return self.model.prepare_g2_text(text)

    def prepare_gen_text_kv(self, features):
        return self.model.prepare_g2_gen_text_kv(features)

    def prepare_task_queries(self, features):
        return self.model.prepare_g2_task_queries(features)

    def read_global(self, queries, views):
        return self.model.read_g2_global(queries, (views,))

    def project_global(self, reader):
        return self.model.project_g2_global(reader)

    def prepare_global_kv(self, projected):
        return self.model.prepare_g2_global_kv(projected)

    def generation_kwargs(self, **prepared):
        with torch.no_grad():
            return {"_g2_condition": stack_g2_prepared(**prepared)}
