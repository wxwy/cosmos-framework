# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""G2 configuration overlay for the existing Phase5 entry/trainer.

No training loop or offline provider lives here. Required run choices have no
formal defaults; a missing DATA_C provider or B/FSDP interface fails explicitly.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from omegaconf import OmegaConf
from torch.distributed.tensor import DTensor

from cosmos_framework.checkpoint.g2_warm_start import G2ModelOnlyCheckpointer
from cosmos_framework.data.generator.action.datasets.g2_training_adapter import (
    G2DataRecipe,
    G2TrainingDataAdapter,
)
from cosmos_framework.data.generator.action.utils.static_task_text import StaticTaskTextRecipe
from cosmos_framework.model.generator.mot.g2_contracts import GlobalMemoryConfig
from cosmos_framework.trainer.g2_training import (
    G2_OPTIMIZER_SELECTION,
    G2StageController,
    G2TrainingPlan,
    build_g2_optimizer,
    parameter_roles,
)

GLOBAL_PREFIXES = ("net.global_memory_reader.", "net.global_memory_projector.")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class G2RunRecipe:
    training: G2TrainingPlan
    text: StaticTaskTextRecipe
    global_memory: GlobalMemoryConfig
    alignment_version: str
    view_recipe_id: str | None
    provider_factory: str
    provider_kwargs: dict[str, Any]
    g1_config_path: str
    g1_config_sha256: str
    save_iter: int

    def __post_init__(self):
        if (self.training.profile == "R2") != self.global_memory.enabled:
            raise ValueError("R1/R2 and Global enabled must agree")
        if self.training.text_recipe_identity != self.text.identity or self.text.cfg_dropout_rate != 0.1:
            raise ValueError("shared text identity/dropout=0.1 must be frozen in the plan")
        if (
            type(self.save_iter) is not int
            or self.save_iter < 1
            or not self.alignment_version
            or len(self.g1_config_sha256) != 64
            or not self.g1_config_path
        ):
            raise ValueError("explicit save interval, alignment and G1 config witness are required")
        if (
            not isinstance(self.provider_factory, str)
            or self.provider_factory.count(":") != 1
            or not isinstance(self.provider_kwargs, dict)
        ):
            raise ValueError("DATA_C provider factory must be an explicit module:function")
        module, factory = self.provider_factory.split(":")
        if not module or not factory or module.startswith(("tools", "g2.offline", "g2.sim")):
            raise ValueError("training provider must not import root tools or simulation producer")

    @classmethod
    def open(cls, path: Path):
        body = json.loads(path.read_text())
        if not isinstance(body, dict):
            raise ValueError("G2 recipe must be an explicit JSON object")
        try:
            body["training"] = G2TrainingPlan.from_dict(body["training"])
            body["text"] = StaticTaskTextRecipe(**body["text"])
            body["global_memory"] = GlobalMemoryConfig(**body["global_memory"])
            return cls(**body)
        except (KeyError, TypeError) as error:
            raise ValueError(f"incomplete/unknown G2 recipe fields: {error}") from error

    @property
    def identity(self):
        # The path itself is operational; actual G1 configuration bytes bind semantics.
        body = asdict(self)
        body.pop("g1_config_path")
        return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def g1_snapshot(self):
        path = Path(self.g1_config_path)
        if not path.is_file() or _sha(path) != self.g1_config_sha256:
            raise ValueError("G1 config snapshot is missing or its SHA256 differs")
        saved = yaml.safe_load(path.read_text())
        if (
            not isinstance(saved, dict)
            or not isinstance(saved.get("model", {}).get("config"), dict)
            or not isinstance(saved.get("optimizer"), dict)
        ):
            raise ValueError("G1 config must contain actual model/optimizer settings")
        local = saved["model"]["config"]
        required = {
            "local_memory_enabled",
            "local_memory_inner_lr",
            "local_memory_ttt_tbptt_steps",
            "local_memory_k_local",
            "local_memory_dim",
            "local_memory_evidence_dim",
            "local_memory_action_dim",
            "local_memory_ttt_dim",
            "local_memory_fast_hidden_dim",
        }
        if not required <= local.keys() or local["local_memory_enabled"] is not True:
            raise ValueError("G1 snapshot must include its complete existing Local recipe")
        return saved

    def configure_args(self, args):
        if (
            getattr(args, "no_ttt", False)
            or getattr(args, "owner_authorized_g1_cross_pair_resume", False)
            or getattr(args, "stop_after_iter", None) is not None
            or getattr(args, "formal_execution_until", None) is not None
        ):
            raise ValueError("G2 uses its own strict new-job plan, not old G1/No-TTT run overrides")
        if not args.job_name.startswith(self.training.profile.lower() + "_g2_"):
            raise ValueError("G2 requires a separate r1_g2_ / r2_g2_ job namespace")
        saved = self.g1_snapshot()["model"]["config"]
        args.t, args.k = saved["local_memory_ttt_tbptt_steps"], saved["local_memory_k_local"]
        args.b, args.ga, args.world_size = (
            self.training.b_stream,
            self.training.grad_accum_steps,
            self.training.world_size,
        )
        args.max_iter, args.save_iter = self.training.total_steps, self.save_iter
        args.warmup = self.training.gen_schedule.warmup_steps  # Native scheduler is replaced before load/steps.

    def check_init_checkpoint(self, checkpoint: Path):
        from torch.distributed.checkpoint import FileSystemReader

        path = checkpoint / "model/.metadata"
        if not path.is_file() or _sha(path) != self.training.init_checkpoint_identity:
            raise ValueError("G1 model-only initialization metadata SHA256 differs from the paired plan")
        names = FileSystemReader(checkpoint / "model").read_metadata().state_dict_metadata
        if not any(name.startswith("net.local_memory_runtime.") for name in names):
            raise ValueError("G1 initialization checkpoint is missing Local weights")

    def overlay(self, config):
        saved = self.g1_snapshot()
        model = config.model.config
        for name, value in saved["model"]["config"].items():
            if name.startswith("local_memory_"):
                setattr(model, name, value)
        if "rectified_flow_training_config" not in saved["model"]["config"]:
            raise ValueError("G1 loss/sampling recipe must be explicit in the verified config snapshot")
        # The native OmniMoT field is a structured RectifiedFlowTrainingConfig.
        # Assigning the G1 YAML dict directly fails OmegaConf type validation.
        # Merge into the field's existing typed schema, preserving its class
        # and all G1 loss/sampling values without altering the recipe.
        flow = saved["model"]["config"]["rectified_flow_training_config"]
        if OmegaConf.is_config(model):
            model.rectified_flow_training_config = OmegaConf.merge(
                model.rectified_flow_training_config, flow
            )
        else:
            # The existing CPU-only lightweight fixture is a SimpleNamespace.
            model.rectified_flow_training_config = flow
        # B owns these model configuration fields and must publish them with
        # its core candidate. No silent insertion into an old model schema.
        if not hasattr(model, "global_memory_enabled") or not hasattr(model, "global_memory_config"):
            raise ValueError("B Global model config interface is unavailable; integrate the fixed B candidate")
        model.global_memory_enabled = self.global_memory.enabled
        model.g2_text_enabled = True
        model.global_memory_config = asdict(self.global_memory)
        model.global_memory_top_variant = self.global_memory.top_view_variant
        inherited = saved["optimizer"]
        if not {"optimizer_type", "betas", "eps", "weight_decay", "fused"} <= inherited.keys():
            raise ValueError("G1 optimizer settings must be explicit; no new low-level defaults")
        for key, value in inherited.items():
            if key not in {"_target_", "keys_to_select", "lr", "lr_multipliers"}:
                setattr(config.optimizer, key, value)
        # Phase5 callbacks (including any actual clipping) come from the G1
        # snapshot. No new loss, inner algorithm, clip or optimizer tuning.
        config.trainer.callbacks = saved.get("trainer", {}).get("callbacks", {})
        config.optimizer.lr = 1e-5
        config.optimizer._target_ = build_g2_optimizer
        config.optimizer.keys_to_select = [G2_OPTIMIZER_SELECTION]
        config.optimizer.lr_multipliers = {
            "global_memory_reader.": 5.0,
            "global_memory_projector.": 5.0,
            "action2llm": 5.0,
            "llm2action": 5.0,
            "action_modality_embed": 5.0,
        }
        config.checkpoint.type = {"_target_": G2ModelOnlyCheckpointer}
        config.checkpoint.keys_to_skip_loading = ["net_ema."]  # Exact Global names added by validated DCP loader.

    def data_adapter(self):
        module, name = self.provider_factory.split(":")
        factory = getattr(importlib.import_module(module), name)
        provider = factory(**self.provider_kwargs, global_memory_enabled=self.global_memory.enabled)
        if provider.evidence_scope != "verified_data_c" or provider.split_identity != self.training.split_identity:
            raise ValueError(
                "formal G2 requires a verified DATA_C provider with the frozen split; fixtures are CPU only"
            )
        return G2TrainingDataAdapter(
            provider,
            G2DataRecipe(
                self.training.data_identity,
                self.alignment_version,
                self.training.state15_recipe_identity,
                self.view_recipe_id,
                self.global_memory,
            ),
        )


def recipe_from_args(args) -> G2RunRecipe | None:
    path = getattr(args, "g2_recipe", None)
    if path is None:
        return None
    existing = getattr(args, "_g2_resolved_recipe", None)
    if existing is None:
        existing = G2RunRecipe.open(path)
        existing.configure_args(args)
        args._g2_resolved_recipe = existing
    return existing


def telemetry_parameter_group(trainer, name):
    """Use existing observer labels for all ACTUALLY active generation params.

    Its generation aggregate includes Global. The exact three-way split and
    update/sample clocks are separately emitted by the G2 controller.
    """
    controller = getattr(trainer, "_g2_stage_controller", None)
    if controller is None:
        return None
    role = controller.roles.get(name)
    if role not in controller.active_roles:
        return None
    return "action" if role.value == "action_adapter" else "generation"


def install_g2_training(model, trainer, recipe: G2RunRecipe, report):
    """Extend the existing init hook; retain the factory/optimizer and DCP paths."""
    original = model.init_optimizer_scheduler

    def checked(optimizer_config, scheduler_config):
        named = dict(model.named_parameters())
        global_names = frozenset(name for name in named if name.startswith(GLOBAL_PREFIXES))
        if recipe.global_memory.enabled and any(
            not any(name.startswith(prefix) for name in global_names) for prefix in GLOBAL_PREFIXES
        ):
            raise ValueError("B must register both Global Reader and projector parameters")
        if not hasattr(model, "g2_parameter_inventory"):
            raise ValueError("B's g2_parameter_inventory interface is required")
        inventory = model.g2_parameter_inventory()
        groups = inventory["parameters"]
        flattened = [name for values in groups.values() for name in values]
        if (
            set(groups) != {"global", "gen", "action_adapter", "frozen"}
            or len(flattened) != len(set(flattened))
            or set(flattened) != set(named)
        ):
            raise ValueError("B's actual parameter inventory is incomplete or overlapping")
        if global_names != frozenset(groups["global"]):
            raise ValueError("B's Global inventory disagrees with registered module names")
        generation = frozenset((*groups["gen"], *groups["action_adapter"]))
        actions = frozenset(groups["action_adapter"])
        roles = parameter_roles(
            model,
            global_names=global_names,
            gen_names=generation,
            action_names=actions,
            profile=recipe.training.profile,
        )
        # B's hook owns the FSDP leaf/forward registration implementation.
        hook = getattr(model, "set_trainable_names", None) or getattr(model.net, "set_trainable_names", None)
        if any(isinstance(parameter, DTensor) for parameter in named.values()) and hook is None:
            raise ValueError("B registered FSDP set_trainable_names hook is required")
        if hook is not None:
            hook(frozenset(roles))
        else:
            for name, parameter in named.items():
                parameter.requires_grad_(name in roles)
        model._g2_optimizer_parameter_names = frozenset(roles)
        optimizer, _old_scheduler = original(optimizer_config, scheduler_config)
        controller = G2StageController(model, optimizer, recipe.training, roles, set_trainability=hook)
        expected_new = frozenset(name for name in model.state_dict() if name.startswith(GLOBAL_PREFIXES))
        if expected_new != frozenset(inventory["new_state_keys"]):
            raise ValueError("B's new Global state inventory disagrees with actual state_dict")
        model._g2_new_global_state_names = expected_new
        trainer._g2_stage_controller, trainer._g2_text_recipe = controller, recipe.text
        trainer._required_checkpoint_iterations = recipe.training.required_checkpoint_iterations
        report["g2_actual_parameter_inventory"] = controller.inventory()
        report["g2_stage_schema"] = controller.state_dict()
        if hasattr(trainer, "g2_evidence_dir"):
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            (trainer.g2_evidence_dir / f"g2_parameter_inventory_before_load_rank_{rank}.json").write_text(
                json.dumps(
                    {
                        "scope": "actual_runtime_before_checkpoint_load",
                        "stage": controller.state_dict(),
                        "parameters": controller.inventory(),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
        return optimizer, controller

    model.init_optimizer_scheduler = checked


def main(argv=None):
    """CPU configuration checker; never constructs a model or opens view payloads."""
    import argparse

    from cosmos_framework.trainer.g2_training import validate_comparison_plans

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", type=Path, required=True)
    parser.add_argument("--pair-with", type=Path)
    args = parser.parse_args(argv)
    try:
        recipe = G2RunRecipe.open(args.check)
        recipe.g1_snapshot()
        report = {"scope": "CONFIG_ONLY_PROVIDER_AND_MODEL_NOT_RUN", "recipe_identity": recipe.identity}
        if args.pair_with is not None:
            other = G2RunRecipe.open(args.pair_with)
            other.g1_snapshot()
            r1, r2 = (recipe, other) if recipe.training.profile == "R1" else (other, recipe)
            if (
                r1.g1_config_sha256 != r2.g1_config_sha256
                or r1.text != r2.text
                or r1.alignment_version != r2.alignment_version
            ):
                raise ValueError("R1/R2 must share the G1 snapshot, text and DATA_C alignment")
            report["comparison"] = validate_comparison_plans(r1.training, r2.training)
        print(json.dumps(report, indent=2, sort_keys=True))
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
