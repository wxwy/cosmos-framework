# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Concrete G2 parameter ownership and FSDP2 trainability boundary; no stage policy."""

from torch import nn
from torch.distributed.fsdp import FSDPModule


def parameter_inventory(model: nn.Module, *, net_prefix: str = "") -> dict:
    groups = {name: [] for name in ("global", "gen", "action_adapter", "frozen")}
    for name, _ in model.named_parameters():
        key = name.removeprefix(net_prefix)
        if net_prefix and not name.startswith(net_prefix):
            group = "frozen"
        elif key.startswith(("global_memory_reader.", "global_memory_projector.")):
            group = "global"
        elif key.startswith("local_memory"):
            group = "frozen"
        elif key.startswith("language_model."):
            group = "gen" if "moe_gen" in key or "k_norm_und_for_gen" in key else "frozen"
        elif key.startswith(("action2llm.", "llm2action.", "action_modality_embed")):
            group = "action_adapter"
        else:
            group = "gen"
        groups[group].append(name)
    return {
        "parameters": {group: tuple(names) for group, names in groups.items()},
        "new_state_keys": tuple(
            name
            for name in model.state_dict()
            if name.removeprefix(net_prefix).startswith(("global_memory_reader.", "global_memory_projector."))
        ),
    }


def set_trainable_names(model: nn.Module, names) -> None:
    """Call on every rank after full optimizer/Local commit and zero_grad.

    This changes flags, not parameter objects or optimizer groups/moments. FSDP2
    refresh is isolated here because its dtype bookkeeping is an internal API;
    real distributed A/B resume remains a required execution-side validation.
    """
    names = set(names)
    params = dict(model.named_parameters())
    unknown = names - params.keys()
    if unknown:
        raise ValueError(f"unknown trainable parameter names: {sorted(unknown)}")
    if any(p.grad is not None for p in params.values()):
        raise RuntimeError("trainability switch requires zero_grad(set_to_none=True) at the transaction boundary")
    groups = []
    for module in model.modules():
        if isinstance(module, FSDPModule):
            group = module._get_fsdp_state()._fsdp_param_group
            if group is not None:
                if not group.is_sharded:
                    raise RuntimeError("trainability switch requires resharded FSDP units")
                groups.append(group)
    for name, parameter in params.items():
        parameter.requires_grad_(name in names)
    for group in groups:
        for parameter in group.fsdp_params:
            if hasattr(parameter, "_unsharded_param"):
                parameter._unsharded_param.requires_grad_(parameter.sharded_param.requires_grad)
        group._init_mp_dtypes()
