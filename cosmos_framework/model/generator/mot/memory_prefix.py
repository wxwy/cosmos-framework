# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Out-of-band Local K/V prefix for native two-way generation attention."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from torch import nn
from torch.nn import functional as F

from cosmos_framework.data.generator.sequence_packing.sequence import SequencePlan
from cosmos_framework.model.generator.mot.local_evidence import (
    ContinualTTTLocalMemoryCore,
    LocalEvidenceEncoder,
)
from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean


class LocalMemoryRuntime(nn.Module):
    """Only slow B0 parameters are registered; fast state stays in the sidecar."""

    def __init__(
        self,
        *,
        evidence_dim: int = 256,
        action_dim: int = 15,
        local_dim: int = 32,
        ttt_dim: int = 64,
        fast_hidden_dim: int = 256,
        inner_lr: float = 0.1,
        ttt_tbptt_steps: int = 16,
        k_local: int = 4,
    ) -> None:
        super().__init__()
        self.encoder = LocalEvidenceEncoder(evidence_dim=evidence_dim, action_dim=action_dim)
        self.core = ContinualTTTLocalMemoryCore(
            evidence_dim=evidence_dim,
            local_dim=local_dim,
            ttt_dim=ttt_dim,
            fast_hidden_dim=fast_hidden_dim,
            inner_lr=inner_lr,
            ttt_tbptt_steps=ttt_tbptt_steps,
            k_local=k_local,
        )


def attach_local_prefixes(
    sequence_plans: list[SequencePlan],
    clean: GenerationDataClean,
    prefixes: tuple[torch.Tensor | None, ...],
    *,
    k_local: int = 4,
    local_dim: int = 32,
) -> tuple[list[SequencePlan], GenerationDataClean]:
    """Copy already materialized native carriers; never mutate dataset-owned inputs."""
    if len(prefixes) != clean.batch_size or len(sequence_plans) != clean.batch_size:
        raise ValueError("Local Memory override requires one entry per native sample")
    if clean.x0_tokens_local_memory is not None or any(plan.has_local_memory for plan in sequence_plans):
        raise ValueError("Duplicate Local Memory authority")
    if any(token is not None and token.shape != (k_local, local_dim) for token in prefixes):
        raise ValueError(f"Local Memory override requires [{k_local},{local_dim}] tokens")
    plans = [
        replace(plan, has_local_memory=token is not None) for plan, token in zip(sequence_plans, prefixes, strict=True)
    ]
    data = replace(clean, x0_tokens_local_memory=[token for token in prefixes if token is not None])
    return plans, data


@dataclass(frozen=True)
class MemoryPrefixContext:
    hidden: torch.Tensor
    sample_offsets: torch.Tensor
    present: torch.Tensor
    k_local: int

    def replace_hidden(self, hidden: torch.Tensor) -> MemoryPrefixContext:
        if hidden.shape != self.hidden.shape or hidden.device != self.hidden.device:
            raise ValueError("Memory Prefix normalized hidden must preserve shape and device")
        return MemoryPrefixContext(hidden, self.sample_offsets, self.present, self.k_local)


def build_memory_prefix_context(
    tokens_by_sample: tuple[torch.Tensor | None, ...] | None,
    projector: nn.Linear,
    modality_embed: torch.Tensor,
    *,
    target_dtype: torch.dtype,
    k_local: int = 4,
) -> MemoryPrefixContext | None:
    if tokens_by_sample is None or all(token is None for token in tokens_by_sample):
        return None
    offsets = [0]
    projected = []
    present = []
    for token in tokens_by_sample:
        present.append(token is not None)
        if token is None:
            offsets.append(offsets[-1])
            continue
        if token.shape != (k_local, projector.in_features):
            raise ValueError(f"Memory Prefix requires [{k_local},{projector.in_features}] Local tokens")
        if token.device != projector.weight.device:
            raise ValueError("Memory Prefix tokens and projector must share a device")
        projected.append(
            F.linear(
                token.to(target_dtype),
                projector.weight.to(target_dtype),
                projector.bias.to(target_dtype) if projector.bias is not None else None,
            )
            + modality_embed.to(target_dtype)
        )
        offsets.append(offsets[-1] + k_local)
    hidden = torch.cat(projected)
    if hidden.dtype != target_dtype:
        raise ValueError("Memory Prefix projection did not preserve native compute dtype")
    return MemoryPrefixContext(
        hidden=hidden,
        sample_offsets=torch.tensor(offsets, device=hidden.device, dtype=torch.long),
        present=torch.tensor(present, device=hidden.device, dtype=torch.bool),
        k_local=k_local,
    )


def prepend_memory_kv(
    memory_k: torch.Tensor,
    memory_v: torch.Tensor,
    memory_offsets: torch.Tensor,
    native_k: torch.Tensor,
    native_v: torch.Tensor,
    native_offsets: torch.Tensor,
    *,
    max_native_len: int,
    k_local: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Interleave [Local, native] per sample while preserving native query offsets."""
    if memory_k.shape != memory_v.shape or native_k.shape != native_v.shape:
        raise ValueError("Memory Prefix K/V shapes must match")
    if native_offsets.numel() == memory_offsets.numel() + 1:
        # Native packing may add one trailing pad segment; it owns no Local rows.
        memory_offsets = torch.cat((memory_offsets, memory_offsets[-1:]))
    elif memory_offsets.numel() != native_offsets.numel():
        raise ValueError("Memory Prefix and native sample counts must match")
    if memory_k.device != native_k.device or memory_offsets.device != native_offsets.device:
        raise ValueError("Memory Prefix and native K/V must share a device")
    sample_count = memory_offsets.shape[0] - 1
    local_lengths = memory_offsets[1:] - memory_offsets[:-1]
    native_offsets_long = native_offsets.to(torch.long)
    native_lengths = native_offsets_long[1:] - native_offsets_long[:-1]
    combined_offsets = torch.cat(
        (native_offsets_long.new_zeros(1), torch.cumsum(local_lengths + native_lengths, dim=0))
    )
    sample_ids = torch.arange(sample_count, device=native_k.device)
    local_rows = torch.repeat_interleave(sample_ids, local_lengths, output_size=memory_k.shape[0])
    native_rows = torch.repeat_interleave(sample_ids, native_lengths, output_size=native_k.shape[0])
    local_destination = combined_offsets[local_rows] + (
        torch.arange(memory_k.shape[0], device=native_k.device) - memory_offsets[local_rows]
    )
    native_destination = (
        combined_offsets[native_rows]
        + local_lengths[native_rows]
        + (torch.arange(native_k.shape[0], device=native_k.device) - native_offsets_long[native_rows])
    )
    total_shape = (memory_k.shape[0] + native_k.shape[0], *native_k.shape[1:])
    combined_k = native_k.new_empty(total_shape)
    combined_v = native_v.new_empty(total_shape)
    return (
        combined_k.index_copy(0, local_destination, memory_k).index_copy(0, native_destination, native_k),
        combined_v.index_copy(0, local_destination, memory_v).index_copy(0, native_destination, native_v),
        combined_offsets.to(native_offsets.dtype),
        max_native_len + k_local,
    )
