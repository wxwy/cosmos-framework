# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Out-of-band Local K/V prefix for native two-way generation attention."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from torch import nn
from torch.nn import functional as F

from cosmos_framework.data.generator.sequence_packing.sequence import SequencePlan
from cosmos_framework.model.generator.mot.g2_contracts import GENLayerKV
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
    global_lengths: tuple[int, ...] | None = None
    local_lengths: tuple[int, ...] | None = None
    global_present: torch.Tensor | None = None
    local_present: torch.Tensor | None = None
    max_prefix_len: int | None = None
    global_kv: tuple[GENLayerKV, ...] | None = None
    # Immutable routing indices for the cached-Global inference path. Construct
    # them before decoder compilation; never slice a symbolic zero-row Local
    # tensor by Python sample lengths inside the compiled attention block.
    cached_local_rows: torch.Tensor | None = None
    cached_joint_rows: torch.Tensor | None = None

    @property
    def max_length(self) -> int:
        return self.k_local if self.max_prefix_len is None else self.max_prefix_len

    def replace_hidden(self, hidden: torch.Tensor) -> MemoryPrefixContext:
        if hidden.shape != self.hidden.shape or hidden.device != self.hidden.device:
            raise ValueError("Memory Prefix normalized hidden must preserve shape and device")
        return replace(self, hidden=hidden)

    def for_layer(self, index: int) -> MemoryPrefixContext:
        return self if self.global_kv is None else replace(self, global_kv=(self.global_kv[index],))


def build_joint_memory_prefix_context(
    global_tokens: tuple[torch.Tensor | None, ...] | None,
    local_context: MemoryPrefixContext | None,
    *,
    target_dtype: torch.dtype,
    k_local: int = 4,
    global_kv: tuple[GENLayerKV, ...] | None = None,
) -> MemoryPrefixContext | None:
    """A's projected Global rows plus existing Local, interleaved per sample.

    The legacy Local-only object is returned unchanged when Global is absent.
    k_local keeps its old meaning; max_length bounds the combined K/V prefix.
    """
    if global_tokens is None or all(t is None for t in global_tokens):
        return local_context
    if local_context is not None and len(global_tokens) + 1 != local_context.sample_offsets.numel():
        raise ValueError("Global/Local sample counts differ")
    reference = next(t for t in global_tokens if t is not None)
    hidden_parts, offsets, g_lengths, l_lengths = [], [0], [], []
    for i, global_token in enumerate(global_tokens):
        g_len = 0 if global_token is None else global_token.shape[0]
        if global_token is not None:
            if (
                global_token.ndim != 2
                or global_token.shape != (64, reference.shape[-1])
                or global_token.device != reference.device
                or global_token.dtype != target_dtype
            ):
                raise ValueError("Global projector must provide [64,D_model] in native device/dtype")
            hidden_parts.append(global_token)
        if local_context is None:
            l_len = 0
        else:
            if local_context.hidden.device != reference.device or local_context.hidden.dtype != target_dtype:
                raise ValueError("Global/Local device/dtype mismatch")
            # Offsets are metadata constructed outside compile, not data-dependent GPU scalars.
            if local_context.local_lengths is None:
                raise ValueError("joint prefix requires Local length metadata from the native builder")
            l_len = local_context.local_lengths[i]
            if l_len:
                # Local rows occur in sample order; tensor slicing retains the scan graph.
                start = sum(l_lengths)
                hidden_parts.append(local_context.hidden[start : start + l_len])
        g_lengths.append(g_len)
        l_lengths.append(l_len)
        offsets.append(offsets[-1] + g_len + l_len)
    hidden = torch.cat(hidden_parts)
    local_indices = joint_indices = None
    if global_kv is not None:
        # The combined hidden layout is [G_i, L_i] per sample, while cached
        # Global and freshly projected Local K/V are each packed separately.
        # Gather Local rows, then reorder [all Global, all Local] into the
        # exact original sample order. Empty Local is a real zero-row tensor.
        local_rows, joint_rows = [], []
        g_total = sum(g_lengths)
        hi = gi = li = 0
        for g, loc in zip(g_lengths, l_lengths, strict=True):
            local_rows.extend(range(hi + g, hi + g + loc))
            joint_rows.extend(range(gi, gi + g))
            joint_rows.extend(range(g_total + li, g_total + li + loc))
            hi, gi, li = hi + g + loc, gi + g, li + loc
        local_indices = torch.tensor(local_rows, device=hidden.device, dtype=torch.long)
        joint_indices = torch.tensor(joint_rows, device=hidden.device, dtype=torch.long)
    global_present = torch.tensor([n > 0 for n in g_lengths], device=hidden.device)
    local_present = torch.tensor([n > 0 for n in l_lengths], device=hidden.device)
    return MemoryPrefixContext(
        hidden=hidden,
        sample_offsets=torch.tensor(offsets, device=hidden.device),
        present=global_present | local_present,
        k_local=k_local,
        global_lengths=tuple(g_lengths),
        local_lengths=tuple(l_lengths),
        global_present=global_present,
        local_present=local_present,
        max_prefix_len=max(g + loc for g, loc in zip(g_lengths, l_lengths, strict=True)),
        global_kv=global_kv,
        cached_local_rows=local_indices,
        cached_joint_rows=joint_indices,
    )


def cached_global_local_kv(
    context: MemoryPrefixContext,
    local_key: torch.Tensor,
    local_value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reassemble per-sample cached Global and newly projected Local rows.

    Routing tensors are prepared with the prefix, outside decoder compile.
    The compiled operation has no Python integer-dependent variable slices.
    """
    cached = context.global_kv[0]
    rows = context.cached_joint_rows
    return (
        torch.cat((cached.key, local_key), dim=0).index_select(0, rows),
        torch.cat((cached.value, local_value), dim=0).index_select(0, rows),
    )


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
        local_lengths=tuple(k_local if flag else 0 for flag in present),
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
    if (
        not torch.is_grad_enabled()
        and memory_offsets.ndim == native_offsets.ndim == 1
        and memory_offsets.numel() == 2
        and native_offsets.numel() in (2, 3)
        and memory_offsets.dtype in (torch.int32, torch.int64)
        and native_offsets.dtype in (torch.int32, torch.int64)
        and memory_k.shape[0] == k_local
        and k_local > 0
        and memory_k.dtype == native_k.dtype
        and memory_v.dtype == native_v.dtype
        and memory_k.device == native_k.device
        and memory_offsets.device == native_offsets.device == native_k.device
    ):
        # Native offsets are zero-based. With one real sample, all Local rows
        # precede all native rows, including any trailing pad segment. Preserve
        # that segment's boundary without expanding sample IDs or scattering.
        # Grad-enabled training and other layouts keep the original routine.
        return (
            torch.cat((memory_k, native_k)),
            torch.cat((memory_v, native_v)),
            torch.cat((native_offsets[:1], native_offsets[1:] + k_local)),
            max_native_len + k_local,
        )
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
