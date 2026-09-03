# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Read-only Local Memory Prefix payloads for two-way packed attention."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


@dataclass(frozen=True)
class MemoryPrefixContext:
    """Per-forward projected Memory Prefix, distinct from native KV cache state."""

    hidden: torch.Tensor
    sample_offsets: torch.Tensor
    present: torch.Tensor

    def validate(self) -> None:
        if self.hidden.ndim != 2:
            raise ValueError(f"Memory Prefix hidden must be rank 2, got {tuple(self.hidden.shape)}.")
        if self.sample_offsets.ndim != 1 or self.sample_offsets.dtype != torch.long:
            raise ValueError("Memory Prefix sample_offsets must be rank-1 torch.long.")
        if self.present.ndim != 1 or self.present.dtype != torch.bool:
            raise ValueError("Memory Prefix present must be rank-1 torch.bool.")
        if self.sample_offsets.numel() != self.present.numel() + 1:
            raise ValueError("Memory Prefix offsets must have exactly one more entry than present.")
        if self.sample_offsets.device != self.hidden.device or self.present.device != self.hidden.device:
            raise ValueError("Memory Prefix hidden, offsets and present must share a device.")
        if self.sample_offsets.numel() == 0 or self.sample_offsets[0].item() != 0:
            raise ValueError("Memory Prefix offsets must start at zero.")
        if self.sample_offsets[-1].item() != self.hidden.shape[0]:
            raise ValueError("Memory Prefix offsets must end at the flattened hidden length.")
        lengths = self.sample_offsets[1:] - self.sample_offsets[:-1]
        if (lengths < 0).any().item() or not torch.equal(self.present, lengths > 0):
            raise ValueError("Memory Prefix present must exactly match positive per-sample lengths.")
        present_lengths = lengths[self.present]
        if present_lengths.numel() == 0 or not torch.equal(present_lengths, present_lengths.new_full(present_lengths.shape, present_lengths[0])):
            raise ValueError("All present Memory Prefix samples must use the same positive K_local.")

    def with_hidden(self, hidden: torch.Tensor) -> "MemoryPrefixContext":
        context = MemoryPrefixContext(hidden=hidden, sample_offsets=self.sample_offsets, present=self.present)
        context.validate()
        return context


def build_memory_prefix_context(
    tokens_by_sample: list[torch.Tensor | None] | None,
    projector: Callable[[torch.Tensor], torch.Tensor],
    modality_embed: torch.Tensor,
    target_dtype: torch.dtype,
) -> MemoryPrefixContext | None:
    """Project Local slots without placing them in the native packed sequence."""
    if tokens_by_sample is None or not any(token is not None for token in tokens_by_sample):
        return None

    present_tokens = [token for token in tokens_by_sample if token is not None]
    assert present_tokens
    for token in present_tokens:
        if token.ndim != 2 or token.shape[0] <= 0:
            raise ValueError(f"Memory Prefix token must have shape [K_local, D_local], got {tuple(token.shape)}.")
    k_local = present_tokens[0].shape[0]
    if any(token.shape[0] != k_local for token in present_tokens):
        raise ValueError("All present Memory Prefix samples must use the same K_local.")

    projected: list[torch.Tensor] = []
    offsets = [0]
    for token in tokens_by_sample:
        if token is None:
            offsets.append(offsets[-1])
            continue
        hidden = projector(token.to(dtype=target_dtype)) + modality_embed.to(dtype=target_dtype)
        projected.append(hidden)
        offsets.append(offsets[-1] + hidden.shape[0])

    hidden_all = torch.cat(projected, dim=0)
    context = MemoryPrefixContext(
        hidden=hidden_all,
        sample_offsets=torch.tensor(offsets, device=hidden_all.device, dtype=torch.long),
        present=torch.tensor([token is not None for token in tokens_by_sample], device=hidden_all.device, dtype=torch.bool),
    )
    context.validate()
    return context


def concat_prefix_with_native_kv(
    prefix_key: torch.Tensor,
    prefix_value: torch.Tensor,
    prefix_offsets: torch.Tensor,
    native_key: torch.Tensor,
    native_value: torch.Tensor,
    native_offsets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Concatenate per-sample ``[MEM, native]`` KV without cross-sample leakage."""
    if prefix_key.shape != prefix_value.shape or native_key.shape != native_value.shape:
        raise ValueError("Memory Prefix and native K/V tensors must have matching shapes.")
    if prefix_offsets.dtype != torch.long or native_offsets.dtype not in (torch.int32, torch.long):
        raise ValueError("Memory Prefix and native offsets must be integer tensors.")
    if prefix_offsets.numel() != native_offsets.numel():
        raise ValueError("Memory Prefix and native offsets must have the same sample count.")
    if prefix_offsets.device != native_offsets.device or prefix_key.device != native_key.device:
        raise ValueError("Memory Prefix/native K/V and offsets must share a device.")

    keys: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    offsets = [0]
    for index in range(prefix_offsets.numel() - 1):
        prefix_start, prefix_end = prefix_offsets[index].item(), prefix_offsets[index + 1].item()
        native_start, native_end = native_offsets[index].item(), native_offsets[index + 1].item()
        key = torch.cat((prefix_key[prefix_start:prefix_end], native_key[native_start:native_end]), dim=0)
        value = torch.cat((prefix_value[prefix_start:prefix_end], native_value[native_start:native_end]), dim=0)
        keys.append(key)
        values.append(value)
        offsets.append(offsets[-1] + key.shape[0])
    key_all = torch.cat(keys, dim=0) if keys else native_key.new_empty((0, *native_key.shape[1:]))
    value_all = torch.cat(values, dim=0) if values else native_value.new_empty((0, *native_value.shape[1:]))
    offsets_tensor = torch.tensor(offsets, device=native_key.device, dtype=native_offsets.dtype)
    return key_all, value_all, offsets_tensor, max((offsets[i + 1] - offsets[i] for i in range(len(offsets) - 1)), default=0)
