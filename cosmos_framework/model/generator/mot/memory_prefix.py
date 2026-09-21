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
    k_local: int = 0

    def validate(self) -> None:
        """Explicit diagnostic validation; may synchronize CUDA tensor values."""
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
        if present_lengths.numel() == 0 or not torch.equal(
            present_lengths, present_lengths.new_full(present_lengths.shape, present_lengths[0])
        ):
            raise ValueError("All present Memory Prefix samples must use the same positive K_local.")
        if self.k_local and not torch.equal(
            present_lengths, present_lengths.new_full(present_lengths.shape, self.k_local)
        ):
            raise ValueError("Memory Prefix k_local metadata must match all present sample lengths.")

    def replace_hidden(self, hidden: torch.Tensor) -> "MemoryPrefixContext":
        """Replace only the hidden payload using metadata-only checks (no value reads)."""
        if hidden.ndim != 2 or hidden.shape != self.hidden.shape:
            raise ValueError(
                f"Memory Prefix replacement hidden must preserve shape {tuple(self.hidden.shape)}, "
                f"got {tuple(hidden.shape)}."
            )
        if hidden.device != self.hidden.device:
            raise ValueError("Memory Prefix replacement hidden must preserve device.")
        if hidden.dtype != self.hidden.dtype:
            raise ValueError("Memory Prefix replacement hidden must preserve dtype.")
        return MemoryPrefixContext(
            hidden=hidden,
            sample_offsets=self.sample_offsets,
            present=self.present,
            k_local=self.k_local,
        )


def _validate_build_metadata(
    offsets: list[int],
    present: list[bool],
    *,
    k_local: int,
    hidden_len: int,
) -> None:
    """Validate immutable prefix invariants using Python metadata only."""
    if not offsets or offsets[0] != 0:
        raise ValueError("Memory Prefix offsets must start at zero.")
    if len(offsets) != len(present) + 1:
        raise ValueError("Memory Prefix offsets must have exactly one more entry than present.")
    if offsets[-1] != hidden_len:
        raise ValueError("Memory Prefix offsets must end at the flattened hidden length.")
    lengths = [end - start for start, end in zip(offsets[:-1], offsets[1:], strict=True)]
    if any(length < 0 for length in lengths):
        raise ValueError("Memory Prefix lengths must be non-negative.")
    if any(is_present != (length > 0) for is_present, length in zip(present, lengths, strict=True)):
        raise ValueError("Memory Prefix present must exactly match positive per-sample lengths.")
    present_lengths = [length for is_present, length in zip(present, lengths, strict=True) if is_present]
    if not present_lengths or any(length != k_local for length in present_lengths):
        raise ValueError("All present Memory Prefix samples must use the same positive K_local.")


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
    present = [token is not None for token in tokens_by_sample]
    for token in tokens_by_sample:
        if token is None:
            offsets.append(offsets[-1])
            continue
        hidden = projector(token.to(dtype=target_dtype)) + modality_embed.to(dtype=target_dtype)
        projected.append(hidden)
        offsets.append(offsets[-1] + hidden.shape[0])

    hidden_all = torch.cat(projected, dim=0)
    _validate_build_metadata(offsets, present, k_local=k_local, hidden_len=hidden_all.shape[0])
    return MemoryPrefixContext(
        hidden=hidden_all,
        sample_offsets=torch.tensor(offsets, device=hidden_all.device, dtype=torch.long),
        present=torch.tensor(present, device=hidden_all.device, dtype=torch.bool),
        k_local=k_local,
    )


def concat_prefix_with_native_kv(
    prefix_key: torch.Tensor,
    prefix_value: torch.Tensor,
    prefix_offsets: torch.Tensor,
    native_key: torch.Tensor,
    native_value: torch.Tensor,
    native_offsets: torch.Tensor,
    *,
    max_native_len: int | None = None,
    max_prefix_len: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Concatenate per-sample [MEM, native] KV without device-to-host value reads."""
    if prefix_key.shape != prefix_value.shape or native_key.shape != native_value.shape:
        raise ValueError("Memory Prefix and native K/V tensors must have matching shapes.")
    if prefix_offsets.dtype != torch.long or native_offsets.dtype not in (torch.int32, torch.long):
        raise ValueError("Memory Prefix and native offsets must be integer tensors.")
    if prefix_offsets.numel() != native_offsets.numel():
        raise ValueError("Memory Prefix and native offsets must have the same sample count.")
    if prefix_offsets.device != native_offsets.device or prefix_key.device != native_key.device:
        raise ValueError("Memory Prefix/native K/V and offsets must share a device.")
    if prefix_key.device != prefix_offsets.device or native_key.device != native_offsets.device:
        raise ValueError("Memory Prefix/native K/V must share devices with their offsets.")

    num_samples = prefix_offsets.shape[0] - 1
    if num_samples == 1:
        key_all = torch.cat((prefix_key, native_key), dim=0)
        value_all = torch.cat((prefix_value, native_value), dim=0)
        offsets_tensor = prefix_offsets.to(dtype=native_offsets.dtype) + native_offsets
        return key_all, value_all, offsets_tensor, prefix_key.shape[0] + native_key.shape[0]

    prefix_offsets_long = prefix_offsets
    native_offsets_long = native_offsets.to(dtype=torch.long)
    prefix_lengths = prefix_offsets_long[1:] - prefix_offsets_long[:-1]
    native_lengths = native_offsets_long[1:] - native_offsets_long[:-1]
    combined_lengths = prefix_lengths + native_lengths
    combined_offsets_long = torch.cat(
        (
            torch.zeros(1, dtype=torch.long, device=native_key.device),
            torch.cumsum(combined_lengths, dim=0),
        )
    )

    sample_ids = torch.arange(num_samples, device=native_key.device, dtype=torch.long)
    prefix_sample_ids = torch.repeat_interleave(
        sample_ids,
        prefix_lengths,
        output_size=prefix_key.shape[0],
    )
    native_sample_ids = torch.repeat_interleave(
        sample_ids,
        native_lengths,
        output_size=native_key.shape[0],
    )

    prefix_local = torch.arange(prefix_key.shape[0], device=native_key.device) - prefix_offsets_long.index_select(
        0, prefix_sample_ids
    )
    native_local = torch.arange(native_key.shape[0], device=native_key.device) - native_offsets_long.index_select(
        0, native_sample_ids
    )
    prefix_dest = combined_offsets_long.index_select(0, prefix_sample_ids) + prefix_local
    native_dest = (
        combined_offsets_long.index_select(0, native_sample_ids)
        + prefix_lengths.index_select(0, native_sample_ids)
        + native_local
    )

    key_all = native_key.new_empty((prefix_key.shape[0] + native_key.shape[0], *native_key.shape[1:]))
    value_all = native_value.new_empty((prefix_value.shape[0] + native_value.shape[0], *native_value.shape[1:]))
    key_all = key_all.index_copy(0, prefix_dest, prefix_key)
    key_all = key_all.index_copy(0, native_dest, native_key)
    value_all = value_all.index_copy(0, prefix_dest, prefix_value)
    value_all = value_all.index_copy(0, native_dest, native_value)

    if max_native_len is None:
        max_native_len = native_key.shape[0]
    if max_prefix_len is None:
        max_prefix_len = prefix_key.shape[0]
    if max_native_len < 0 or max_prefix_len < 0:
        raise ValueError("Memory Prefix max sequence lengths must be non-negative Python metadata.")

    return (
        key_all,
        value_all,
        combined_offsets_long.to(dtype=native_offsets.dtype),
        max_native_len + max_prefix_len,
    )
