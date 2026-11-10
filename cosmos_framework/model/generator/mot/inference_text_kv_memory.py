# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Request-local text K/V reuse for diffusion inference (cosmos3-local).

Caches understanding (text) K/V from the first denoising forward and reuses
them on later steps within the same request.

Supports single-sample packs (dense attention) and multi-sample packs such as
batched CFG (cond+uncond as B=2) via varlen attention with per-sample isolation.
Prefix-aware B1 keeps native varlen attention; the current request's Local
Prefix K/V is supplied separately on every decoder forward. Explicitly bound
frozen requests may reuse native first-fill projections within that request.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from cosmos_framework.model.attention import attention
from cosmos_framework.model.generator.mot.attention import (
    _FrozenPrefixTextKVToStore,
    _PrefixTextKVToStore,
    SplitInfo,
    dispatch_attention,
    two_way_attention,
)
from cosmos_framework.model.generator.utils.memory import KVToStore, MemoryState, MemoryValue
from cosmos_framework.data.generator.sequence_packing.runtime import (
    SequencePack,
    drop_pad_segment,
    from_und_gen_splits,
    get_gen_seq,
)


def prefix_projection_binding(prefixes) -> tuple | None:
    """Host-only identity/version proof; retain the source to prevent ID reuse.

    Unversioned inference tensors and tensor subclasses conservatively retain
    the original path. No tensor values or device synchronization are needed.
    This runs outside compiled decoders, once per sampling step.
    """
    if torch.compiler.is_compiling() or torch.is_grad_enabled() or prefixes is None or len(prefixes) != 1:
        return None
    token = prefixes[0]
    if type(token) is not torch.Tensor or token.requires_grad or token.is_inference():
        return None
    return (
        token,
        token._version,
        token.data_ptr(),
        tuple(token.shape),
        token.stride(),
        token.storage_offset(),
        token.dtype,
        token.device,
    )


class UndKVCache:
    """Fixed per-layer cache for understanding (text) tokens."""

    def __init__(self) -> None:
        self.k_und: torch.Tensor | None = None
        self.v_und: torch.Tensor | None = None
        self.prefix_key_input: torch.Tensor | None = None
        self.prefix_key_output: torch.Tensor | None = None
        self.prefix_value_output: torch.Tensor | None = None
        self.is_initialized = False

    def store(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        prefix_key_input: torch.Tensor | None = None,
        prefix_key_output: torch.Tensor | None = None,
        prefix_value_output: torch.Tensor | None = None,
    ) -> None:
        """Store und K/V. ``k``/``v``: [B,S_und,H,D]."""
        if (prefix_key_input is None) != (prefix_key_output is None):
            raise ValueError("Prefix normalization reference requires input and output")
        if prefix_key_input is not None and (
            prefix_key_input.shape != prefix_key_output.shape or prefix_key_input.device != prefix_key_output.device
        ):
            raise ValueError("Prefix normalization reference shape/device mismatch")
        if prefix_value_output is not None and (
            prefix_key_output is None
            or prefix_value_output.shape != prefix_key_output.shape
            or prefix_value_output.dtype != prefix_key_output.dtype
            or prefix_value_output.device != prefix_key_output.device
        ):
            raise ValueError("Frozen Prefix value reference shape/dtype/device mismatch")
        self.k_und = k.detach().clone()
        self.v_und = v.detach().clone()
        self.prefix_key_input = None if prefix_key_input is None else prefix_key_input.detach().clone()
        self.prefix_key_output = None if prefix_key_output is None else prefix_key_output.detach().clone()
        self.prefix_value_output = None if prefix_value_output is None else prefix_value_output.detach().clone()
        self.is_initialized = True

    def get(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_initialized or self.k_und is None or self.v_und is None:
            raise AssertionError("UndKVCache not initialized — must call store() first")
        return self.k_und, self.v_und

    def clear(self) -> None:
        self.k_und = self.v_und = None
        self.prefix_key_input = self.prefix_key_output = None
        self.prefix_value_output = None
        self.is_initialized = False


@dataclass
class InferenceTextKVMemoryValue(MemoryValue):
    """Read-only snapshot passed into attention for one decoder layer."""

    und_k_cached: torch.Tensor | None
    und_v_cached: torch.Tensor | None
    frame_idx: int
    gen_len: int
    """Total gen tokens across all samples (``_num_full_tokens``)."""
    kv_reorder_indices: torch.Tensor | None = None
    """Indices that turn ``[all und|all gen]`` into ``[und_i|gen_i|...]``."""
    cumulative_seqlen_q: torch.Tensor | None = None
    cumulative_seqlen_kv: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_kv: int = 0
    for_cuda_graphs: bool = False
    local_prefix_length: int = 0
    causal_pad_length: int = 0
    prefix_key_input_cached: torch.Tensor | None = None
    prefix_key_output_cached: torch.Tensor | None = None
    prefix_value_output_cached: torch.Tensor | None = None
    reuse_prefix_projections: bool = False


class InferenceTextKVMemoryState(MemoryState):
    """Request-local memory state that reuses static text K/V during diffusion inference."""

    def __init__(
        self,
        text_kv_cache: list[UndKVCache],
        *,
        local_prefix_length: int = 0,
        frozen_prefix_binding: tuple | None = None,
        current_prefixes=None,
    ) -> None:
        if type(local_prefix_length) is not int or local_prefix_length < 0:
            raise ValueError("local_prefix_length must be a nonnegative integer")
        self._text_kv_cache = text_kv_cache
        self.local_prefix_length = local_prefix_length
        current = prefix_projection_binding(current_prefixes) if frozen_prefix_binding is not None else None
        self.reuse_prefix_projections = (
            local_prefix_length > 0
            and current is not None
            and frozen_prefix_binding[0] is current[0]
            and frozen_prefix_binding[1:] == current[1:]
        )
        self._causal_pad_length = 0
        self._gen_len = 0
        self._kv_reorder_indices: torch.Tensor | None = None
        self._cumulative_seqlen_q: torch.Tensor | None = None
        self._cumulative_seqlen_kv: torch.Tensor | None = None
        self._max_seqlen_q = 0
        self._max_seqlen_kv = 0

    def init(self, hidden_states: dict, device: torch.device) -> None:
        self._gen_len = int(hidden_states["_num_full_tokens"])
        # Precompute the multi-sample layout outside the compiled decoder layers.
        # Doing this here avoids converting CUDA offsets to Python lists and
        # rebuilding the same chunked K/V layout once per layer.
        # Real-sample offsets. Everything below counts samples off these -- the ``numel() > 2``
        # multi-sample test, the per-sample lengths, and the varlen ranges handed to attention --
        # and the towers carry the trailing padding as one extra segment, which would read as a
        # further sample: a single-sample pack would take the multi-sample path and give the
        # padding its own varlen range.
        und_off = hidden_states.get("_causal_seq_offsets")
        gen_off = hidden_states.get("_full_only_seq_offsets")
        if isinstance(und_off, torch.Tensor):
            und_off = drop_pad_segment(hidden_states, und_off)
        if isinstance(gen_off, torch.Tensor):
            gen_off = drop_pad_segment(hidden_states, gen_off)
        if isinstance(und_off, torch.Tensor) and isinstance(gen_off, torch.Tensor) and und_off.shape != gen_off.shape:
            raise RuntimeError(f"und/gen sample counts disagree: und={und_off.numel() - 1} gen={gen_off.numel() - 1}")
        if self.local_prefix_length:
            if und_off is None or gen_off is None or und_off.numel() != 2 or gen_off.numel() != 2:
                raise ValueError("Prefix-aware text K/V requires a single native sample")
            # Preserve the real native tower layout, including its pad segment.
            self._causal_pad_length = hidden_states["causal_seq"].shape[0] - int(hidden_states["_num_causal_tokens"])
            return
        if isinstance(und_off, torch.Tensor) and isinstance(gen_off, torch.Tensor) and und_off.numel() > 2:
            und_off = und_off.to(dtype=torch.int32)
            gen_off = gen_off.to(dtype=torch.int32)
            und_lens = und_off[1:] - und_off[:-1]
            gen_lens = gen_off[1:] - gen_off[:-1]
            sample_ids = torch.arange(und_lens.numel(), dtype=torch.int64, device=und_off.device)
            und_sample_ids = torch.repeat_interleave(sample_ids, und_lens.to(dtype=torch.int64))
            gen_sample_ids = torch.repeat_interleave(sample_ids, gen_lens.to(dtype=torch.int64))

            # Stable sorting preserves UND-before-GEN within each sample because
            # the source stream is [all UND | all GEN].
            self._kv_reorder_indices = torch.argsort(
                torch.cat((und_sample_ids, gen_sample_ids)),
                stable=True,
            )
            self._cumulative_seqlen_q = gen_off
            self._cumulative_seqlen_kv = und_off + gen_off
            self._max_seqlen_q = int(gen_lens.max())
            self._max_seqlen_kv = int((und_lens + gen_lens).max())
        else:
            # Single-sample: keep the legacy dense path.
            self._kv_reorder_indices = None
            self._cumulative_seqlen_q = None
            self._cumulative_seqlen_kv = None
            self._max_seqlen_q = 0
            self._max_seqlen_kv = 0

    def read_for_layer(self, layer_idx: int) -> InferenceTextKVMemoryValue:
        cache = self._text_kv_cache[layer_idx]
        if self.reuse_prefix_projections and cache.is_initialized and cache.prefix_value_output is None:
            raise ValueError("Frozen Prefix projection cache is incomplete")
        if cache.is_initialized:
            und_k_cached, und_v_cached = cache.get()  # und_k/v: [1,N_und,H_kv,D]
        else:
            und_k_cached, und_v_cached = None, None

        return InferenceTextKVMemoryValue(
            und_k_cached=und_k_cached,
            und_v_cached=und_v_cached,
            frame_idx=1 if cache.is_initialized else 0,
            gen_len=self._gen_len,
            kv_reorder_indices=self._kv_reorder_indices,
            cumulative_seqlen_q=self._cumulative_seqlen_q,
            cumulative_seqlen_kv=self._cumulative_seqlen_kv,
            max_seqlen_q=self._max_seqlen_q,
            max_seqlen_kv=self._max_seqlen_kv,
            for_cuda_graphs=False,
            local_prefix_length=self.local_prefix_length,
            causal_pad_length=self._causal_pad_length,
            prefix_key_input_cached=cache.prefix_key_input,
            prefix_key_output_cached=cache.prefix_key_output,
            prefix_value_output_cached=cache.prefix_value_output,
            reuse_prefix_projections=self.reuse_prefix_projections,
        )

    def write_for_layer(self, layer_idx: int, kv_to_store: KVToStore) -> None:
        cache = self._text_kv_cache[layer_idx]
        if cache.is_initialized:
            return
        # Prefix-aware fill tags the otherwise unused GEN slots explicitly;
        # ordinary text-cache captures retain the original GEN/UND tuple.
        prefix_value = None
        if isinstance(kv_to_store, _FrozenPrefixTextKVToStore):
            if (
                not self.reuse_prefix_projections
                or kv_to_store.native_key_output.ndim != 4
                or kv_to_store.native_key_output.shape[0] != 1
                or kv_to_store.native_key_output.shape != kv_to_store.native_value_output.shape
                or kv_to_store.native_key_output.shape[1] < self.local_prefix_length
                or kv_to_store.und_indices.ndim != 1
            ):
                raise ValueError("Frozen Prefix projection capture requires a bound native K/V pair")
            prefix_key = kv_to_store.native_key_output[:, : self.local_prefix_length]
            prefix_value = kv_to_store.native_value_output[:, : self.local_prefix_length]
            kv_to_store = _PrefixTextKVToStore(
                kv_to_store.prefix_key_input,
                prefix_key,
                kv_to_store.native_key_output.index_select(1, kv_to_store.und_indices),
                kv_to_store.native_value_output.index_select(1, kv_to_store.und_indices),
            )
        _gen_k, _gen_v, und_k, und_v = kv_to_store
        if isinstance(kv_to_store, _PrefixTextKVToStore):
            if (
                self.local_prefix_length <= 0
                or kv_to_store.prefix_key_input.ndim != 4
                or kv_to_store.prefix_key_input.shape[0] != 1
                or kv_to_store.prefix_key_input.shape[1] != self.local_prefix_length
            ):
                raise ValueError("Prefix normalization reference requires the current B1 Prefix")
            cache.store(
                und_k,
                und_v,
                prefix_key_input=kv_to_store.prefix_key_input,
                prefix_key_output=kv_to_store.prefix_key_output,
                prefix_value_output=prefix_value,
            )
        else:
            cache.store(und_k, und_v)

    def is_gen_only(self) -> bool:
        # Require every layer: layer-0 alone can be True after a partial first forward (e.g. OOM mid-stack).
        return all(cache.is_initialized for cache in self._text_kv_cache)

    def requires_natten_metadata(self) -> bool:
        return False


def make_inference_text_kv_cache(num_layers: int) -> list[UndKVCache]:
    """Create per-layer request-local text K/V caches for one CFG branch."""
    return [UndKVCache() for _ in range(num_layers)]


def _attention_gen_with_cached_text(
    packed_query_states: SequencePack,
    packed_key_states: SequencePack,
    packed_value_states: SequencePack,
    memory_value: InferenceTextKVMemoryValue,
) -> tuple[SequencePack, KVToStore | None]:
    """Gen-only attention attending to cached text K/V plus current gen K/V.

    Single-sample packs use the dense attention API. Multi-sample packs (e.g.
    batched CFG with cond+uncond) rebuild KV in sample-major order
    ``[und_0|gen_0|und_1|gen_1|...]`` and call varlen attention so sample ``i``'s
    gen queries only see sample ``i``'s und+gen keys.
    """
    q_gen = get_gen_seq(packed_query_states)  # [S_curr, H, D]
    k_gen = get_gen_seq(packed_key_states)  # [S_curr, H_kv, D]
    v_gen = get_gen_seq(packed_value_states)  # [S_curr, H_kv, D]

    gen_len = memory_value.gen_len
    k_curr = k_gen[:gen_len].unsqueeze(0)  # [1, S_gen_real, H_kv, D]
    v_curr = v_gen[:gen_len].unsqueeze(0)  # [1, S_gen_real, H_kv, D]
    q_curr = q_gen.unsqueeze(0)  # [1, S_curr, H, D]

    kv_reorder_indices = memory_value.kv_reorder_indices
    cumulative_seqlen_q = memory_value.cumulative_seqlen_q
    cumulative_seqlen_kv = memory_value.cumulative_seqlen_kv
    multi_sample = (
        kv_reorder_indices is not None
        and cumulative_seqlen_q is not None
        and cumulative_seqlen_kv is not None
        and memory_value.und_k_cached is not None
    )

    if not multi_sample:
        kv_parts_k = [k_curr]
        kv_parts_v = [v_curr]
        if memory_value.und_k_cached is not None:
            assert memory_value.und_v_cached is not None
            kv_parts_k.insert(0, memory_value.und_k_cached)
            kv_parts_v.insert(0, memory_value.und_v_cached)

        k_full = torch.cat(kv_parts_k, dim=1)  # [1, S_total, H_kv, D]
        v_full = torch.cat(kv_parts_v, dim=1)  # [1, S_total, H_kv, D]

        attn_result = attention(
            query=q_curr,
            key=k_full,
            value=v_full,
            is_causal=False,
            return_lse=False,
        )
    else:
        assert memory_value.und_v_cached is not None
        assert kv_reorder_indices is not None
        assert cumulative_seqlen_q is not None and cumulative_seqlen_kv is not None
        und_k = memory_value.und_k_cached  # [1, N_und, H_kv, D]
        und_v = memory_value.und_v_cached
        k_full = torch.cat((und_k, k_curr), dim=1).index_select(1, kv_reorder_indices)
        v_full = torch.cat((und_v, v_curr), dim=1).index_select(1, kv_reorder_indices)

        attn_result = attention(
            query=q_curr[:, :gen_len],
            key=k_full,
            value=v_full,
            is_causal=False,
            return_lse=False,
            cumulative_seqlen_Q=cumulative_seqlen_q,
            cumulative_seqlen_KV=cumulative_seqlen_kv,
            max_seqlen_Q=memory_value.max_seqlen_q,
            max_seqlen_KV=memory_value.max_seqlen_kv,
        )

    assert isinstance(attn_result, torch.Tensor)
    gen_out = attn_result.squeeze(0).flatten(-2, -1)  # [S_gen_real, H*D]
    # Match legacy behavior: padded query rows (if any) stay in q_gen but only
    # the real prefix is produced by attention; zero-fill the tail so padded
    # slots stay finite (new_empty can leave NaN/Inf from recycled BF16 memory).
    if gen_out.shape[0] < q_gen.shape[0]:
        padded = q_gen.new_zeros(q_gen.shape[0], gen_out.shape[-1])
        padded[: gen_out.shape[0]] = gen_out
        gen_out = padded

    output = from_und_gen_splits(
        gen_out.new_empty(0, gen_out.shape[-1]),
        gen_out,
        packed_query_states,
    )
    return output, None


def dispatch_attention_with_text_kv_memory(
    packed_query_states: SequencePack,
    packed_key_states: SequencePack,
    packed_value_states: SequencePack,
    attention_mask: object | SplitInfo,
    natten_metadata: dict | None = None,
    memory_value: MemoryValue | None = None,
    packed_key_states_normalized: SequencePack | None = None,
    memory_prefix_key_states: torch.Tensor | None = None,
    memory_prefix_value_states: torch.Tensor | None = None,
    memory_prefix_sample_offsets: torch.Tensor | None = None,
    memory_prefix_max_len: int = 4,
    memory_prefix_key_source_states: torch.Tensor | None = None,
) -> tuple[SequencePack, KVToStore | None]:
    """Dispatch attention with optional request-local text K/V reuse.

    Falls through to standard ``dispatch_attention`` when ``memory_value`` is
    ``None`` or still on the first (cache-fill) step.
    """
    has_prefix = memory_prefix_key_states is not None
    prefix_aware = isinstance(memory_value, InferenceTextKVMemoryValue) and memory_value.local_prefix_length > 0
    if memory_prefix_key_source_states is not None and not prefix_aware:
        raise ValueError("Prefix normalization input requires Prefix-aware text K/V")
    # Branch on SymBool directly. SymBool != Python bool can fail in Dynamo's
    # symbolic comparison (BooleanAtom), even when Prefix length is backed.
    if prefix_aware:
        if not has_prefix:
            raise ValueError("Prefix-aware text K/V requires the current request's Prefix on every step")
    elif has_prefix:
        raise ValueError("Prefix-aware text K/V requires the current request's Prefix on every step")
    if prefix_aware:
        if (
            torch.is_grad_enabled()
            or natten_metadata is not None
            or attention_mask.is_three_way
            or attention_mask.control_stream_token_ranges is not None
            or getattr(attention_mask, "flex_block_mask", None) is not None
            or getattr(attention_mask, "multiview_maskless", None) is not None
        ):
            raise ValueError("Prefix-aware text K/V supports only native two_way inference")
        if (
            memory_prefix_value_states is None
            or memory_prefix_sample_offsets is None
            or memory_prefix_sample_offsets.numel() != 2
            or memory_prefix_key_states.shape != memory_prefix_value_states.shape
            or memory_prefix_key_states.shape[0] != memory_value.local_prefix_length
            or memory_prefix_max_len != memory_value.local_prefix_length
        ):
            raise ValueError("Prefix-aware text K/V requires one complete native Prefix")
        if memory_prefix_key_source_states is not None and (
            memory_prefix_key_source_states.shape != memory_prefix_key_states.shape
            or memory_prefix_key_source_states.device != memory_prefix_key_states.device
        ):
            raise ValueError("Prefix normalization input must preserve key shape/device")
        if memory_value.frame_idx > 0:
            return _attention_gen_with_cached_text_and_prefix(
                packed_query_states,
                packed_key_states,
                packed_value_states,
                memory_value,
                memory_prefix_key_states,
                memory_prefix_value_states,
                attention_mask,
                memory_prefix_sample_offsets,
                memory_prefix_key_source_states,
            )
        return two_way_attention(
            packed_query_states,
            packed_key_states,
            packed_value_states,
            packed_key_states_normalized=packed_key_states_normalized,
            memory_prefix_key_states=memory_prefix_key_states,
            memory_prefix_value_states=memory_prefix_value_states,
            memory_prefix_sample_offsets=memory_prefix_sample_offsets,
            memory_prefix_max_len=memory_prefix_max_len,
            _capture_prefix_text_kv=True,
            memory_prefix_key_source_states=memory_prefix_key_source_states,
            _capture_prefix_projections=memory_value.reuse_prefix_projections,
        )
    if isinstance(memory_value, InferenceTextKVMemoryValue) and memory_value.frame_idx > 0:
        return _attention_gen_with_cached_text(
            packed_query_states,
            packed_key_states,
            packed_value_states,
            memory_value,
        )
    return dispatch_attention(
        packed_query_states,
        packed_key_states,
        packed_value_states,
        attention_mask,
        natten_metadata=natten_metadata,
        memory_value=None,
        packed_key_states_normalized=packed_key_states_normalized,
        memory_prefix_key_states=memory_prefix_key_states,
        memory_prefix_value_states=memory_prefix_value_states,
        memory_prefix_sample_offsets=memory_prefix_sample_offsets,
        memory_prefix_max_len=memory_prefix_max_len,
    )


def _attention_gen_with_cached_text_and_prefix(
    query: SequencePack,
    key: SequencePack,
    value: SequencePack,
    memory_value: InferenceTextKVMemoryValue,
    prefix_key: torch.Tensor,
    prefix_value: torch.Tensor,
    attention_mask: SplitInfo,
    prefix_offsets: torch.Tensor,
    prefix_key_source: torch.Tensor | None = None,
) -> tuple[SequencePack, KVToStore]:
    """Reuse native GEN attention with current Prefix, cached UND and current GEN.

    UND never attends to Prefix or GEN, so its per-layer K/V is static.
    Prefix K/V comes from current projections or the same request's explicitly
    bound native projection capture. The normalization reference is unchanged.
    Reuse native fill's normalization rounding only for an identical projected
    key input. Changed input uses current normalization; V is always current.
    The reference is request-local and never stored in a Session.
    """
    und_key, und_value = memory_value.und_k_cached, memory_value.und_v_cached
    if und_key is None or und_value is None:
        raise ValueError("Prefix-aware generation requires initialized text K/V")
    source = memory_value.prefix_key_input_cached
    reference = memory_value.prefix_key_output_cached
    if (source is None) != (reference is None):
        raise ValueError("Incomplete Prefix normalization reference")
    if source is not None:
        if (
            prefix_key_source is None
            or source.ndim != 4
            or reference.ndim != 4
            or source.shape[0] != 1
            or reference.shape[0] != 1
            or source.shape[1:] != prefix_key_source.shape
            or reference.shape[1:] != prefix_key.shape
            or source.dtype != prefix_key_source.dtype
            or reference.dtype != prefix_key.dtype
            or source.device != prefix_key_source.device
            or reference.device != prefix_key.device
        ):
            raise ValueError("Prefix normalization input/reference identity mismatch")
        # No tolerance, host synchronization or ignored current Prefix. This
        # data-dependent choice also keeps the changed-input calculation live.
        same_input = torch.all(prefix_key_source == source[0])
        prefix_key = torch.where(same_input, reference[0], prefix_key)
    q_gen = get_gen_seq(query)
    pad = memory_value.causal_pad_length
    und_k = torch.nn.functional.pad(und_key[0], (0, 0, 0, 0, 0, pad))
    und_v = torch.nn.functional.pad(und_value[0], (0, 0, 0, 0, 0, pad))
    dummy_und_q = q_gen.new_zeros(und_k.shape[0], q_gen.shape[1], q_gen.shape[2])
    result, _ = dispatch_attention(
        from_und_gen_splits(dummy_und_q, q_gen, query),
        from_und_gen_splits(und_k, get_gen_seq(key), key),
        from_und_gen_splits(und_v, get_gen_seq(value), value),
        attention_mask,
        memory_prefix_key_states=prefix_key,
        memory_prefix_value_states=prefix_value,
        memory_prefix_sample_offsets=prefix_offsets,
        memory_prefix_max_len=memory_value.local_prefix_length,
    )
    gen_out = get_gen_seq(result)
    empty_k = und_key[:, :0]
    empty_v = und_value[:, :0]
    # The initialized cache ignores writes. A non-None sentinel also prevents
    # PackedAttentionMoT from exporting unused pre-scatter GEN K/V, which would
    # change the compiled precision boundary. Dummy UND queries are discarded.
    return from_und_gen_splits(gen_out.new_empty(0, gen_out.shape[-1]), gen_out, query), (
        empty_k,
        empty_v,
        empty_k,
        empty_v,
    )


def install_inference_memory_attention_dispatch(
    net: torch.nn.Module,
) -> list[tuple[torch.nn.Module, object]]:
    """Temporarily install text-KV memory attention dispatch.

    Returns prior ``(attn_module, dispatch_fn)`` pairs for restore. Refuses to
    overwrite an unexpected dispatcher (e.g. CP).
    """
    previous: list[tuple[torch.nn.Module, object]] = []
    try:
        for layer in net.language_model.model.layers:
            attn = layer.self_attn
            current = attn.dispatch_attention_fn
            if current is not dispatch_attention and current is not dispatch_attention_with_text_kv_memory:
                current_name = getattr(current, "__name__", type(current).__name__)
                raise RuntimeError(
                    "Cannot install memory-aware attention dispatch over "
                    f"{current_name}; request-local text K/V reuse requires the default "
                    "dispatch_attention (the CP path is excluded by its eligibility guard)."
                )
            previous.append((attn, current))
            attn.dispatch_attention_fn = dispatch_attention_with_text_kv_memory
    except Exception:
        for attn, previous_fn in previous:
            attn.dispatch_attention_fn = previous_fn
        raise
    return previous


def restore_inference_attention_dispatch(previous: list[tuple[torch.nn.Module, object]]) -> None:
    """Restore attention dispatch functions saved by ``install_inference_memory_attention_dispatch``."""
    for attn, previous_fn in previous:
        attn.dispatch_attention_fn = previous_fn
