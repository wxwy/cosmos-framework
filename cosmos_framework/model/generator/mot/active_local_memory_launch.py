# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Launch-time assembly for the active Local-Memory production route.

The canonical segment ABI (``local_memory_segment``), its owner
(``canonical_segment_runtime``), its active wiring (``production_active_wiring``),
its native forward (``omni_mot_model._active_local_memory_forward``), its data
side (``canonical_local_memory_producer``), and its window driver
(``active_local_memory_driver``) all exist independently -- nothing owned the
seam that turns them into one training run.  This module is that owner.

Two facts shape the design:

* The adapter must wrap the model's **already-registered** modules.  A second
  encoder/core copy would train weights the optimizer never sees, and
  ``validate_slow_inventory`` rejects it outright.
* ``dataloader_train`` is a local of ``ImaginaireTrainer.train`` rather than an
  attribute, so a callback cannot reach the datasets through the trainer.  They
  are handed over at config-build time instead.

The outer loader's batch content is never consumed on this route
(``_active_local_memory_forward`` reads only the two ``psm_local_memory_*``
markers), so the datasets here exist purely to feed the canonical producer.
"""

from __future__ import annotations

import os
from typing import Any, Mapping

import torch
import torch.distributed as dist

from cosmos_framework.utils.callback import Callback

from .active_local_memory_driver import ActiveLocalMemoryWindowDriver
from .canonical_segment_runtime import CanonicalSegmentRuntimeOwner
from .config_checkpoint_contract import canonical_slow_inventory, validate_slow_inventory
from .local_evidence import (
    CANONICAL_EVIDENCE_FEATURE_CONFIG,
    ContinualTTTLocalMemoryCore,
    LocalEvidenceEncoder,
)
from .local_memory_segment import RankLocalSegmentScheduler
from .local_memory_segment_adapter import CanonicalLocalMemorySegmentAdapter, LocalMemorySegmentSidecar
from .production_active_wiring import ProductionActiveWiringRegistry
from .production_segment_wiring import CanonicalSegmentWiring

DEFAULT_B_STREAM = 8
DEFAULT_TBPTT_STEPS = 16


def canonical_runtime_module(model: Any) -> torch.nn.Module:
    """Return ``net.local_memory_runtime``, or fail closed."""
    runtime = getattr(getattr(model, "net", None), "local_memory_runtime", None)
    if not isinstance(runtime, torch.nn.Module):
        raise RuntimeError("active Local-Memory requires the registered net.local_memory_runtime owner")
    return runtime


def canonical_segment_adapter_from_model(model: Any) -> CanonicalLocalMemorySegmentAdapter:
    """Return only an adapter bound to the model's already-registered modules.

    Mirrors ``_canonical_production_adapter_from_model``: resolve the registered
    encoder/core, validate their identity, build the adapter once, cache it on the
    model, and re-validate the binding on every later call so a swapped module
    cannot silently keep training the stale copy.
    """
    runtime = canonical_runtime_module(model)
    encoder = getattr(runtime, "evidence_encoder", None)
    core = getattr(runtime, "ttt_core", None)
    if not isinstance(encoder, LocalEvidenceEncoder) or not isinstance(core, ContinualTTTLocalMemoryCore):
        raise RuntimeError("active Local-Memory requires the registered canonical encoder and TTT core")
    if encoder.feature_config is not CANONICAL_EVIDENCE_FEATURE_CONFIG:
        raise RuntimeError("active Local-Memory encoder has a non-canonical feature config")
    validate_slow_inventory(runtime, runtime_encoder=encoder, runtime_core=core)
    adapter = getattr(model, "_canonical_segment_adapter", None)
    if adapter is None:
        adapter = CanonicalLocalMemorySegmentAdapter(encoder, core, LocalMemorySegmentSidecar())
        model._canonical_segment_adapter = adapter
    if (
        not isinstance(adapter, CanonicalLocalMemorySegmentAdapter)
        or adapter.encoder is not encoder
        or adapter.core is not core
    ):
        raise RuntimeError("canonical segment adapter is not bound to the registered Local-Memory owner")
    return adapter


def canonical_slow_parameters_from_model(model: Any) -> tuple[torch.nn.Parameter, ...]:
    """Return the whole trained slow set, in optimizer-selector order.

    ``CanonicalSegmentRuntimeOwner`` clears these on a grad-scaler skip and on
    every abort path, so the tuple must cover exactly the parameters this route
    trains -- the four selector groups, not merely the encoder and core.
    """
    runtime = canonical_runtime_module(model)
    net = getattr(model, "net", None)
    projector = getattr(net, "local_memory2llm", None)
    modality = getattr(net, "local_memory_modality_embed", None)
    if projector is None or modality is None:
        raise RuntimeError("active Local-Memory requires local_memory2llm and local_memory_modality_embed")
    validate_slow_inventory(
        runtime,
        runtime_encoder=getattr(runtime, "evidence_encoder", None),
        runtime_core=getattr(runtime, "ttt_core", None),
    )
    inventory = canonical_slow_inventory(runtime, projector, modality)
    return tuple(inventory.values())


class SuiteRoutedSegmentProducer:
    """Route each stream to its own suite's producer.

    A producer is bound to one dataset and therefore to one category, while the
    window driver owns a single producer for its whole slot catalog.  This router
    closes that gap and satisfies the driver's duck-typed surface
    (``block_count``, ``produce``, ``ttt_tbptt_steps``, ``source_digest``).
    """

    def __init__(self, producers: Mapping[str, Any], *, source_digest: str) -> None:
        if not producers:
            raise ValueError("active Local-Memory requires at least one suite producer")
        if not source_digest:
            raise ValueError("active Local-Memory requires a non-empty source digest")
        widths = {int(producer.ttt_tbptt_steps) for producer in producers.values()}
        if len(widths) != 1:
            raise ValueError("suite producers disagree on the TBPTT width")
        self.producers = dict(producers)
        self.ttt_tbptt_steps = widths.pop()
        self.source_digest = source_digest

    def _route(self, stream: Any) -> Any:
        producer = self.producers.get(stream.category)
        if producer is None:
            raise ValueError(f"active Local-Memory has no producer for category {stream.category!r}")
        return producer

    def block_count(self, stream: Any) -> int:
        return int(self._route(stream).block_count(stream))

    def produce(self, stream: Any, *, cursor: int) -> Any:
        return self._route(stream).produce(stream, cursor=cursor)


def canonical_segment_streams(producers: Mapping[str, Any], *, b_stream: int) -> tuple[Any, ...]:
    """Enumerate every whole-block episode of every suite into one slot catalog.

    Entries sharing a ``slot_id`` are consumed by the driver in catalog order, so
    each suite's episodes are spread round-robin over that suite's slots.  An
    episode with fewer frames than one TBPTT block is skipped rather than
    admitted, because the frozen plan only ever covers whole blocks.
    """
    from cosmos_framework.data.generator.action.datasets.canonical_local_memory_producer import (
        CanonicalSegmentStream,
    )

    categories = tuple(sorted(producers))
    if not categories:
        raise ValueError("active Local-Memory requires at least one suite producer")
    if b_stream < len(categories):
        raise ValueError("active Local-Memory needs at least one slot per suite")
    by_slot: dict[int, list[Any]] = {slot: [] for slot in range(b_stream)}
    for index, category in enumerate(categories):
        producer = producers[category]
        slots = tuple(slot for slot in range(b_stream) if slot % len(categories) == index)
        ep_vals = producer.frame_source._ep_vals
        admitted = 0
        for position in range(len(ep_vals)):
            episode_index = int(ep_vals[position])
            probe = CanonicalSegmentStream(
                slot_id=slots[0], episode_index=episode_index, episode_position=position, category=category
            )
            if producer.block_count(probe) <= 0:
                continue
            slot = slots[admitted % len(slots)]
            admitted += 1
            by_slot[slot].append(
                CanonicalSegmentStream(
                    slot_id=slot, episode_index=episode_index, episode_position=position, category=category
                )
            )
    streams = tuple(stream for slot in range(b_stream) for stream in by_slot[slot])
    if not streams:
        raise RuntimeError("active Local-Memory found no episode holding a whole TBPTT block")
    return streams


class ActiveLocalMemoryLaunchCallback(Callback):
    """Build the canonical segment owner and attach the window driver at train start.

    ``suite_datasets`` maps each suite name to its dataset; the window size is read
    from ``config.trainer.grad_accum_iter`` at ``on_train_start`` so the driver's
    member count cannot drift from the accumulation boundary the trainer enforces
    on both ends.
    """

    # Resume wiring design v0.4: surface the driver's data-progress state on the
    # DCP ``"dataloader"`` slot.  The interface lives here (not on the driver)
    # because ``_DataloaderWrapper`` walks ``callbacks._callbacks``; the driver is
    # not a group member.
    checkpoint_component: str = "dataloader"

    def __init__(
        self,
        suite_datasets: Mapping[str, Any],
        *,
        b_stream: int = DEFAULT_B_STREAM,
        ttt_tbptt_steps: int = DEFAULT_TBPTT_STEPS,
        manifest_digest: str,
        config_digest: str,
        source_digest: str,
        plan_chain_id: str = "active-local-window",
        member_layout: str = "single",
        group_size: int = 1,
    ) -> None:
        super().__init__()
        if not suite_datasets:
            raise ValueError("active Local-Memory requires at least one suite dataset")
        if b_stream <= 0 or ttt_tbptt_steps <= 0:
            raise ValueError("active Local-Memory requires a positive B_stream and TBPTT width")
        self.suite_datasets = dict(suite_datasets)
        self.b_stream = b_stream
        self.ttt_tbptt_steps = ttt_tbptt_steps
        self.manifest_digest = manifest_digest
        self.config_digest = config_digest
        self.source_digest = source_digest
        if member_layout not in {"single", "a2"} or group_size <= 0:
            raise ValueError("invalid active member layout/group size")
        if (member_layout == "single" and group_size != 1) or (member_layout == "a2" and group_size != b_stream):
            raise ValueError("active group size must match the configured layout/slot count")
        self.member_layout, self.group_size = member_layout, group_size
        self.plan_chain_id = plan_chain_id
        self.driver: ActiveLocalMemoryWindowDriver | None = None
        self._pending_resume_state: dict[str, Any] | None = None

    # ---- checkpoint surface (resume wiring design v0.4) ---------------------

    def has_checkpoint_state(self) -> bool:
        return True

    def state_dict(self) -> dict[str, Any]:
        if self.driver is None:
            raise RuntimeError("active Local-Memory cannot save: driver not attached")
        return self.driver.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        # ``checkpointer.load`` runs before ``on_train_start``, so the driver may
        # not exist yet; buffer the state and apply it once the driver is built.
        if self.driver is None:
            self._pending_resume_state = state_dict
        else:
            self.driver.load_state_dict(state_dict)

    def on_train_start(self, model: Any, iteration: int = 0) -> None:
        """Assemble the owner over the live model and attach the window driver."""
        # The driver's data-progress state is not checkpointed: ``_stream_index`` /
        # ``_active_stream`` / ``_active_cursor`` / ``_window_index`` and the
        # scheduler's ``cumulative_valid_consumer_exposure`` plus its
        # ``stable_slots`` / ``terminal_slots`` / ``admission_order`` /
        # ``committed_identities`` guards are all rebuilt from scratch here.  A
        # Fail closed unless the dataloader slot already delivered a pending
        # resume state: resuming without it would silently replay the episode
        # catalogue from its first episode while model/optim/LR-schedule continue
        # from the checkpoint.
        if iteration > 0 and self._pending_resume_state is None:
            raise RuntimeError(
                "active Local-Memory cannot resume: the window driver's slot frontier and the "
                "segment scheduler's exposure/guard state are not checkpointed, so resuming at "
                f"iteration {iteration} would silently replay the episode catalogue from the "
                "beginning. Restart the run from scratch."
            )
        from cosmos_framework.data.generator.action.datasets.canonical_local_memory_producer import (
            CanonicalLocalMemorySegmentProducer,
        )

        trainer = getattr(self, "trainer", None)
        if trainer is None:
            raise RuntimeError("active Local-Memory launch requires a bound trainer")
        if self.driver is not None:
            raise RuntimeError("active Local-Memory launch ran twice")
        # ``CallBackGroup`` builds callbacks with ``cosmos_framework.utils.lazy_config
        # .instantiate``, whose recursion descends only through nodes carrying their own
        # ``_target_``.  A nested mapping such as ``suite_datasets`` carries none, so it
        # is returned verbatim and its leaves reach us still lazy.  Resolve them here.
        # ``instantiate`` returns an already-materialized dataset unchanged, so this is a
        # no-op on a path that resolved them upstream and never double-builds.
        from cosmos_framework.utils.lazy_config import instantiate

        self.suite_datasets = {
            category: instantiate(dataset) for category, dataset in self.suite_datasets.items()
        }
        window_members = int(trainer.config.trainer.grad_accum_iter)
        adapter = canonical_segment_adapter_from_model(model)
        wiring = CanonicalSegmentWiring(adapter, canonical_slow_parameters_from_model(model))
        categories = tuple(sorted(self.suite_datasets))
        scheduler = RankLocalSegmentScheduler(
            rank=dist.get_rank() if dist.is_initialized() else 0,
            target_distribution={category: 1.0 / len(categories) for category in categories},
        )
        driver_type = ActiveLocalMemoryWindowDriver
        driver_extra = {}
        if self.member_layout == "a2":
            if dist.is_initialized() and dist.get_world_size() != 1:
                raise RuntimeError("A2 delivery currently validates single-rank training only")
            from .grouped_active_driver import GroupedActiveLocalMemoryWindowDriver
            from .grouped_active_runtime import GroupedActiveWiringRegistry, GroupedSegmentRuntimeOwner
            # Abort must clear ALL partial gradients, including the newly trained
            # baseline heads, not only the four Local groups.
            wiring = CanonicalSegmentWiring(adapter, tuple(p for p in model.net.parameters() if p.requires_grad))
            registry = GroupedActiveWiringRegistry(GroupedSegmentRuntimeOwner(scheduler, wiring))
            driver_type = GroupedActiveLocalMemoryWindowDriver
            driver_extra = dict(group_size=self.group_size, manifest_digest=self.manifest_digest,
                                config_digest=self.config_digest)
        else:
            registry = ProductionActiveWiringRegistry(CanonicalSegmentRuntimeOwner(scheduler, wiring))
        producers = {
            category: CanonicalLocalMemorySegmentProducer(
                self.suite_datasets[category],
                category=category,
                ttt_tbptt_steps=self.ttt_tbptt_steps,
                manifest_digest=self.manifest_digest,
                config_digest=self.config_digest,
                source_digest=self.source_digest,
            )
            for category in categories
        }
        driver = driver_type(
            **driver_extra,
            registry=registry,
            producer=SuiteRoutedSegmentProducer(producers, source_digest=self.source_digest),
            streams=canonical_segment_streams(producers, b_stream=self.b_stream),
            window_members=window_members,
            plan_chain_id=self.plan_chain_id,
            catalog_digest=self._catalog_digest(),
            queue_seed=self._queue_seed(),
            prefetch_depth=int(os.environ.get("PSM_ACTIVE_PREFETCH_DEPTH", "4")),
        )
        driver.attach(trainer, model)
        self.driver = driver
        metrics_path = os.environ.get("PSM_ACTIVE_METRICS_PATH")
        if metrics_path:
            from .grouped_active_metrics import ActiveDeliveryMetrics
            trainer.callbacks._callbacks.append(
                ActiveDeliveryMetrics(trainer=trainer, driver=driver, path=metrics_path)
            )
        validation_path = os.environ.get("PSM_A2_POSTTRAIN_VALIDATION_PATH")
        if validation_path:
            if self.member_layout != "a2":
                raise ValueError("native A2 validation requires the A2 route")
            from .a2_native_validation import A2NativeValidation
            trainer.callbacks._callbacks.append(A2NativeValidation(driver=driver, path=validation_path))
        if self._pending_resume_state is not None:
            driver.load_state_dict(self._pending_resume_state)
            self._pending_resume_state = None

    def _catalog_digest(self) -> str:
        """Catalog-order identity derived from the three frozen digests.

        Mirrors the epoch-reuse ``queue_seed`` derivation (candidate (b)): the
        same catalog reproduces the same value across runs/resumes, while a
        changed manifest/config/source yields a different one.
        """
        import hashlib

        preimage = f"{self.manifest_digest}|{self.config_digest}|{self.source_digest}"
        return hashlib.sha256(preimage.encode("utf-8")).hexdigest()

    def _queue_seed(self) -> int:
        """Per-slot reuse permutation seed derived from the catalog identity (§4.6 b).

        ``queue_seed = int(sha256(manifest|config|source).hexdigest()[:16], 16)``: the
        same catalog reproduces the same permutation stream across runs/resumes, and a
        changed catalog no longer inherits the previous ordering.
        """
        return int(self._catalog_digest()[:16], 16)
