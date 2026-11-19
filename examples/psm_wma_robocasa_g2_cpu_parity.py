"""Compare fixed Phase5 G2 CPU consumers with Sync and Spawn before any GPU diagnostic."""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import fields, is_dataclass
from enum import Enum

import numpy as np
import torch

from cosmos_framework.data.generator.action.utils.cached_pixel_geometry import is_pixel_shape_proxy
from cosmos_framework.model.generator.mot.robocasa_exact_window_local import (
    ExactWindowRankPlanner,
    ExactWindowSegmentProducer,
)
from cosmos_framework.trainer.no_ttt_async_loader import NoTTTAsyncLoader
from cosmos_framework.trainer.no_ttt_cpu_loader import _rng_snapshot
from cosmos_framework.utils.no_ttt_loader_profile import LoaderProfile
from examples.psm_wma_robocasa_corrected_phase5 import _g2_recipe, parser, preflight


def _update_digest(digest, value):
    """Hash typed fields and exact tensor bytes without retaining a history of samples."""
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise ValueError("CPU parity payload unexpectedly contains CUDA")
        _update_digest(digest, (str(value.dtype), tuple(value.shape)))
        if is_pixel_shape_proxy(value):
            digest.update(b"cached-pixel-proxy")
        else:
            digest.update(memoryview(value.detach().reshape(-1).contiguous().view(torch.uint8).numpy()))
    elif isinstance(value, np.ndarray):
        _update_digest(digest, torch.from_numpy(value))
    elif isinstance(value, Enum):
        _update_digest(digest, (type(value).__qualname__, value.value))
    elif is_dataclass(value):
        _update_digest(digest, (type(value).__qualname__, [(f.name, getattr(value, f.name)) for f in fields(value)]))
    elif isinstance(value, dict):
        digest.update(b"{")
        for key in sorted(value):
            _update_digest(digest, key)
            _update_digest(digest, value[key])
        digest.update(b"}")
    elif isinstance(value, (list, tuple)):
        digest.update(b"[" if isinstance(value, list) else b"(")
        for item in value:
            _update_digest(digest, item)
        digest.update(b"]" if isinstance(value, list) else b")")
    else:
        data = json.dumps(value, allow_nan=False).encode()
        digest.update(len(data).to_bytes(8, byteorder="big"))
        digest.update(data)


def _restore_rng(state):
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])


def _consume(producer, planner, windows, *, loader=None):
    digest = hashlib.sha256()
    frontier = planner.initial_frontier()
    counts = {"consumers": 0, "dropped": 0, "selected_view_rows": 0}
    for iteration in range(windows):
        plan = planner.plan_window(frontier)
        if loader is not None:
            loader.begin(plan, preview_next=iteration + 1 < windows)
        for requests in plan.members:
            for request in requests:
                segment = producer.produce(request) if loader is None else loader.materialize(request)
                _update_digest(digest, (request, segment))
                for payload in segment.consumer_payload[0]:
                    if payload is not None:
                        counts["consumers"] += 1
                        counts["dropped"] += int(payload["text_dropout_applied"])
                        counts["selected_view_rows"] += len(payload["g2_views"])
            # Deliberately interleave all three parent RNGs between GA members.
            _update_digest(digest, (random.random(), float(np.random.rand()), torch.rand(())))
        frontier = plan.candidate_frontier
        if loader is not None:
            loader.finish()
            loader.checkpoint_boundary()
    _update_digest(digest, (frontier, _rng_snapshot()))
    return {**counts, "ordered_segment_rng_and_frontier_sha256": digest.hexdigest()}


def main(argv=None):
    options = parser()
    options.description = __doc__
    options.add_argument("--cpu-workers", type=int, choices=(1, 2, 4, 8), required=True)
    options.add_argument("--cpu-rank", type=int, choices=range(4), default=0)
    options.add_argument("--cpu-windows", type=int, choices=(1, 2), default=1)
    args = options.parse_args(argv)
    if (
        os.environ.get("CUDA_VISIBLE_DEVICES") != ""
        or args.g2_recipe is None
        or args.phase != "fresh"
        or (args.world_size, args.ga, args.b, args.t) != (4, 4, 8, 16)
        or args.transform_rng_mode != "legacy_global"
        or args.num_workers
        or args.full_segment_prefetch
        or args.snapshot10
    ):
        options.error("requires GPU-hidden, fresh G2 world4/GA4/B8/T16 with legacy-global CPU parity")
    report, _, catalog, _ = preflight(args)
    recipe = _g2_recipe(args)
    producer = ExactWindowSegmentProducer(catalog, config_digest=report["config_digest"])
    planner = ExactWindowRankPlanner(catalog, rank=args.cpu_rank, world_size=4, b_stream=8, active_ga=4)
    profile = LoaderProfile(args.cpu_workers, 1, True, 120.0)
    saved = _rng_snapshot()
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    initial = _rng_snapshot()
    try:
        sync = _consume(producer, planner, args.cpu_windows)
        _restore_rng(initial)
        loader = NoTTTAsyncLoader(
            producer,
            profile,
            planner,
            g2_provider_factory=recipe.provider_factory,
            g2_provider_kwargs=recipe.provider_kwargs,
        )
        try:
            spawn = _consume(producer, planner, args.cpu_windows, loader=loader)
        finally:
            loader.close()
    finally:
        _restore_rng(saved)
    if sync != spawn or (spawn["selected_view_rows"] > 0) is not recipe.global_memory.enabled:
        raise RuntimeError("real Sync/Spawn G2 CPU payload, order, Global mode or RNG differs")
    print(
        json.dumps(
            {
                "scope": "Phase5 CPU consumers; artificial interleaved parent RNG; no model or optimizer",
                "g2_recipe_identity": recipe.identity,
                "rank": args.cpu_rank,
                "world_size": args.world_size,
                "windows": args.cpu_windows,
                "workers": args.cpu_workers,
                "compact_geometry": catalog.raw.compact_cached_video,
                **spawn,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
