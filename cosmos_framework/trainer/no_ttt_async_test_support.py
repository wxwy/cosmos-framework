"""Spawn-safe synthetic data and text processor for CPU-only async loader tests.

There is no production CLI route to this recipe. Source/cache data and tokenizer
are doubles; ActionTransformPipeline, SegmentProducer and the async state path
are real. Delays are inside the source/transform calls, not parent-side sleeps.
"""

from __future__ import annotations

import json
import os
import random
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import torch

from cosmos_framework.data.generator.action.datasets.action_sft_dataset import ActionSFTDataset
from cosmos_framework.data.generator.action.utils.cached_pixel_geometry import (
    CachedGeometryResize,
    cached_pixel_placeholder,
)
from cosmos_framework.data.generator.action.utils.transforms import ActionTransformPipeline
from cosmos_framework.model.generator.mot.robocasa_exact_window_local import (
    ExactWindowLocalCatalog,
    ExactWindowSegmentProducer,
)
from cosmos_framework.model.generator.mot.robocasa_exact_window_local_test import _FakeRaw
from cosmos_framework.trainer.no_ttt_async_recipe import AsyncWorkerState

IMPORTED_CUDA_VISIBLE = os.environ.get("CUDA_VISIBLE_DEVICES")
IMPORTED_RANK = os.environ.get("RANK")


class SyntheticTextProcessor:
    def tokenize_text(self, caption, **kwargs):
        assert kwargs == {"is_video": False, "use_system_prompt": False}
        return [11, *caption.encode(), 12]


class SyntheticRaw(_FakeRaw):
    def __init__(self, counts, *, delay=0.0, fault="", fault_flat=-1):
        super().__init__(counts)
        self.delay, self.fault, self.fault_flat = delay, fault, fault_flat
        self.compact_cached_video = True

    def __getitem__(self, index):
        if self.delay:
            time.sleep(self.delay)
        hit = bool(self.fault) and (self.fault_flat < 0 or index >= self.fault_flat)
        if hit:
            if self.fault in {"error", "worker_error"}:
                raise OSError("injected async synthetic Source Read error")
            if self.fault == "crash":
                os._exit(17)
            if self.fault in {"timeout", "ignore_term"}:
                if self.fault == "ignore_term":
                    signal.signal(signal.SIGTERM, signal.SIG_IGN)
                time.sleep(60)
            if self.fault == "cache_change":
                raise ValueError("injected async cache file changed during worker lifetime")
            if self.fault == "rng":
                random.random()
        row = super().__getitem__(index)
        row.update(
            ai_caption=f"Pick up mug in episode {row['episode_index']}",
            video=cached_pixel_placeholder(compact=True),
            conditioning_fps=torch.tensor(24.0),
            mode="wam",
            domain_id=torch.tensor(0),
            viewpoint="concat_view",
            idle_frames=torch.tensor(0),
            global_row_indices=torch.arange(row["start_frame"], row["start_frame"] + 17),
            window_frame_indices=torch.arange(row["start_frame"], row["start_frame"] + 17),
            latent_source_frame_indices=torch.arange(row["start_frame"], row["start_frame"] + 17, 4),
        )
        if hit and self.fault in {"identity", "source"}:
            row["task_class"] = "foreign"
        if hit and self.fault in {"order", "reorder"}:
            row["start_frame"] += 1
        if hit and self.fault == "large_packet":
            row["test_padding"] = torch.zeros(20000)
        return row


class TimedActionPipeline(ActionTransformPipeline):
    def __init__(self, *, delay=0.0, fault="", **kwargs):
        self.delay, self.fault = delay, fault
        super().__init__(**kwargs)

    def __call__(self, item, resolution):
        if self.delay:
            time.sleep(self.delay)
        if self.fault == "transform_error":
            raise ValueError("injected async Transform error")
        if self.fault == "transform_rng":
            random.random()
        return super().__call__(item, resolution)


def build_synthetic_producer(
    count=33,
    t=16,
    world=4,
    rank=0,
    rate=0.1,
    source_delay=0.0,
    transform_delay=0.0,
    fault="",
    fault_flat=-1,
):
    del rank  # Catalog is global; the real planner owns rank assignment.
    raw = SyntheticRaw((count,) * max(256, 8 * world * 4), delay=source_delay, fault=fault, fault_flat=fault_flat)
    kwargs = dict(
        tokenizer_config={"synthetic": True},
        cfg_dropout_rate=rate,
        max_action_dim=64,
        append_viewpoint_info=True,
        append_duration_fps_timestamps=True,
        append_resolution_info=True,
        append_idle_frames=True,
        format_prompt_as_json=True,
    )
    with patch(
        "cosmos_framework.data.generator.augmentors.text_tokenizer.lazy_instantiate",
        return_value=SyntheticTextProcessor(),
    ):
        transform = TimedActionPipeline(delay=transform_delay, fault=fault, **kwargs)
    transform.video_resize = CachedGeometryResize(transform.video_resize, (192, 320, 160, 320))
    wrapped = ActionSFTDataset(raw, transform, resolution=None)
    catalog = ExactWindowLocalCatalog(wrapped, ttt_tbptt_steps=t)
    return ExactWindowSegmentProducer(catalog, config_digest="config")


@dataclass(frozen=True)
class SyntheticAsyncRecipe:
    count: int = 33
    t: int = 16
    world: int = 4
    rate: float = 0.1
    source_delay: float = 0.0
    transform_delay: float = 0.0
    fault: str = ""
    fault_flat: int = -1
    witness: str = ""
    caption_key: str = "ai_caption"
    token_key: str = "text_token_ids"

    def build(self):
        if IMPORTED_CUDA_VISIBLE != "" or IMPORTED_RANK is not None:
            raise AssertionError("async worker import must be CPU-only and rank-free")
        if torch.cuda.is_initialized() or torch.utils.data.get_worker_info() is None:
            raise AssertionError("synthetic async recipe must run in real CPU DataLoader worker")
        if torch.get_num_threads() != 1 or os.environ.get("RANK") is not None:
            raise AssertionError("async worker thread/rank environment not isolated")
        if self.witness:
            path = Path(self.witness)
            path.mkdir(parents=True, exist_ok=True)
            (path / f"worker_{os.getpid()}.json").write_text(
                json.dumps({"pid": os.getpid(), "fault": self.fault, "at": time.monotonic()})
            )
        producer = build_synthetic_producer(
            self.count,
            self.t,
            self.world,
            rate=0,
            source_delay=self.source_delay,
            transform_delay=self.transform_delay,
            fault=self.fault,
            fault_flat=self.fault_flat,
        )
        if self.fault == "late_order":
            state = OrderedCompletionState(producer, rate=self.rate, cache_reader=None)
            state.witness = Path(self.witness)
            return state
        return AsyncWorkerState(producer, rate=self.rate, cache_reader=None)


class OrderedCompletionState(AsyncWorkerState):
    def prepare(self, request):
        marker = self.witness / "nonzero_slot_completed"
        if request.identity.slot_id == 0:
            until = time.monotonic() + 45
            while not marker.exists() and time.monotonic() < until:
                time.sleep(0.01)
            if not marker.exists():
                raise RuntimeError("ordered-completion test barrier expired")
        result = super().prepare(request)
        if request.identity.slot_id != 0:
            marker.touch()
        return result


@dataclass(frozen=True)
class BaselineRawRecipe:
    """Only the old backend's RAW/cache boundary is doubled in its RED witness."""

    count: int = 19
    world: int = 4

    def reader(self):
        return SyntheticRaw((self.count,) * max(256, 8 * self.world * 4))

    def read(self, job, reader):
        return tuple((flat, reader[flat]) for flat, _ in job.sources)
