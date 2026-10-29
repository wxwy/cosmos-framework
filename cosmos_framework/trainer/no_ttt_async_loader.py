"""Bounded two-Window CPU preparation; only ordered CFG selection stays in Trainer.

The DataLoader transports owned bytes rather than mmap tensor views. It knows
nothing about models, optimizer state or the committed frontier. All random
choices remain on the original consuming process and at the original position.
"""

from __future__ import annotations

import io
import os
import pickle
import random
import resource
import time
from collections import deque
from dataclasses import dataclass, fields, replace
from pathlib import Path

import torch
from torch.utils.data import Dataset, Sampler

from cosmos_framework.trainer.no_ttt_cpu_loader import (
    NoTTTCpuLoader,
    OwnedDataLoader,
    OwnedSpawnContext,
    _cpu_tree,
    _rng_snapshot,
    _token,
    _worker_init,
)


@dataclass(frozen=True)
class AsyncJob:
    generation: int
    window: int
    request: object
    file_versions: tuple = ()

    @property
    def token(self):
        return _token(self.request)


class PreviewSampler(Sampler):
    def __init__(self):
        self.jobs = deque()

    def __iter__(self):
        while True:
            if not self.jobs:
                raise RuntimeError("async preview credits exhausted before committed promotion")
            yield self.jobs.popleft()


def _versions(paths):
    result = []
    for name in paths:
        info = Path(name).stat()
        result.append((str(name), info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns))
    return tuple(result)


def _check_versions(expected):
    if _versions([row[0] for row in expected]) != expected:
        raise ValueError("async prepared source/cache file changed before consumption")


def _transport_tree(value):
    # Video is a geometry carrier on the required cached-latent path. Preserve
    # every value/shape/dtype while avoiding exporting a dense zero storage.
    if isinstance(value, dict):
        out = {key: _transport_tree(item) for key, item in value.items() if key != "video"}
        if "video" in value:
            video = value["video"]
            _geometry_bytes(video)
            out["video"] = video.new_zeros(()).expand(video.shape)
        return out
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise ValueError("async packet cannot contain CUDA tensors")
        return value.clone(memory_format=torch.contiguous_format)
    if isinstance(value, (tuple, list)):
        return type(value)(_transport_tree(item) for item in value)
    if hasattr(value, "__dataclass_fields__"):
        return replace(value, **{f.name: _transport_tree(getattr(value, f.name)) for f in fields(value)})
    return value


def _transport_bytes(value):
    """Conservative pre-clone size accounting; geometry carriers use one scalar."""
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size() + 512
    if isinstance(value, dict):
        return (
            sum(
                len(str(key).encode()) + (_geometry_bytes(item) if key == "video" else _transport_bytes(item))
                for key, item in value.items()
            )
            + 128
        )
    if isinstance(value, (tuple, list)):
        return sum(_transport_bytes(item) for item in value) + 128
    if hasattr(value, "__dataclass_fields__"):
        return sum(_transport_bytes(getattr(value, f.name)) for f in fields(value)) + 128
    if isinstance(value, str):
        return len(value.encode()) + 64
    return 128


def _geometry_bytes(video):
    if not isinstance(video, torch.Tensor) or video.device.type != "cpu" or not video.numel():
        raise ValueError("async cached geometry must be nonempty CPU tensor")
    if all(stride == 0 for stride in video.stride()):
        zero = video[(0,) * video.ndim].item() == 0
    else:
        zero = not bool(video.count_nonzero())
    if not zero:
        raise ValueError("async cached video must be a zero CPU geometry carrier")
    return video.element_size() + 512


class PreparedDataset(Dataset):
    def __init__(self, recipe, max_packet_bytes):
        self.recipe, self.max_packet_bytes = recipe, max_packet_bytes
        self.state = None

    def __getitem__(self, job):
        if self.state is None:
            self.state = self.recipe.build()
        state = self.state
        _check_versions(job.file_versions)
        segment, events = state.prepare(job.request)
        _check_versions(job.file_versions)
        if _transport_bytes((segment, state.empty_tokens)) > self.max_packet_bytes:
            raise ValueError("async prepared segment exceeds byte bound before transport cloning")
        package = _transport_tree((segment, state.empty_tokens))
        output = io.BytesIO()
        torch.save(package, output)
        packet = output.getvalue()
        if len(packet) > self.max_packet_bytes:
            raise ValueError("async prepared segment exceeds encoded packet byte bound")
        if torch.cuda.is_initialized():
            raise RuntimeError("async CPU worker initialized CUDA")
        cache = state.cache_reader
        return {
            "generation": job.generation,
            "window": job.window,
            "token": job.token,
            "file_versions": [list(row) for row in job.file_versions],
            "pid": os.getpid(),
            "packet": packet,
            "events": events,
            "rate": state.rate,
            "caption_key": state.caption_key,
            "token_key": state.token_key,
            "worker_maxrss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "cache_mapped_bytes": sum(getattr(cache, "_mapped_sizes", {}).values()),
        }


class NoTTTAsyncLoader(NoTTTCpuLoader):
    """One active + one speculative Window, with parent-only consumption/commit.

    Explicit ``recipe`` is a test injection seam, never a launcher option. The
    production recipe only reconstructs the exact validated local SFT pipeline.
    """

    def __init__(self, producer, profile, planner, *, recipe=None):
        if not profile.workers:
            raise ValueError("workers0 must keep the original synchronous path")
        if planner.b_stream != 8 or planner.active_ga != 4:
            raise ValueError("async preparation requires the matched B8/GA4 profile")
        from cosmos_framework.trainer.no_ttt_async_recipe import AsyncRecipe

        self.producer, self.profile, self.planner = producer, profile, planner
        self.recipe = AsyncRecipe.from_producer(producer, profile) if recipe is None else recipe
        self.file_witnesses = recipe is None  # injected synthetic recipes have no external data files
        self._cache_records = {record.key: record for record in producer.catalog.raw.catalog.episodes}
        if len(pickle.dumps(self.recipe, protocol=5)) > 64 * 1024**2:
            raise ValueError("async CPU metadata recipe exceeds 64MiB")
        self.sampler = PreviewSampler()
        self.generator = torch.Generator(device="cpu").manual_seed(0)
        self.context = OwnedSpawnContext()
        self.loader = OwnedDataLoader(
            PreparedDataset(self.recipe, profile.max_packet_bytes),
            batch_size=None,
            sampler=self.sampler,
            num_workers=profile.workers,
            prefetch_factor=profile.prefetch_factor,
            persistent_workers=profile.persistent_workers,
            pin_memory=False,
            timeout=profile.timeout_s,
            multiprocessing_context=self.context,
            worker_init_fn=_worker_init,
            generator=self.generator,
        )
        self.iterator = self.pending = None
        self.preview = None
        self.index = self.serial = self.generation = 0
        self.closed = False
        self.last_cleanup = []
        self.active_rng = None
        self.binding = self._binding()

    def _binding(self):
        catalog = self.producer.catalog
        raw = catalog.raw
        return (
            catalog.manifest_digest,
            catalog.cache_corpus_digest,
            catalog.source_binding_digest,
            raw.catalog.manifest_sha256,
            raw.catalog.corpus_digest,
            raw.source_reader.source_binding_digest,
            self.producer.config_digest,
        )

    def _admit(self, plan, serial):
        self.sampler.jobs.extend(
            AsyncJob(self.generation, serial, request, self._request_versions(request))
            for member in plan.members
            for request in member
        )

    def _request_versions(self, request):
        if not self.file_witnesses:
            return ()
        raw = self.producer.catalog.raw
        record = self._cache_records[request.episode.key]
        pairs = (
            (raw.catalog.cache_root, record.relative_path),
            (raw.source_reader.source_root, raw.source_reader._bound[record.key].data_file),
        )
        paths = []
        for root, relative in pairs:
            directory, rel = Path(root).resolve(), Path(relative)
            path = directory / rel
            if rel.is_absolute() or ".." in rel.parts or not path.resolve().is_relative_to(directory):
                raise ValueError("async source/cache witness path escaped its root")
            paths.append(str(path))
        return _versions(paths)

    def _begin(self, plan):
        if self.closed or self.pending is not None or self.binding != self._binding():
            raise RuntimeError("async loader closed, pending, or source identity changed")
        self.started = time.monotonic()
        self.active_rng = (
            _rng_snapshot(),
            torch.cuda.get_rng_state() if torch.cuda.is_initialized() else None,
        )
        self.index = 0
        self.record_trace = self.serial < 3 or self.serial % 1000 == 0
        self.metrics = {
            "schema": "nottt_async_loader_timing_v1",
            "trainer_pid": os.getpid(),
            "worker_pids": [],
            "loader_profile": self.profile.async_identity(),
            "execution": "source_cache_transform_worker_cfg_select_parent",
            "timing_scope": "host_monotonic_not_measured_gpu_idle",
            "raw_wait_host_ms": 0.0,
            "collate_host_ms": 0.0,
            "transform_host_ms": 0.0,
            "source_read_ms": 0.0,
            "worker_source_ms_sum": 0.0,
            "worker_transform_ms_sum": 0.0,
            "main_decode_cfg_ms": 0.0,
            "worker_cache_last": {},
            "trace": [],
            "trace_window_sampled": self.record_trace,
        }
        if self.preview is not None:
            if plan != self.preview:
                raise ValueError("speculative plan differs from committed predecessor")
        else:
            self._admit(plan, self.serial)
        self.preview = self.planner.plan_window(plan.candidate_frontier)
        self._admit(self.preview, self.serial + 1)
        self.pending = tuple(request for member in plan.members for request in member)
        if len(self.sampler.jobs) > 2 * self.planner.b_stream * self.planner.active_ga:
            raise RuntimeError("async preview metadata exceeded two-Window bound")
        if self.iterator is None:
            self.iterator = iter(self.loader)

    def _materialize(self, request):
        if (
            self.pending is None
            or self.index >= len(self.pending)
            or request != self.pending[self.index]
            or self.binding != self._binding()
        ):
            raise ValueError("async consume request/source order mismatch")
        started = time.monotonic()
        result = next(self.iterator)
        ready = time.monotonic()
        self.metrics["raw_wait_host_ms"] += (ready - started) * 1000
        if (
            result["generation"] != self.generation
            or result["window"] != self.serial
            or result["token"] != _token(request)
            or type(result["pid"]) is not int
            or result["pid"] == os.getpid()
            or result["pid"] not in {process.pid for process in self.context.processes}
            or not isinstance(result["packet"], bytes)
            or len(result["packet"]) > self.profile.max_packet_bytes
            or (result["rate"], result["caption_key"], result["token_key"])
            != (self.recipe.rate, self.recipe.caption_key, self.recipe.token_key)
        ):
            raise ValueError("async packet generation/window/token/PID/CFG mismatch")
        if result["file_versions"] != [list(row) for row in self._request_versions(request)]:
            raise ValueError("async prepared source/cache file changed before consumption")
        for event in result["events"]:
            key = {"cpu_source": "worker_source_ms_sum", "cpu_transform": "worker_transform_ms_sum"}.get(event["stage"])
            if key is None or event["pid"] != result["pid"] or not event["start"] <= event["end"]:
                raise ValueError("async worker timing provenance malformed")
            self.metrics[key] += (event["end"] - event["start"]) * 1000
        # Only bytes from our owned worker are decoded; this is not an external
        # checkpoint or user-provided pickle. Shared mmap storage never crosses.
        segment, empty_tokens = torch.load(io.BytesIO(result["packet"]), map_location="cpu", weights_only=False)
        _cpu_tree(segment)
        segment.validate(self.planner.catalog.ttt_tbptt_steps)
        if (
            segment.episode_id != (request.episode.uid,)
            or segment.segment_provenance.config_digest != self.producer.config_digest
            or segment.segment_provenance.manifest_digest != self.binding[0]
            or segment.segment_provenance.source_digest != request.identity.source_digest
            or segment.segment_provenance.segment_id != request.identity.segment_id
            or segment.slot_id.tolist() != [request.identity.slot_id]
            or segment.consumer_step[0, : request.valid_count].tolist()
            != list(
                range(
                    request.identity.cursor * self.planner.catalog.ttt_tbptt_steps,
                    request.identity.cursor * self.planner.catalog.ttt_tbptt_steps + request.valid_count,
                )
            )
            or segment.consumer_valid[0].sum().item() != request.valid_count
        ):
            raise ValueError("async prepared Segment identity/consumer mismatch")
        # No speculative RNG state exists. This draw is exactly where the
        # synchronous producer's original tokenizer would have drawn it.
        for payload in segment.consumer_payload[0]:
            if payload is not None and self.recipe.rate > 0 and random.random() < self.recipe.rate:
                payload[self.recipe.caption_key] = ""
                payload[self.recipe.token_key] = empty_tokens.clone()
        self.metrics["main_decode_cfg_ms"] += (time.monotonic() - ready) * 1000
        if result["pid"] not in self.metrics["worker_pids"]:
            self.metrics["worker_pids"].append(result["pid"])
        self.metrics["worker_cache_last"][str(result["pid"])] = {
            key: result[key] for key in ("worker_maxrss_kib", "cache_mapped_bytes")
        }
        if self.record_trace:
            self.metrics["trace"].extend(
                {
                    **event,
                    "window": self.serial,
                    "member": self.index // self.planner.b_stream,
                    "token": result["token"],
                }
                for event in result["events"]
            )
        self._trace(
            {
                "stage": "main_thread_wait",
                "start": started,
                "end": ready,
                "pid": os.getpid(),
                "window": self.serial,
                "token": result["token"],
            }
        )
        self._trace(
            {"stage": "consumer_ready", "at": time.monotonic(), "window": self.serial, "token": result["token"]}
        )
        self.index += 1
        return segment

    def _trace(self, event):
        if self.record_trace:
            self.metrics["trace"].append(event)

    def finish(self):
        if self.pending is None or self.index != len(self.pending):
            raise RuntimeError("async loader cannot finish incomplete active Window")
        # Called only after the real optimizer and frontier have committed.
        # Later cleanup/telemetry failures cannot rewind a committed RNG stream.
        self.active_rng = None
        self.pending = None
        self.serial += 1
        self._trace({"stage": "committed_iter", "at": time.monotonic(), "window": self.serial - 1})
        self.metrics["step_window_host_wall_ms_before_telemetry"] = (time.monotonic() - self.started) * 1000
        if not self.profile.persistent_workers:
            # Explicit nonpersistent mode intentionally discards the neutral
            # preview and reopens workers next Window. It does not publish data.
            self._shutdown()
            self.sampler.jobs.clear()
            self.preview = None
        return dict(self.metrics)

    def checkpoint_boundary(self):
        if self.closed or self.pending is not None:
            raise RuntimeError("async checkpoint requires no active uncommitted Window")
        # Speculative CPU work is RNG-neutral and has no state in the frontier.
        # It can finish in its bounded queues while DCP reads committed state.
        return None

    def close(self):
        self.closed = True
        self.generation += 1
        try:
            self._shutdown()
        finally:
            if self.active_rng is not None:
                import numpy as np

                cpu, cuda = self.active_rng
                random.setstate(cpu[0])
                np.random.set_state(cpu[1])
                torch.set_rng_state(cpu[2])
                if cuda is not None:
                    torch.cuda.set_rng_state(cuda)
                self.active_rng = None
            self.pending = self.preview = None
            self.sampler.jobs.clear()
