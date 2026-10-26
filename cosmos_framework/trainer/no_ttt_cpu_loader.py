# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Opt-in CPU DataLoader for frozen No-TTT raw segment requests.

Source/Arrow reads and every stochastic transform stay in the parent. Workers
own only metadata, read-only mmap cache readers and small CPU source snapshots.
Neither producer, planner, tokenizer, model nor frontier is sent to a worker.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import pickle
import random
import resource
import threading
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from multiprocessing.context import SpawnContext, SpawnProcess
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from cosmos_framework.utils.no_ttt_loader_profile import LoaderProfile


def _rng_snapshot():
    import numpy as np

    return random.getstate(), np.random.get_state(), torch.get_rng_state()


def _require_same_rng(before):
    import numpy as np

    after = _rng_snapshot()
    if (
        before[0] != after[0]
        or before[1][0] != after[1][0]
        or not np.array_equal(before[1][1], after[1][1])
        or before[1][2:] != after[1][2:]
        or not torch.equal(before[2], after[2])
    ):
        raise ValueError("CPU RAW source/cache work must not consume transform/model RNG")


def _worker_init(_worker_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        os.environ.pop(key, None)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
    torch.set_num_threads(1)
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU worker inherited CUDA state")


class MappedEpisodeReader:
    """Use the original per-window checks; mmap payload with bounded metadata LRU.

    Episode header checks below match RoboCasaExactWindowEpisodeReader._episode.
    All read_window/read_identity validation remains in the original reader.
    """

    def __new__(cls, catalog, max_episodes, max_bytes):
        from cosmos_framework.data.generator.action.datasets.robocasa_exact_window_cache import (
            RoboCasaExactWindowEpisodeReader,
        )

        class Reader(RoboCasaExactWindowEpisodeReader):
            def __init__(self):
                super().__init__(catalog, max_cached_episodes=max_episodes)
                self._mapped_sizes = OrderedDict()
                self._versions = {}

            def _episode(self, record):
                path = catalog.cache_root / record.relative_path
                if not path.resolve().is_relative_to(catalog.cache_root.resolve()):
                    raise ValueError("worker cache path escapes catalog")
                stat = path.stat()
                version = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
                if self._versions.setdefault(record.key, version) != version:
                    raise ValueError("mmap Episode changed during reader lifetime")
                if record.key in self._loaded:
                    self._mapped_sizes.move_to_end(record.key)
                    return super()._episode(record)
                path = catalog.cache_root / record.relative_path
                if not path.resolve().is_relative_to(catalog.cache_root.resolve()):
                    raise ValueError("worker cache path escapes catalog")
                size = path.stat().st_size
                if size <= 0 or size > max_bytes or record.window_count > 65536:
                    raise ValueError("worker Episode exceeds mmap metadata/file bound")
                while self._mapped_sizes and (
                    len(self._mapped_sizes) >= max_episodes or sum(self._mapped_sizes.values()) + size > max_bytes
                ):
                    key, _ = self._mapped_sizes.popitem(last=False)
                    self._loaded.pop(key)
                payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
                if (
                    not isinstance(payload, dict)
                    or payload.get("format") != "exact_window_v1"
                    or payload.get("source_format") != "lerobot_v3"
                ):
                    raise ValueError("worker Episode format mismatch")
                for key, expected in (
                    ("suite", catalog.suite),
                    ("task_class", record.key.task_class),
                    ("episode_index", record.key.episode_index),
                ):
                    if key in payload and payload[key] != expected:
                        raise ValueError("worker Episode identity mismatch")
                metadata = payload.get("metadata")
                if not isinstance(metadata, dict) or metadata.get("camera_set") != "left_wrist":
                    raise ValueError("worker Episode camera metadata mismatch")
                for key, expected in (
                    ("task_class", record.key.task_class),
                    ("task_slug", record.key.task_slug),
                    ("episode_index", record.key.episode_index),
                ):
                    if key in metadata and metadata[key] != expected:
                        raise ValueError("worker Episode metadata identity mismatch")
                windows = payload.get("windows")
                if not isinstance(windows, dict) or set(windows) != {str(step) for step in record.window_starts}:
                    raise ValueError("worker Episode windows mismatch")
                self._loaded[record.key] = payload
                self._mapped_sizes[record.key] = size
                after = path.stat()
                if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns) != version:
                    self._loaded.pop(record.key)
                    self._mapped_sizes.pop(record.key)
                    raise ValueError("mmap Episode changed during load")
                return payload

        return Reader()


@dataclass(frozen=True)
class RawRecipe:
    catalog: Any
    contract: Any
    source_digest: str
    max_episodes: int
    max_cache_bytes: int

    @classmethod
    def from_producer(cls, producer, profile):
        raw = producer.catalog.raw
        # SourceReader.read_at() consults cache identities. Replace BOTH parent
        # caches too, otherwise that identity read would eagerly deserialize all
        # Episode latent storage before workers ever start. No file is changed.
        parent_reader = MappedEpisodeReader(raw.catalog, 8, profile.max_cache_bytes)
        raw.cache_reader = raw.source_reader.cache_reader = parent_reader
        return cls(
            raw.catalog,
            raw.contract,
            producer.catalog.source_binding_digest,
            max(1, 8 // profile.workers),
            profile.max_cache_bytes,
        )

    def read(self, job, reader):
        from types import SimpleNamespace

        from cosmos_framework.data.generator.action.datasets.robocasa_exact_window_cached_sft import (
            RoboCasaExactWindowCachedDataset,
        )

        snapshots = job.sources
        items = dict(snapshots)
        index = {flat: (source.key, source.start_frame) for flat, source in snapshots}
        source = SimpleNamespace(
            catalog=self.catalog, index=SimpleNamespace(catalog=self.catalog), source_binding_digest=self.source_digest
        )
        source.index = _SnapshotIndex(self.catalog, index)
        source.read_at = items.__getitem__
        raw = RoboCasaExactWindowCachedDataset(self.catalog, source, self.contract, cache_reader=reader)
        raw.compact_cached_video = True  # Transport zero pixels as shape metadata, reconstruct in parent.
        before = _rng_snapshot()
        rows = tuple((flat, raw[flat]) for flat, _ in snapshots)
        _require_same_rng(before)
        return rows

    def reader(self):
        return MappedEpisodeReader(self.catalog, self.max_episodes, self.max_cache_bytes)


class _SnapshotIndex:
    def __init__(self, catalog, rows):
        self.catalog, self.rows = catalog, rows

    def __getitem__(self, index):
        return self.rows[index]


@dataclass(frozen=True)
class RawJob:
    token: str
    flat_indices: tuple[int, ...]
    source_bytes: bytes

    @property
    def sources(self):
        # Trusted parent-created CPU snapshots, never an external pickle file.
        return pickle.loads(self.source_bytes)


def _token(request):
    return hashlib.sha256(
        json.dumps(
            {"identity": asdict(request.identity), "epoch": request.binding_epoch, "count": request.valid_count},
            sort_keys=True,
        ).encode()
    ).hexdigest()


def _cpu_tree(value, *, clone=False):
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise ValueError("CPU loader cannot transport non-CPU tensors")
        return value.clone(memory_format=torch.contiguous_format) if clone else value
    if isinstance(value, dict):
        return {key: _cpu_tree(item, clone=clone) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu_tree(item, clone=clone) for item in value)
    if is_dataclass(value):
        return replace(
            value, **{field.name: _cpu_tree(getattr(value, field.name), clone=clone) for field in fields(value)}
        )
    return value


class RawDataset(Dataset):
    def __init__(self, recipe, max_packet_bytes):
        self.recipe, self.max_packet_bytes = recipe, max_packet_bytes
        self._reader = None

    def __getitem__(self, job):
        started = time.perf_counter()
        if self._reader is None:
            self._reader = self.recipe.reader()
        rows = []
        estimate = 0
        for flat, item in self.recipe.read(job, self._reader):
            item = dict(item)
            video = item.pop("video", None)
            if video is not None:
                if (
                    video.device.type != "cpu"
                    or tuple(video.shape) != (3, 17, 256, 512)
                    or video.dtype != torch.uint8
                    or bool(torch.count_nonzero(video))
                ):
                    raise ValueError("worker shape-only cached video mismatch")
                item["_mp_zero_video"] = True
            estimate += sum(
                value.numel() * value.element_size()
                if isinstance(value, torch.Tensor)
                else 4 * len(value)
                if isinstance(value, str)
                else 64
                for value in item.values()
            )
            if estimate > self.max_packet_bytes:
                raise ValueError("worker raw packet exceeds bound before cloning/encoding")
            rows.append((flat, _cpu_tree(item, clone=True)))
        # A bytes envelope avoids tensor-view storage IPC accidentally exporting
        # an entire mmap Episode into /dev/shm; only cloned exact windows travel.
        estimate = sum(
            value.numel() * value.element_size()
            for _, row in rows
            for value in row.values()
            if isinstance(value, torch.Tensor)
        )
        if estimate > self.max_packet_bytes:
            raise ValueError("worker raw packet tensor bytes exceed bound")
        output = io.BytesIO()
        torch.save(rows, output)
        packet = output.getvalue()
        if len(packet) > self.max_packet_bytes:
            raise ValueError("worker encoded packet exceeds bound")
        if torch.cuda.is_initialized():
            raise RuntimeError("CPU worker touched CUDA")
        return {
            "token": job.token,
            "packet": packet,
            "pid": os.getpid(),
            "prepare_ms": (time.perf_counter() - started) * 1000,
            "worker_maxrss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "cache_mapped_bytes": sum(getattr(self._reader, "_mapped_sizes", {}).values()),
            "cache_mapped_episodes": len(getattr(self._reader, "_mapped_sizes", {})),
        }


class WindowSampler(Sampler):
    def __init__(self):
        self.jobs = ()

    def __iter__(self):
        return iter(self.jobs)

    def __len__(self):
        return len(self.jobs)


class CpuSpawnProcess(SpawnProcess):
    def start(self):
        # Apply before interpreter startup/unpickling, not merely worker_init_fn.
        # CUDA and Arrow-related modules can be imported while decoding a recipe.
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("CPU loader startup must run on the training main thread")
        values = {
            "CUDA_VISIBLE_DEVICES": "",
            "COSMOS_DEVICE": "cpu",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
        removed = ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")
        previous = {key: os.environ.get(key) for key in (*values, *removed)}
        try:
            os.environ.update(values)
            for key in removed:
                os.environ.pop(key, None)
            super().start()
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


class OwnedSpawnContext(SpawnContext):
    """Retain handles even when DataLoader construction fails midway through spawn."""

    def __init__(self):
        self.processes = []

    def Process(self, *args, **kwargs):
        process = CpuSpawnProcess(*args, **kwargs)
        self.processes.append(process)
        return process


class OwnedDataLoader(DataLoader):
    """Retain the official iterator during its fallible worker startup."""

    def _get_iterator(self):
        from torch.utils.data.dataloader import _MultiProcessingDataLoaderIter

        self.check_worker_number_rationality()
        iterator = _MultiProcessingDataLoaderIter.__new__(_MultiProcessingDataLoaderIter)
        self._owned_iterator = iterator
        _MultiProcessingDataLoaderIter.__init__(iterator, self)
        return iterator


class NoTTTCpuLoader:
    def __init__(self, producer, profile: LoaderProfile, *, recipe=None):
        if profile.workers == 0:
            raise ValueError("zero workers must use the original produce path")
        self.producer, self.profile = producer, profile
        recipe = recipe or RawRecipe.from_producer(producer, profile)
        if len(pickle.dumps(recipe, protocol=5)) > 64 * 1024**2:
            raise ValueError("worker metadata recipe exceeds 64MiB; cannot copy a live reader/model")
        self.sampler = WindowSampler()
        self.generator = torch.Generator(device="cpu").manual_seed(0)
        self.context = OwnedSpawnContext()
        self.loader = OwnedDataLoader(
            RawDataset(recipe, profile.max_packet_bytes),
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
        self.iterator = None
        self.pending = None
        self.index = 0
        self.closed = False
        self.last_cleanup = []

    def begin(self, plan):
        try:
            return self._begin(plan)
        except BaseException:
            self.close()
            raise

    def _begin(self, plan):
        if self.closed or self.pending is not None:
            raise RuntimeError("CPU loader closed or previous window pending")
        self.started = time.perf_counter()
        self.metrics = {
            "schema": "nottt_cpu_loader_timing_v1",
            "trainer_pid": os.getpid(),
            "worker_pids": [],
            "loader_profile": self.profile.identity(),
            "source_read_ms": 0.0,
            "worker_prepare_ms_sum": 0.0,
            "raw_wait_host_ms": 0.0,
            "decode_host_ms": 0.0,
            "materialize_host_ms_including_transform": 0.0,
            "transform_host_ms": 0.0,
            "collate_host_ms": 0.0,
            "timing_scope": "host_wall_not_gpu_idle",
            "worker_cache_last": {},
        }
        jobs = []
        started = time.perf_counter()
        before = _rng_snapshot()
        for member in plan.members:
            for request in member:
                prepared = self.producer.prepare(request, raw_getter=self.producer.catalog.raw.source_reader.read_at)
                snapshots = tuple((request.episode.flat_start + step, source) for step, source in prepared.items)
                encoded = pickle.dumps(_cpu_tree(snapshots, clone=True), protocol=5)
                if len(encoded) > min(self.profile.max_packet_bytes, 1024**2):
                    raise ValueError("parent CPU source snapshot exceeds packet bound")
                jobs.append(RawJob(_token(request), tuple(flat for flat, _ in snapshots), encoded))
        _require_same_rng(before)
        self.metrics["source_read_ms"] = (time.perf_counter() - started) * 1000
        self.sampler.jobs = tuple(jobs)
        self.pending = tuple(jobs)
        self.index = 0
        self.iterator = iter(self.loader)

    def materialize(self, request):
        try:
            return self._materialize(request)
        except BaseException:
            self.close()
            raise

    def _materialize(self, request):
        if self.pending is None or self.index >= len(self.pending) or _token(request) != self.pending[self.index].token:
            raise ValueError("CPU loader request order/token mismatch")
        started = time.perf_counter()
        result = next(self.iterator)
        self.metrics["raw_wait_host_ms"] += (time.perf_counter() - started) * 1000
        if (
            result["token"] != _token(request)
            or type(result["pid"]) is not int
            or result["pid"] == os.getpid()
            or not isinstance(result["packet"], bytes)
            or len(result["packet"]) > self.profile.max_packet_bytes
        ):
            raise ValueError("CPU loader result identity/PID/packet mismatch")
        if result["pid"] not in self.metrics["worker_pids"]:
            self.metrics["worker_pids"].append(result["pid"])
        self.metrics["worker_cache_last"][str(result["pid"])] = {
            key: result[key] for key in ("worker_maxrss_kib", "cache_mapped_bytes", "cache_mapped_episodes")
        }
        self.metrics["worker_prepare_ms_sum"] += result["prepare_ms"]
        started = time.perf_counter()
        values = torch.load(io.BytesIO(result["packet"]), map_location="cpu", weights_only=True)
        expected = self.pending[self.index].flat_indices
        if not isinstance(values, list) or tuple(flat for flat, _ in values) != expected:
            raise ValueError("CPU loader returned incorrect window sequence")
        rows = dict(values)
        from cosmos_framework.data.generator.action.utils.cached_pixel_geometry import cached_pixel_placeholder

        for row in rows.values():
            _cpu_tree(row)
            if row.pop("_mp_zero_video", False):
                row["video"] = cached_pixel_placeholder(compact=self.producer.catalog.raw.compact_cached_video)
        self.metrics["decode_host_ms"] += (time.perf_counter() - started) * 1000
        prepared = self.producer.prepare(request, raw_getter=rows.__getitem__)
        wrapped = self.producer.catalog.wrapped_sft
        transform = wrapped._transform

        def timed(*args, **kwargs):
            start = time.perf_counter()
            try:
                return transform(*args, **kwargs)
            finally:
                self.metrics["transform_host_ms"] += (time.perf_counter() - start) * 1000

        started = time.perf_counter()
        wrapped._transform = timed
        try:
            segment = self.producer.materialize(prepared)
        finally:
            wrapped._transform = transform
        self.metrics["materialize_host_ms_including_transform"] += (time.perf_counter() - started) * 1000
        self.index += 1
        return segment

    def finish(self):
        if self.pending is None or self.index != len(self.pending):
            raise RuntimeError("CPU loader cannot finish an incomplete window")
        self.pending, self.sampler.jobs = None, ()
        if not self.profile.persistent_workers:
            self._shutdown()
        self.metrics["step_window_host_wall_ms_before_telemetry"] = (time.perf_counter() - self.started) * 1000
        return dict(self.metrics)

    def _shutdown(self):
        iterator = self.iterator
        if iterator is None:
            iterator = getattr(self.loader, "_owned_iterator", None)
        if iterator is None and not self.context.processes:
            return
        processes = tuple(process for process in self.context.processes if process.pid is not None)
        if iterator is not None and getattr(iterator, "_worker_pids_set", False):
            from torch.utils.data._utils import signal_handling

            signal_handling._remove_worker_pids(id(iterator))
            iterator._worker_pids_set = False
        done = getattr(iterator, "_workers_done_event", None)
        if done is not None:
            done.set()
        for process in processes:
            if process.is_alive():
                process.terminate()
        deadline = time.monotonic() + 2
        for process in processes:
            process.join(max(0, deadline - time.monotonic()))
        for process in processes:
            if process.is_alive():
                process.kill()
        deadline = time.monotonic() + 3
        for process in processes:
            process.join(max(0, deadline - time.monotonic()))
        remaining = [process.pid for process in processes if process.is_alive()]
        self.last_cleanup = [{"pid": process.pid, "exitcode": process.exitcode} for process in processes]
        # Isolated PyTorch-private shutdown seam: queues carry bytes, no CUDA or
        # shared tensor storage. Prevent destructor from joining a dead feeder.
        if iterator is not None:
            iterator._shutdown = True
        for queue in (*getattr(iterator, "_index_queues", ()), getattr(iterator, "_worker_result_queue", None)):
            if queue is not None:
                queue.cancel_join_thread()
                queue.close()
        self.loader._iterator = self.iterator = None
        self.loader._owned_iterator = None
        self.context.processes = [process for process in processes if process.is_alive()]
        if remaining:
            raise RuntimeError(f"owned DataLoader workers survived bounded cleanup: {remaining}")

    def close(self):
        self.closed = True
        try:
            self._shutdown()
        finally:
            self.pending, self.sampler.jobs = None, ()
