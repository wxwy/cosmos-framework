"""Bounded per-work-item timing; CUDA durations use completed device events."""

from __future__ import annotations

import math
import time
from contextlib import contextmanager

import torch


class PipelineTiming:
    def __init__(self, device: torch.device | str) -> None:
        self.device = torch.device(device)
        self._records: list[tuple] = []
        self._active = False
        self._finished = False

    @contextmanager
    def measure(self, name: str):
        if name not in ("independent_encode1", "visual96", "fast_update", "native_generation"):
            raise ValueError("unknown Local pipeline timing stage")
        if self._active or self._finished or len(self._records) >= 4:
            raise RuntimeError("one bounded non-nested timing scope per pipeline work item")
        self._active = True
        if self.device.type == "cuda":
            with torch.cuda.device(self.device):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
        else:
            start, end = time.perf_counter(), None
        try:
            yield
        finally:
            if self.device.type == "cuda":
                with torch.cuda.device(self.device):
                    end.record()
            else:
                end = time.perf_counter()
            self._records.append((name, start, end))
            self._active = False

    def finish(self) -> dict:
        if self._active or self._finished:
            raise RuntimeError("timings must be finalized once after all stages")
        self._finished = True
        if self.device.type == "cuda" and self._records:
            # Events are recorded on the sole model owner's current stream.
            # Reading elapsed_time before the final event completes is invalid.
            self._records[-1][2].synchronize()
        stages = {}
        for name, start, end in self._records:
            elapsed = start.elapsed_time(end) if self.device.type == "cuda" else (end - start) * 1000
            if not math.isfinite(elapsed) or elapsed < 0:
                raise RuntimeError("invalid Local pipeline device timing")
            previous = stages.setdefault(name, {"count": 0, "total_ms": 0.0})
            previous["count"] += 1
            previous["total_ms"] += elapsed
        return {"timing_source": "cuda_events" if self.device.type == "cuda" else "cpu_perf_counter", "stages": stages}
