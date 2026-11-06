"""Bounded duration summaries; callers provide their own metadata lock."""

from __future__ import annotations

import math
from collections import deque


class DurationWindow:
    def __init__(self, capacity: int = 4096):
        if type(capacity) is not int or not 1 <= capacity <= 4096:
            raise ValueError("bounded profiling window required")
        self.values = deque(maxlen=capacity)
        self.count = 0
        self.total_ms = 0.0

    def add(self, elapsed_ms: float):
        if type(elapsed_ms) not in (int, float) or not math.isfinite(elapsed_ms) or elapsed_ms < 0:
            raise ValueError("duration must be finite nonnegative milliseconds")
        self.count += 1
        self.total_ms += elapsed_ms
        self.values.append(elapsed_ms)

    def summary(self):
        values = sorted(self.values)
        return {
            "count": self.count,
            "total_ms": self.total_ms,
            "recent_count": len(values),
            "p50_ms": values[max(0, math.ceil(len(values) * 0.5) - 1)] if values else None,
            "p95_ms": values[max(0, math.ceil(len(values) * 0.95) - 1)] if values else None,
            "percentile_scope": "recent_bounded_nearest_rank",
        }
