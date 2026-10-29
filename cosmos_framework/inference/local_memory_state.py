"""One immutable publication for inference fast state, visual cache and response.

Records are prepared off to the side. The adapter owns publication under the
shared lock; readers retain a coherent snapshot, never three mutable dictionaries.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True)
class CachedMemoryResponse:
    request_key: str
    payload_sha256: str
    body: bytes


@dataclass(frozen=True)
class MemorySnapshot:
    fast: Mapping[str, Any]
    visual: Mapping[str, Any]
    responses: Mapping[str, CachedMemoryResponse]

    @classmethod
    def build(cls, fast=(), visual=(), responses=()) -> MemorySnapshot:
        return cls(*(MappingProxyType(dict(value)) for value in (fast, visual, responses)))

    def without(self, session_id: str) -> MemorySnapshot:
        return self.build(
            *(
                {key: value for key, value in values.items() if key != session_id}
                for values in (self.fast, self.visual, self.responses)
            )
        )


class MemoryStateStore:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.snapshot = MemorySnapshot.build()
