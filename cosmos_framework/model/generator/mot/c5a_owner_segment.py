"""Test-only facade over the shared production Local Memory authority."""
from __future__ import annotations

from .runtime_authority import (
    AdmissionAuthority,
    AdmissionCapability,
    ProductionRuntimeAuthority,
    ReplayRecord,
)


class C5AOwnerSegmentCPU(ProductionRuntimeAuthority):
    """Synthetic CPU facade; authority/state semantics live in runtime_authority."""
