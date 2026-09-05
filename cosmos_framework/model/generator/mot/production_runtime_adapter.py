"""Production Local Memory runtime adapter over the shared authority."""
from __future__ import annotations

import torch

from .local_evidence import ContinualTTTLocalMemoryCore, LocalEvidenceEncoder
from .runtime_authority import AdmissionAuthority, ProductionRuntimeAuthority, ReplayRecord


class ProductionLocalMemoryRuntime:
    """CPU-reference production seam; no model/config/checkpoint side effects."""

    def __init__(self, authority: AdmissionAuthority, encoder: LocalEvidenceEncoder, core: ContinualTTTLocalMemoryCore) -> None:
        self.authority = authority
        self._runtime = ProductionRuntimeAuthority(authority, encoder, core)

    @property
    def runtime_authority(self) -> ProductionRuntimeAuthority:
        """The shared production authority; references the registered encoder/core objects."""
        return self._runtime

    @property
    def write_count(self) -> int:
        return self._runtime.c5_write_count

    def begin_segment(self, owner_key: str) -> None:
        self._runtime.begin(owner_key)

    def admit_evidence(self, *, owner_key: str, source_identity: str, source_timestep: int, source: dict[str, torch.Tensor], epoch: int = 0) -> tuple[int, int] | ReplayRecord:
        capability = self.authority.issue(owner_key=owner_key, source_identity=source_identity, source_timestep=source_timestep, source=source, epoch=epoch)
        return self._runtime.admit(capability, source=source)

    def materialize(self, owner_key: str, *, emit_prewrite_tokens: bool = False) -> torch.Tensor:
        return self._runtime.materialize(owner_key, emit_prewrite_tokens=emit_prewrite_tokens)

    def backward(self, owner_key: str, loss: torch.Tensor) -> None:
        self._runtime.backward_and_mark(owner_key, loss)

    def mark_external_backward(self, owner_key: str, *, loss: torch.Tensor) -> None:
        self._runtime.mark_external_backward(owner_key, loss=loss)

    def finish(self, owner_key: str, *, terminal: bool) -> None:
        self._runtime.finish(owner_key, terminal=terminal)

    def abort(self, owner_key: str) -> None:
        self._runtime.abort(owner_key)

    def reset(self, owner_key: str) -> None:
        self._runtime.reset(owner_key)

    @staticmethod
    def disabled_path(sample: torch.Tensor) -> tuple[tuple[torch.Tensor, None], torch.Tensor]:
        packed = (sample.clone(), None)
        return packed, packed[0].square().mean()
