"""CPU/static adapter for canonical Local-Memory segments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .local_evidence import ContinualTTTFastState, ContinualTTTLocalMemoryCore, LocalEvidenceEncoder
from .local_memory_segment import LocalMemoryTransaction, SegmentBatch, SegmentIdentity


@dataclass(frozen=True)
class SegmentScanResult:
    local_tokens: torch.Tensor
    local_present: torch.Tensor
    state_out: ContinualTTTFastState
    payloads: tuple[Any, ...]
    locals: tuple[torch.Tensor | None, ...]
    identities: tuple[tuple[int, str, int], ...]


class LocalMemorySegmentSidecar:
    """In-memory detached fast-state cache; scheduler identities remain authoritative."""

    def __init__(self) -> None:
        self._records: dict[int, tuple[SegmentIdentity, ContinualTTTFastState]] = {}

    def read(self, identity: SegmentIdentity) -> ContinualTTTFastState | None:
        record = self._records.get(identity.slot_id)
        if record is None:
            return None
        previous, state = record
        if (identity.episode_id != previous.episode_id or identity.source_digest != previous.source_digest
                or identity.cursor != previous.cursor + 1):
            raise ValueError("segment sidecar identity is not a canonical continuation.")
        return state

    def commit(self, identity: SegmentIdentity, state: ContinualTTTFastState) -> None:
        if identity.training_stream_end:
            self._records.pop(identity.slot_id, None)
            return
        self._records[identity.slot_id] = (identity, ContinualTTTFastState(*(value.detach().clone() for value in state)))

    def reset(self, identity: SegmentIdentity) -> None:
        self._records.pop(identity.slot_id, None)


class CanonicalLocalMemorySegmentAdapter:
    def __init__(self, encoder: LocalEvidenceEncoder, core: ContinualTTTLocalMemoryCore, sidecar: LocalMemorySegmentSidecar) -> None:
        self.encoder, self.core, self.sidecar = encoder, core, sidecar
        self._pending_scan: tuple[SegmentIdentity, LocalMemoryTransaction, SegmentScanResult] | None = None

    @property
    def pending_scan(self) -> tuple[SegmentIdentity, LocalMemoryTransaction, SegmentScanResult] | None:
        """Read-only pending capability for the CPU/static wiring bridge."""
        return self._pending_scan

    def pending(self) -> tuple[SegmentIdentity, LocalMemoryTransaction, SegmentScanResult] | None:
        """Return the exact graph-bearing pending capability without mutation."""
        return self._pending_scan

    def committed_snapshot(self) -> tuple[tuple[SegmentIdentity, ContinualTTTFastState], ...]:
        """Return detached copies of the committed sidecar frontier."""
        return tuple(
            (identity, ContinualTTTFastState(*(value.detach().to(dtype=torch.float32).clone() for value in state)) )
            for _, (identity, state) in sorted(self.sidecar._records.items())
        )

    def discard_pending(self, identity: SegmentIdentity, transaction: LocalMemoryTransaction, result: SegmentScanResult) -> None:
        pending = self._pending_scan
        if pending is None or pending[0] is not identity or pending[1] is not transaction or pending[2] is not result:
            raise RuntimeError("segment pending discard requires the exact pending capability.")
        self._pending_scan = None

    def scan(
        self, segment: SegmentBatch, *, identity: SegmentIdentity, transaction: LocalMemoryTransaction
    ) -> SegmentScanResult:
        if identity not in transaction.scheduler.admission_order or (
            identity.slot_id, identity.episode_id, identity.cursor
        ) not in transaction.plan.members:
            raise ValueError("segment scan requires an admitted identity in its transaction plan.")
        segment.validate(self.core.ttt_tbptt_steps)
        state_in = self.sidecar.read(identity)
        # Evidence is built dataset-side on CPU while the live encoder sits wherever the
        # model was placed, and the core derives its state device from these tensors.
        # Align them here, not inside the encoder: `encode_segment` is also reached by the
        # baseline path, whose evidence already matches the model.  A no-op on CPU, so the
        # offline smoke keeps exercising the same code.
        device = next(self.encoder.parameters()).device
        if state_in is not None:
            state_in = ContinualTTTFastState(*(value.to(device) for value in state_in))
        tokens, state_out, present = self.core.scan_segment_masked_encoded_many(
            self.encoder, segment.evidence_visual_summary_prev.to(device),
            segment.evidence_executed_action_prev.to(device), segment.evidence_valid.to(device),
            state_in, create_graph=True,
        )
        self._psm_diag_probe(segment, state_in, tokens, present)
        payloads, locals_, identities = segment.gather_consumers(tokens, present)
        result = SegmentScanResult(tokens, present, state_out, tuple(payloads), tuple(locals_), tuple(identities))
        self._pending_scan = (identity, transaction, result)
        return result

    def _psm_diag_probe(self, segment, state_in, tokens, present) -> None:
        """TEMP DIAGNOSTIC (enabled by ``PSM_DIAG_EVIDENCE=1``): one-shot numeric probe.

        Reports the first segment's evidence values, validity mask, fast-state graph
        status and parameter magnitudes, and registers a hook that prints the gradient
        the real loss delivers to the Local tokens.  ``None`` for the hook means the
        tokens never received a gradient; a non-zero value means the graph is intact up
        to the tokens, so any all-zero slow gradient is produced inside this scan.
        """
        import os

        if os.environ.get("PSM_DIAG_EVIDENCE") != "1":
            return
        if getattr(self, "_psm_diag_done", False):
            # Keep reporting only the loss-side token gradient: on the first optimizer step
            # `local_memory2llm.weight` is exactly zero, so ``dL/dtoken = grad_out * W`` is
            # identically zero and every runtime parameter is starved.  Watching this value
            # across members shows the step at which W leaves zero and the Local runtime
            # starts receiving gradient.
            if tokens.requires_grad:
                self._psm_diag_hook(tokens)
            return
        self._psm_diag_done = True
        with torch.no_grad():
            visual = segment.evidence_visual_summary_prev
            action = segment.evidence_executed_action_prev
            valid = segment.evidence_valid
            print(
                f"[DIAG-EV] visual max={visual.abs().max().item():.3e} "
                f"nonzero={int((visual != 0).sum())}/{visual.numel()}",
                flush=True,
            )
            print(
                f"[DIAG-EV] action max={action.abs().max().item():.3e} "
                f"nonzero={int((action != 0).sum())}/{action.numel()}",
                flush=True,
            )
            print(f"[DIAG-EV] valid true={int(valid.sum())}/{valid.numel()} present={int(present.sum())}", flush=True)
            state_repr = "None" if state_in is None else [bool(value.requires_grad) for value in state_in]
            print(
                f"[DIAG-EV] tokens requires_grad={tokens.requires_grad} "
                f"max={tokens.abs().max().item():.3e} state_in.requires_grad={state_repr}",
                flush=True,
            )
            for name, parameter in list(self.encoder.named_parameters()) + list(self.core.named_parameters()):
                print(
                    f"[DIAG-EV] param {name}: max={parameter.abs().max().item():.3e} "
                    f"rg={parameter.requires_grad} dtype={parameter.dtype}",
                    flush=True,
                )
        if tokens.requires_grad:
            self._psm_diag_hook(tokens)
        else:
            print("[DIAG-EV] tokens.requires_grad=False: graph already severed at scan exit", flush=True)

    def _psm_diag_hook(self, tokens: torch.Tensor) -> None:
        """Print the gradient the real loss delivers to the Local tokens of one segment."""
        count = getattr(self, "_psm_diag_count", 0) + 1
        self._psm_diag_count = count
        if count > 4 and count % 32:
            tokens.register_hook(lambda grad: None)
            return

        def _report(grad: torch.Tensor) -> None:
            print(
                f"[DIAG-EV] scan#{count} dL/dtoken max={grad.abs().max().item():.3e} "
                f"nonzero={int((grad != 0).sum())}/{grad.numel()}",
                flush=True,
            )
            return None

        tokens.register_hook(_report)

    def commit(self, identity: SegmentIdentity, result: SegmentScanResult, *, transaction: LocalMemoryTransaction) -> None:
        """Persist detached fast state only after the trainer transaction succeeds."""
        pending = self._pending_scan
        if (transaction.terminal_failure_code is not None or transaction.slow_grads_cleared
                or not transaction.completed_members or transaction.completed_members[-1] != identity
                or pending is None or pending[0] != identity or pending[1] is not transaction or pending[2] is not result):
            raise RuntimeError("segment sidecar commit requires successful trainer transaction.")
        self.sidecar.commit(identity, result.state_out)
        self._pending_scan = None
