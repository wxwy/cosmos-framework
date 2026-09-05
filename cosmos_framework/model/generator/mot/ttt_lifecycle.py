"""R09-B TTT v0.3.2 active-wiring lifecycle (training side).

Owns the per-owner segment state machine around the production runtime
authority: the forward seam (admit graph-free raw evidence, read-after-(t-1)
prefix reads, exactly one detached write per window, closing-window pre-write
witness materialize + arm), the external-backward seam (mark -> commit ->
terminal reset), and the optimizer-transaction resolution seam
(SUCCESS / SCALER_SKIP / EXCEPTION). 1 micro-batch = 1 native window = 1
evidence row; the trainer performs exactly one ``loss.backward()`` per
micro-batch and commits strictly before the next micro-batch forward.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from cosmos_framework.utils.callback import Callback

from .local_evidence import ContinualTTTFastState, ContinualTTTLocalMemoryCore, LocalEvidenceEncoder
from .production_runtime_adapter import ProductionLocalMemoryRuntime
from .runtime_authority import AdmissionAuthority, ReplayRecord

RESOLUTION_SUCCESS = "SUCCESS"
RESOLUTION_SCALER_SKIP = "SCALER_SKIP"
RESOLUTION_EXCEPTION = "EXCEPTION"
_RESOLUTIONS = (RESOLUTION_SUCCESS, RESOLUTION_SCALER_SKIP, RESOLUTION_EXCEPTION)


def scaler_found_inf(grad_scaler: torch.amp.GradScaler, optimizer: torch.optim.Optimizer) -> bool:
    """Read GradScaler's internal per-optimizer found-inf state recorded by ``step()``.

    Fail-closed when the state is unavailable; only consulted on the
    local_ttt_enabled path.
    """
    states = getattr(grad_scaler, "_per_optimizer_states", None)
    optimizer_state = states.get(id(optimizer)) if states is not None else None
    if optimizer_state is None:
        raise RuntimeError("GradScaler has no state for this optimizer; step() must run first.")
    found_inf_per_device = optimizer_state.get("found_inf_per_device")
    if not found_inf_per_device:
        raise RuntimeError("GradScaler recorded no found-inf state for this optimizer.")
    return any(bool(value.item()) for value in found_inf_per_device.values())


@dataclass
class _OwnerState:
    """Persistent per-owner chronology mirror (survives individual segments)."""

    epoch: int = 0
    committed_state: ContinualTTTFastState | None = None
    committed_timestep: int = -1


@dataclass
class _OpenSegment:
    """One live transaction; the authority owns the raw rows and replay state."""

    epoch: int
    base_state: ContinualTTTFastState | None
    row_count: int = 0
    candidate: ContinualTTTFastState | None = None
    armed: bool = False
    terminal: bool = False


class TTTLifecycle:
    """Per-owner TTT segment lifecycle; holds fast state and is never an nn.Module."""

    def __init__(
        self,
        runtime: ProductionLocalMemoryRuntime,
        *,
        encoder: LocalEvidenceEncoder,
        core: ContinualTTTLocalMemoryCore,
        local_memory2llm: torch.nn.Module,
        modality_embed: torch.nn.Parameter,
    ) -> None:
        if not isinstance(core, ContinualTTTLocalMemoryCore) or not isinstance(encoder, LocalEvidenceEncoder):
            raise ValueError(
                "TTT lifecycle requires the registered LocalEvidenceEncoder/ContinualTTTLocalMemoryCore objects."
            )
        if runtime.runtime_authority.encoder is not encoder or runtime.runtime_authority.core is not core:
            raise ValueError("authority must reference the same registered Local module objects; copies are forbidden.")
        if not isinstance(modality_embed, torch.nn.Parameter):
            raise ValueError("local_memory_modality_embed must be an nn.Parameter.")
        self._runtime = runtime
        self._encoder = encoder
        self._core = core
        self._slow_parameters: tuple[torch.nn.Parameter, ...] = tuple(
            list(encoder.parameters())
            + list(core.parameters())
            + list(local_memory2llm.parameters())
            + [modality_embed]
        )
        self._owners: dict[str, _OwnerState] = {}
        self._segments: dict[str, _OpenSegment] = {}
        self._loss: torch.Tensor | None = None

    @classmethod
    def from_registered_modules(cls, net: torch.nn.Module) -> TTTLifecycle:
        """Build over the registered Local modules of ``net``; no second trainable copy."""
        runtime = getattr(net, "local_history_runtime", None)
        if runtime is None:
            raise ValueError("local_ttt_enabled requires the registered local_history_runtime owner.")
        encoder = getattr(runtime, "encoder", None)
        backend = getattr(runtime, "recurrent_backend", None)
        local_memory2llm = getattr(net, "local_memory2llm", None)
        modality_embed = getattr(net, "local_memory_modality_embed", None)
        if local_memory2llm is None or modality_embed is None:
            raise ValueError("local_ttt_enabled requires local_memory2llm and local_memory_modality_embed.")
        return cls(
            ProductionLocalMemoryRuntime(AdmissionAuthority(), encoder, backend),
            encoder=encoder,
            core=backend,
            local_memory2llm=local_memory2llm,
            modality_embed=modality_embed,
        )

    @property
    def runtime(self) -> ProductionLocalMemoryRuntime:
        return self._runtime

    @property
    def slow_parameters(self) -> tuple[torch.nn.Parameter, ...]:
        return self._slow_parameters

    def scaler_found_inf(self, grad_scaler: torch.amp.GradScaler, optimizer: torch.optim.Optimizer) -> bool:
        """Whether the trainer's GradScaler step recorded found-inf (a skip) for this optimizer.

        Callers must check ``grad_scaler.is_enabled()`` first: a disabled
        scaler never records per-optimizer state and never skips.
        """
        return scaler_found_inf(grad_scaler, optimizer)

    def process_sample(
        self,
        *,
        owner_key: str,
        terminal: bool,
        valid: bool,
        source: dict[str, torch.Tensor],
    ) -> torch.Tensor | None:
        """Advance one native window (one evidence row); return the ``[K_local, D_local]`` prefix token.

        Window t reads exactly the post-update result of rows 0..t-1 and never
        its own row: intermediate windows read the detached candidate, closing
        windows read the pre-write witness graph built by materialize.
        """
        if not isinstance(owner_key, str) or not owner_key:
            raise ValueError("owner_key must be a non-empty string.")
        if not isinstance(terminal, bool) or not isinstance(valid, bool):
            raise ValueError("terminal and valid must be bool.")
        if torch.is_inference_mode_enabled() or not torch.is_grad_enabled():
            raise RuntimeError("TTT lifecycle requires ordinary training grad mode.")
        if not valid:
            # N_valid_window=0: no admit, no read, no prefix. A terminal
            # zero-valid window closes its segment by abort, never by commit.
            if terminal:
                self._abort_segment(owner_key)
                self._reset_owner(owner_key)
            return None
        owner = self._owners.setdefault(owner_key, _OwnerState())
        segment = self._segments.get(owner_key)
        if segment is None:
            self._runtime.begin_segment(owner_key)
            segment = _OpenSegment(epoch=owner.epoch, base_state=owner.committed_state)
            self._segments[owner_key] = segment
        source_timestep = owner.committed_timestep + segment.row_count + 1
        admission = self._runtime.admit_evidence(
            owner_key=owner_key,
            source_identity=f"{owner_key}#w{source_timestep}",
            source_timestep=source_timestep,
            source=source,
            epoch=segment.epoch,
        )
        if isinstance(admission, ReplayRecord):
            raise RuntimeError("live admission must not resolve to a committed replay")
        segment.row_count += 1
        device = next(iter(source.values())).device
        if segment.candidate is None:
            # The detached candidate resumes from the last committed state so
            # later segments of the same owner continue the chronology instead
            # of restarting from w0.
            base = segment.base_state if segment.base_state is not None else self._core.initial_state(1, device=device)
            segment.candidate = self._core.detach_state(base)
        # Exactly one detached write per window; the read is taken from the
        # pre-write state (read-after-(t-1)) before the write advances the candidate.
        with torch.no_grad():
            evidence_t = self._encoder(**source)[:, -1]
            key_t, query_base_t, value_t = self._core.project_evidence(evidence_t)
            detached_token = self._core.read_many(self._core.project_queries(query_base_t), segment.candidate)[0]
        _, updated_candidate, _ = self._core.step_projected_many(
            key_t=key_t,
            query_base_t=query_base_t,
            value_t=value_t,
            state_in=segment.candidate,
            valid=torch.ones(1, dtype=torch.bool, device=device),
            create_graph=False,
        )
        segment.candidate = self._core.detach_state(updated_candidate)
        if segment.row_count != self._core.ttt_tbptt_steps and not terminal:
            return detached_token
        # Closing window (segment full or terminal remainder): the authority
        # replays the pending raw rows from the committed base with
        # create_graph=True, emitting pre-write witness tokens
        # (read(S_{t-1}) per row). This window's read is the last row's
        # witness token — bitwise the same values as the detached candidate
        # read above, now carrying the witness graph. The committed/candidate
        # numeric state is driven exclusively by the detached write sequence;
        # materialize provides the equivalent differentiable graph and does
        # not apply the writes a second time.
        witness = self._runtime.materialize(owner_key, emit_prewrite_tokens=True)
        segment.armed = True
        segment.terminal = terminal
        return witness[-1]

    def observe_loss(self, loss: torch.Tensor) -> None:
        """Record this micro-batch's unscaled native loss (the trainer ``training_step`` return value)."""
        self._loss = loss

    def on_after_backward(self) -> None:
        """Mark armed segments after the trainer's single backward, then commit; terminal resets."""
        armed = [owner_key for owner_key, segment in self._segments.items() if segment.armed]
        try:
            for owner_key in armed:
                self._runtime.mark_external_backward(owner_key, loss=self._loss)
        except Exception:
            self.abort_open_segments()
            raise
        for owner_key in armed:
            segment = self._segments.pop(owner_key)
            owner = self._owners[owner_key]
            self._runtime.finish(owner_key, terminal=segment.terminal)
            if segment.terminal:
                owner.epoch += 1
                owner.committed_state = None
                owner.committed_timestep = -1
            else:
                owner.committed_state = ContinualTTTFastState(*(value.detach().clone() for value in segment.candidate))
                owner.committed_timestep += segment.row_count
        self._loss = None

    def abort_open_segments(self) -> None:
        """Roll every open segment back to the last committed state; clear armed marks."""
        for owner_key in list(self._segments):
            self._abort_segment(owner_key)
        self._loss = None

    def resolve_transaction(self, result: str) -> None:
        """Handle the trainer's exactly-once optimizer-transaction resolution."""
        if result not in _RESOLUTIONS:
            raise ValueError(f"unknown transaction resolution: {result!r}")
        if result == RESOLUTION_SUCCESS:
            if any(segment.armed for segment in self._segments.values()):
                raise RuntimeError("an armed segment survived the backward seam without commit.")
        elif result == RESOLUTION_SCALER_SKIP:
            # Option B semantics: prior commits stay valid; only the slow side
            # does not advance. Drop any accumulated meta-gradients so the next
            # backward starts clean.
            for parameter in self._slow_parameters:
                parameter.grad = None
        # RESOLUTION_EXCEPTION is process-fatal: the trainer re-raises the
        # original exception; no runtime rollback is attempted here.

    def _abort_segment(self, owner_key: str) -> None:
        if self._segments.pop(owner_key, None) is not None:
            self._runtime.abort(owner_key)

    def _reset_owner(self, owner_key: str) -> None:
        self._runtime.reset(owner_key)
        owner = self._owners.setdefault(owner_key, _OwnerState())
        owner.epoch += 1
        owner.committed_state = None
        owner.committed_timestep = -1


class TTTLifecycleCallback(Callback):
    """Trainer-seam glue for the R09-B TTT external-backward lifecycle.

    ``on_before_backward`` records this micro-batch's unscaled native loss;
    ``on_after_backward`` (after the trainer's exactly-once backward) marks
    every armed segment with the dual evidence (finite loss + traversed
    witness graph), commits it, and resets terminal owners. The model the
    trainer passes to these hooks is already DDP-unwrapped. When
    ``local_ttt_enabled=False`` the model never creates ``_ttt_lifecycle``
    and both hooks are exact no-ops.
    """

    @staticmethod
    def _lifecycle(model: torch.nn.Module) -> TTTLifecycle | None:
        model = getattr(model, "module", model)
        return getattr(model, "_ttt_lifecycle", None)

    def on_before_backward(self, model: torch.nn.Module, loss: torch.Tensor, iteration: int = 0) -> None:
        lifecycle = self._lifecycle(model)
        if lifecycle is not None:
            lifecycle.observe_loss(loss)

    def on_after_backward(self, model: torch.nn.Module, iteration: int = 0) -> None:
        lifecycle = self._lifecycle(model)
        if lifecycle is not None:
            lifecycle.on_after_backward()
