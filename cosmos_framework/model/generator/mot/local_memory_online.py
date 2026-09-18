"""Transactional online TTT memory, separate from training GA and optimizer state.

Inputs are causal PREVIOUS-observation 96-d features and the ACTUALLY executed,
normalized 10-d action. Predicted actions and future latent labels are not evidence.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace

import torch

from .local_evidence import ContinualTTTFastState, LocalMemoryRuntime


@dataclass(frozen=True)
class OnlineTransition:
    slot_id: int
    episode_id: str
    consumer_step: int
    previous_visual_summary: torch.Tensor | None = None
    previous_executed_action: torch.Tensor | None = None
    reset_before: bool = False


@dataclass(frozen=True)
class PreparedOnlineMemory:
    tokens: tuple[torch.Tensor | None, ...]


class OnlineLocalMemorySession:
    """Per-model, multi-slot state; caller serializes concurrent requests."""

    def __init__(self, runtime: LocalMemoryRuntime) -> None:
        if not isinstance(runtime, LocalMemoryRuntime):
            raise TypeError("online memory requires the loaded canonical LocalMemoryRuntime")
        self.runtime = runtime
        self._records = {}
        self._pending = None
        self._candidate_records = None

    @staticmethod
    def _fingerprint(item: OnlineTransition) -> str:
        digest = hashlib.sha256(repr((item.slot_id, item.episode_id, item.consumer_step, item.reset_before)).encode())
        for value in (item.previous_visual_summary, item.previous_executed_action):
            if value is not None:
                digest.update(str((value.dtype, tuple(value.shape))).encode())
                digest.update(value.detach().float().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def prepare(self, transitions: tuple[OnlineTransition, ...]) -> PreparedOnlineMemory:
        if self._pending is not None or not transitions:
            raise RuntimeError("online memory already pending or request empty")
        if len({item.slot_id for item in transitions}) != len(transitions):
            raise ValueError("one online batch may not repeat a stream slot")
        records = self._records.copy()
        priors, fingerprints, replay, valid = [], [], [], []
        for item in transitions:
            if (
                type(item.slot_id) is not int
                or item.slot_id < 0
                or not item.episode_id
                or type(item.consumer_step) is not int
                or item.consumer_step < 0
            ):
                raise ValueError("invalid online slot/episode/step")
            if item.consumer_step == 0:
                if item.previous_visual_summary is not None or item.previous_executed_action is not None:
                    raise ValueError("S0 must have absent previous evidence")
            else:
                if item.reset_before:
                    raise ValueError("reset must start at consumer step zero")
                for value, width in ((item.previous_visual_summary, 96), (item.previous_executed_action, 10)):
                    if (
                        not isinstance(value, torch.Tensor)
                        or value.shape != (width,)
                        or not torch.isfinite(value).all()
                    ):
                        raise ValueError("online evidence must be finite canonical visual96/action10")
            fingerprint = self._fingerprint(item)
            prior = records.get(item.slot_id)
            repeated = prior is not None and prior[:2] == (item.episode_id, item.consumer_step)
            if repeated and fingerprint != prior[2]:
                raise ValueError("online replay changed evidence bytes")
            if item.consumer_step == 0 and not repeated:
                if prior is not None and not item.reset_before:
                    raise ValueError("reusing a slot requires explicit reset_before")
                prior = None
            elif not repeated and (prior is None or prior[0] != item.episode_id or prior[1] + 1 != item.consumer_step):
                raise ValueError("online evidence is not an exact continuation")
            priors.append(prior)
            fingerprints.append(fingerprint)
            replay.append(repeated)
            valid.append(item.consumer_step > 0 and not repeated)
        core, encoder = self.runtime.ttt_core, self.runtime.evidence_encoder
        device = next(encoder.parameters()).device
        with torch.inference_mode(False), torch.enable_grad():
            fresh = core.initial_state(len(transitions), device=device)
            state = ContinualTTTFastState(
                *(
                    torch.stack(
                        [
                            initial[row] if prior is None else prior[3][field][0].to(device).clone()
                            for row, prior in enumerate(priors)
                        ]
                    )
                    for field, initial in enumerate(fresh)
                )
            )
            visual = torch.zeros(len(transitions), 1, 96, device=device)
            action = torch.zeros(len(transitions), 1, 10, device=device)
            for row, item in enumerate(transitions):
                if valid[row]:
                    visual[row, 0] = item.previous_visual_summary.detach().to(device).clone()
                    action[row, 0] = item.previous_executed_action.detach().to(device).clone()
            mask = torch.tensor(valid, dtype=torch.bool, device=device).unsqueeze(1)
            local, updated, _ = core.scan_segment_masked_encoded_many(
                encoder,
                visual,
                action,
                mask,
                state,
                create_graph=False,
            )
            outputs = []
            for row, item in enumerate(transitions):
                token = priors[row][4] if replay[row] else (local[row, 0] if valid[row] else None)
                token = None if token is None else token.detach().clone()
                row_state = ContinualTTTFastState(*(value[row : row + 1].detach().float().clone() for value in updated))
                records[item.slot_id] = (
                    item.episode_id,
                    item.consumer_step,
                    fingerprints[row],
                    row_state,
                    None if token is None else token.clone(),
                )
                outputs.append(token)
        prepared = PreparedOnlineMemory(tuple(outputs))
        self._pending, self._candidate_records = prepared, records
        return prepared

    def commit(self, prepared: PreparedOnlineMemory) -> None:
        if prepared is not self._pending:
            raise RuntimeError("online commit capability is stale or foreign")
        self._records = self._candidate_records
        self._pending = self._candidate_records = None

    def abort(self, prepared: PreparedOnlineMemory) -> None:
        if prepared is not self._pending:
            raise RuntimeError("online abort capability is stale or foreign")
        self._pending = self._candidate_records = None

    def reset(self, slot_id: int) -> None:
        if self._pending is not None:
            raise RuntimeError("cannot reset an online slot with a pending generation")
        self._records.pop(slot_id, None)


def generate_with_local_memory(model, data_batch, transitions, session, *, use_local_tokens=True, **kwargs):
    """Inject causal clean prefixes into the REAL generation action path.

    The client provides canonical executed-action evidence, not predicted actions.
    No state is published if native generation fails or returns non-finite actions.
    """
    if getattr(model.net, "local_memory_runtime", None) is not session.runtime:
        raise ValueError("online memory belongs to another model")
    if not getattr(model.config, "local_memory_enabled", False) or model.training:
        raise ValueError("online generation requires an eval-mode Local-enabled model")
    if len(data_batch.get("sequence_plan", ())) != len(transitions):
        raise ValueError("online transitions and native sequence plans must align")
    prepared = session.prepare(tuple(transitions))
    try:
        tokens = prepared.tokens if use_local_tokens else (None,) * len(transitions)
        batch = dict(data_batch)
        batch["local_memory"] = list(tokens)
        batch["sequence_plan"] = [
            replace(plan, has_local_memory=token is not None)
            for plan, token in zip(data_batch["sequence_plan"], tokens, strict=True)
        ]
        with torch.inference_mode():
            result = model.generate_samples_from_batch(batch, **kwargs)
        actions = result.get("action")
        if not isinstance(actions, (list, tuple)) or len(actions) != len(transitions):
            raise ValueError("native policy generation returned no aligned actions")
        if any(not isinstance(value, torch.Tensor) or not torch.isfinite(value).all() for value in actions):
            raise ValueError("native policy generation returned non-finite actions")
    except Exception:
        session.abort(prepared)
        raise
    session.commit(prepared)
    return result
