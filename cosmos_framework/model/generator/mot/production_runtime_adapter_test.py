import pytest
import torch

from .local_evidence import ContinualTTTLocalMemoryCore, LocalEvidenceEncoder
from .production_runtime_adapter import ProductionLocalMemoryRuntime
from .runtime_authority import AdmissionAuthority


def _source(value: float = 0.0) -> dict[str, torch.Tensor]:
    return {"history_visual_summary": torch.full((1, 1, 96), value), "local_history_action": torch.zeros(1, 1, 10),
            "history_age_steps": torch.zeros(1, 1, dtype=torch.long), "history_dt_s": torch.zeros(1, 1, 1),
            "history_mask": torch.ones(1, 1, dtype=torch.bool)}


def _runtime(steps: int = 1) -> ProductionLocalMemoryRuntime:
    return ProductionLocalMemoryRuntime(AdmissionAuthority(), LocalEvidenceEncoder(), ContinualTTTLocalMemoryCore(ttt_tbptt_steps=steps))


def test_production_runtime_one_step_and_reset_epoch() -> None:
    runtime = _runtime(); source = _source(); runtime.begin_segment("ep/0")
    assert runtime.admit_evidence(owner_key="ep/0", source_identity="s0:0", source_timestep=0, source=source) == (0, 0)
    token = runtime.materialize("ep/0")
    runtime.backward("ep/0", token.float().sum() * 0); runtime.finish("ep/0", terminal=False)
    assert runtime.write_count == 1
    runtime.reset("ep/0"); runtime.begin_segment("ep/0")
    with pytest.raises(ValueError, match="epoch"):
        runtime._runtime.admit(runtime.authority.issue(owner_key="ep/0", source_identity="s0:0", source_timestep=0, source=source, epoch=0), source=source)


def test_production_disabled_path_is_zero_write_and_none_payload() -> None:
    runtime = _runtime(); sample = torch.tensor([[1.0, 2.0]])
    packed, loss = runtime.disabled_path(sample)
    assert packed[1] is None and torch.equal(packed[0], sample) and torch.equal(loss, sample.square().mean())
    assert runtime.write_count == 0
