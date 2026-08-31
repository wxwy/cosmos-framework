import json

import torch
from torch import nn

from cosmos_framework.callbacks.r09_b1_runtime_probe import R09B1RuntimeProbeCallback
from cosmos_framework.model.generator.mot.local_evidence import (
    LocalEvidenceEncoder,
    LocalHistoryRuntime,
    StatelessLocalReplayReadout,
    TTTLocalMemoryBackend,
)


def test_probe_records_ttt_state_and_training_gradient_contract(tmp_path) -> None:
    model = nn.Module()
    model.net = nn.Module()
    model.net.local_history_runtime = LocalHistoryRuntime(
        LocalEvidenceEncoder(evidence_dim=8, visual_dim=4, action_dim=3, max_age_steps=8),
        StatelessLocalReplayReadout(evidence_dim=8, local_dim=5, hidden_dim=8),
        TTTLocalMemoryBackend(evidence_dim=8, local_dim=5, segment_steps=4),
    )
    model.net.local_memory2llm = nn.Linear(5, 7).to(dtype=torch.bfloat16)
    model.net.local_memory_modality_embed = nn.Parameter(torch.zeros(7, dtype=torch.bfloat16))
    probe_path = tmp_path / "probe.json"
    probe = R09B1RuntimeProbeCallback(str(probe_path))
    probe.on_train_start(model)
    optimizer = torch.optim.SGD(probe._targets(model).values(), lr=0.1)
    tokens, present, _ = model.net.local_history_runtime(
        history_visual_summary=torch.randn(2, 5, 4),
        local_history_action=torch.randn(2, 5, 3),
        history_age_steps=torch.zeros(2, 5, dtype=torch.long),
        history_dt_s=torch.ones(2, 5, 1),
        history_mask=torch.ones(2, 5, dtype=torch.bool),
    )
    assert present.all()
    (model.net.local_memory2llm(tokens).float().sum() + model.net.local_memory_modality_embed.float().sum()).backward()
    probe.on_before_optimizer_step(model, optimizer, None, None, iteration=0)
    probe.on_training_step_end(model, {}, {}, torch.tensor(1.0), iteration=1)
    payload = json.loads(probe_path.read_text())
    assert payload["representative_step"]["optimizer_matches_targets"] is True
    assert not payload["representative_step"]["unexpected_optimizer_names"]
    assert all(payload["state_contract"]["members_detached"])
    assert payload["state_contract"]["segment_token_max_abs_diff"] == 0.0
    assert all(payload["state_contract"]["segment_members_exact"])
    assert payload["state_contract"]["bytes_per_sample"] == 169
    assert not any(item["present"] for item in payload["representative_step"]["gradient_groups"]["encoder"])
    for group in ("local_memory2llm", "local_memory_modality_embed"):
        assert all(item["present"] and item["finite"] and item["max_abs"] > 0 for item in payload["representative_step"]["gradient_groups"][group])
