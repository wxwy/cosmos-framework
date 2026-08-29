import json

import torch
from torch import nn

from cosmos_framework.callbacks.r09_a1_runtime_probe import R09A1RuntimeProbeCallback
from cosmos_framework.model.generator.mot.local_evidence import (
    LocalEvidenceEncoder,
    LocalHistoryRuntime,
    RecurrentLocalMemoryBackend,
    StatelessLocalReplayReadout,
)


def test_probe_records_exact_targets_gradients_and_state_contract(tmp_path) -> None:
    model = nn.Module()
    model.net = nn.Module()
    model.net.local_history_runtime = LocalHistoryRuntime(
        LocalEvidenceEncoder(evidence_dim=8, visual_dim=4, action_dim=3, max_age_steps=8),
        StatelessLocalReplayReadout(evidence_dim=8, local_dim=5, hidden_dim=8),
        RecurrentLocalMemoryBackend(evidence_dim=8, local_dim=5),
    )
    model.net.local_memory2llm = nn.Linear(5, 7)
    model.net.local_memory_modality_embed = nn.Parameter(torch.zeros(7))
    probe = R09A1RuntimeProbeCallback(str(tmp_path / "probe.json"))
    probe.on_train_start(model)
    targets = probe._targets(model)
    optimizer = torch.optim.SGD(targets.values(), lr=0.1)
    tokens, present, _ = model.net.local_history_runtime(
        history_visual_summary=torch.randn(2, 3, 4),
        local_history_action=torch.randn(2, 3, 3),
        history_age_steps=torch.zeros(2, 3, dtype=torch.long),
        history_dt_s=torch.ones(2, 3, 1),
        history_mask=torch.ones(2, 3, dtype=torch.bool),
    )
    assert present.all()
    (model.net.local_memory2llm(tokens).sum() + model.net.local_memory_modality_embed.sum()).backward()
    probe.on_before_optimizer_step(model, optimizer, None, None, iteration=0)
    probe.on_training_step_end(model, {}, {}, torch.tensor(1.0), iteration=1)
    payload = json.loads((tmp_path / "probe.json").read_text())
    assert payload["representative_step"]["optimizer_matches_targets"] is True
    assert not payload["representative_step"]["unexpected_optimizer_names"]
    assert all(record["present"] and record["finite"] for record in payload["representative_step"]["gradients"].values())
    assert all(payload["representative_step"]["active_group_has_nonzero_grad"].values())
    assert payload["state_contract"]["segment_token_max_abs_diff"] == 0.0
    assert payload["state_contract"]["segment_state_before_detach_requires_grad"] is True
    assert payload["state_contract"]["segment_state_after_detach_requires_grad"] is False
    assert payload["state_contract"]["segment_detach_value_exact"] is True
    assert payload["state_contract"]["reset_all_mask_selected_absent"] is True
