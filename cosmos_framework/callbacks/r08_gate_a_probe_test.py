import json

import torch
from torch import nn

from cosmos_framework.callbacks.r08_gate_a_probe import R08GateAProbeCallback


def test_probe_records_optimizer_gradients_and_updates(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("cosmos_framework.callbacks.r08_gate_a_probe.distributed.is_rank0", lambda: True)
    model = nn.Module()
    model.net = nn.Module()
    model.net.local_history_runtime = nn.Linear(2, 2)
    model.net.local_memory2llm = nn.Linear(2, 2)
    model.net.local_memory_modality_embed = nn.Parameter(torch.zeros(2))
    output = tmp_path / "probe.json"
    callback = R08GateAProbeCallback(str(output))
    callback.on_training_step_start(model, {}, iteration=0)
    optimizer = torch.optim.SGD(model.net.parameters(), lr=0.1)
    loss = sum(value.square().sum() for value in model.net.parameters())
    loss.backward()
    callback.on_before_optimizer_step(model, type("Container", (), {"optimizers": [optimizer]})(), None, None, iteration=0)
    optimizer.step()
    callback.on_training_step_end(model, {}, {}, loss, iteration=1)
    payload = json.loads(output.read_text())
    assert all(payload["optimizer_membership"].values())
    assert all(summary["present"] and summary["finite"] for summary in payload["gradients"].values())
    assert any(summary["max_abs"] > 0 for summary in payload["updates"].values())
