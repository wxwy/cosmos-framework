"""Actual service batch method + original per-image encode hooks; CPU boundaries."""

from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.inference import robocasa_boundary_capture as capture
from cosmos_framework.inference import robocasa_boundary_capture_test as b
from cosmos_framework.inference import robocasa_session_transactions_test as s
from cosmos_framework.scripts import robocasa_feature_http_test as h

wire_rig = h.wire_rig
clock_kernel = h.clock_kernel
capture_methods = h.capture_methods


@pytest.mark.parametrize("count", [1, 2, 4, 8])
def test_actual_shared_policy_batch_keeps_native_generation_and_observes_each_encode_once(
    wire_rig, capture_methods, monkeypatch, count
):
    rig = wire_rig
    model, calls = b._model(capture_methods)
    rig.service.model = model
    updates = [object() for _ in range(count)]
    staged = []
    leases = [
        dict(rig.lease, session_id=f"session{i}", episode_id=f"episode{i}", slot_id=f"slot{i}") for i in range(count)
    ]

    def context(update):
        i = updates.index(update)
        return dict(
            lease={k: v for k, v in leases[i].items() if k != "capability"},
            request_key=f"request{i}",
            chunk_id=f"chunk{i}",
            source_step=0,
            identity=h._identity(),
            deadline_ns=105 * 10**9,
            clock_domain=rig.actor.clock_domain,
        )

    rig.service.local_memory_adapter = SimpleNamespace(
        mode="required",
        prefixes=lambda u: (None,),
        boundary_context=context,
        stage_boundary=lambda u, boundary: staged.append((u, boundary)),
    )
    rig.service.transactions = SimpleNamespace(media_room=lambda: (0, 0))
    single, group = capture.capture_policy_boundary, capture.capture_policy_boundaries
    monkeypatch.setattr(
        capture, "capture_policy_boundary", lambda *a, **kw: single(*a, **kw, clock_ns=lambda: 100 * 10**9)
    )
    monkeypatch.setattr(
        capture,
        "capture_policy_boundaries",
        lambda m, contexts: group(m, [dict(c, clock_ns=lambda: 100 * 10**9) for c in contexts]),
    )
    generations = []

    def generate(batch, **kw):
        generations.append(len(batch["video"]))
        assert kw["_local_memory_prefixes"] == (None,) * count
        model._normalize_video_databatch_inplace(batch)
        full = model._encode_vision_x0_tokens_prefix(batch["video"], [[0]] * count)
        cropped = model._remove_padding_from_latent(full, batch["image_size"])
        assert all(x.shape[-2:] == (10, 20) for x in cropped)
        return {"action": [torch.zeros(1, 17, 64) for _ in range(count)], "vision": None}

    model.generate_samples_from_batch = generate
    output = rig.service._transaction_batch([s.payload(l) for l in leases], updates)
    assert generations == [count] and len(calls) == count and len(output.actions) == count
    assert all(c.shape == (1, 3, 1, 192, 320) for c in calls)
    assert [u for u, boundary in staged] == updates
    assert [boundary.header["lease"]["session_id"] for u, boundary in staged] == [l["session_id"] for l in leases]
    assert all(boundary.header["policy_request_key"] == f"request{i}" for i, (u, boundary) in enumerate(staged))
    assert all(len(a) == 16 and all(len(row) == 15 for row in a) for a in output.actions)
    assert not rig.boundaries  # Only the original transaction may publish staged boundaries.
