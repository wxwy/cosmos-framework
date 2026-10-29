"""Actual native loop and transaction facade with CPU Local state/fake physics."""

from __future__ import annotations

import base64
import io
import json
import threading
import time

import numpy as np
import pytest
import torch
from PIL import Image

from cosmos_framework.inference.robocasa_batch_transactions_test import make_native as make_native
from cosmos_framework.inference.robocasa_session_transactions import Reply, json_bytes
from cosmos_framework.inference.robocasa_session_transactions_test import clock_kernel as clock_kernel
from cosmos_framework.inference.robocasa_session_transactions_test import payload, register
from cosmos_framework.scripts.robocasa_transactions_b1_test import DirectTransport, Response
from cosmos_framework.simulation.robocasa import closed_loop_eval as native
from cosmos_framework.simulation.robocasa import eval_v2_adapter as adapter_module
from cosmos_framework.simulation.robocasa.eval_v2_transactions import PolicyTransport


class FakeEnv:
    def __init__(self):
        self.steps = 0
        self.actions = []
        self.obs = {f"{cam}_image": np.zeros((256, 256, 3), np.uint8) for cam in native.CAMS}

    def reset(self):
        self.steps = 0
        return self.obs

    def _check_success(self):
        return self.steps >= 32

    def step(self, action):
        self.actions.append(action.copy())
        self.steps += 1
        return self.obs, 0.0, False, {}

    def get_ep_meta(self):
        return {"lang": "Actual Episode instruction / 关闭冰箱"}


def settings(mode="none"):
    return dict(
        server_url="http://cpu",
        predict_timeout_s=5,
        max_infer_batch_size=1,
        execution_profile="phase_b",
        save_generated_video=mode,
        save_rollout_mp4=False,
        encoder_grace_s=0.05,
        horizon=32,
    )


def facade(obj, actor, lease, *, mode="none", deadline=None, drop=(), media_transport=None):
    return PolicyTransport(
        settings(mode),
        {"lease": lease, "prediction_clock_domain": actor.clock_domain},
        time.monotonic() + 10 if deadline is None else deadline,
        transport=DirectTransport(obj, drop),
        media_transport=media_transport or DirectTransport(obj),
    )


def test_native_32_step_loop_corrected_actions_two_replans_and_real_End_cleanup(tmp_path, monkeypatch, make_native):
    obj, actor = make_native(1, 1)
    lease = register(actor)

    def decode(value):
        return torch.from_numpy(np.array(Image.open(io.BytesIO(base64.b64decode(value))), copy=True)).permute(2, 0, 1)

    obj.local_memory_adapter.decode_image = decode
    generate = obj.model.generate_samples_from_batch

    def valid_actions(batch, **kwargs):
        samples = generate(batch, **kwargs)
        raw = [0.2, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0.5]
        action = torch.zeros(1, 17, 64)
        action[:, 1:, :15] = torch.tensor(raw) * 2 - 1
        samples["action"] = [action.clone() for _ in batch["prompt"]]
        return samples

    obj.model.generate_samples_from_batch = valid_actions
    assigned = facade(obj, actor, lease)
    # Inject only transport construction; the actual facade and native loop run.
    from cosmos_framework.simulation.robocasa import eval_v2_transactions

    monkeypatch.setattr(eval_v2_transactions, "PolicyTransport", lambda *a: assigned)
    monkeypatch.setattr(native, "build_state_token", lambda _: [0.0] * 15)
    env = FakeEnv()
    episode = adapter_module.NativeEpisode(env, settings())
    assert episode.initialize()["initial_success"] is False
    attempt = {
        "task": "CloseFridge",
        "episode": 0,
        "attempt_id": "attempt-1",
        "stream_id": "stream",
        "seed": 0,
        "manifest_digest": "a" * 64,
    }
    result = episode.execute(
        attempt,
        reset_index=0,
        directory=tmp_path / "episode",
        transaction_assignment={"lease": lease, "prediction_clock_domain": actor.clock_domain},
        episode_deadline=time.monotonic() + 10,
    )
    assert result["observation"]["environment_success"] is True and len(env.actions) == 32
    assert result["local_session"]["reset_ack"] is True and actor.info()["sessions"] == 0
    rows = [json.loads(x) for x in (tmp_path / "episode/predictions.jsonl").read_text().splitlines()]
    assert [r["source_step"] for r in rows] == [0, 16]
    assert [r["local_memory"]["prefix_present"] for r in rows] == [False, True]
    assert all(r["transaction"]["assignment"]["session_id"] == lease["session_id"] for r in rows)
    assert len(obj.native_calls) == 2


@pytest.mark.parametrize("lost", [False, True])
def test_one_successful_native_frontier_is_never_delivered_twice(make_native, lost):
    obj, actor = make_native(1, 1)
    lease = register(actor)
    client = facade(obj, actor, lease, drop=("/v2/inference",) if lost else ())
    request = payload(lease)
    client.post("http://cpu/predict", json=request, timeout=5)
    with pytest.raises(ValueError, match="frontier twice"):
        client.post("http://cpu/predict", json=request, timeout=5)
    assert len(obj.native_calls) == 1
    client.close()


def test_expired_Episode_cannot_mint_another_prediction_budget(make_native):
    obj, actor = make_native(1, 1)
    lease = register(actor)
    client = facade(obj, actor, lease, deadline=time.monotonic() - 1)
    with pytest.raises(TimeoutError, match="Episode deadline"):
        client.post("http://cpu/predict", json=payload(lease), timeout=5)
    with pytest.raises(RuntimeError, match="new Attempt"):
        client.post("http://cpu/predict", json=payload(lease), timeout=5)
    assert not obj.native_calls
    client.close()


def test_failed_generation_poisoned_facade_cannot_retry_as_new_predict(make_native):
    obj, actor = make_native(1, 1)
    lease = register(actor)
    client = facade(obj, actor, lease)
    actor.generate_many = lambda *a: (_ for _ in ()).throw(RuntimeError("injected native failure"))
    with pytest.raises(RuntimeError, match="transaction API rejected"):
        client.post("http://cpu/predict", json=payload(lease), timeout=5)
    with pytest.raises(RuntimeError, match="new Attempt"):
        client.post("http://cpu/predict", json=payload(lease, 1, 0), timeout=5)
    client.close()
    assert actor.info()["sessions"] == 0


def test_media_response_after_sink_fence_cannot_write_or_resurrect_directory(make_native):
    obj, actor = make_native(1, 1)
    lease = register(actor)
    entered, release = threading.Event(), threading.Event()
    key, sha = "b" * 64, "c" * 64

    class LateMedia:
        def request(self, *a, **k):
            entered.set()
            assert release.wait(3)
            return Response(
                Reply(
                    200,
                    json_bytes(
                        {"request_key": key, "payload_sha256": sha, "media": {"status": "ready", "video": ["x"]}}
                    ),
                )
            )

    client = facade(obj, actor, lease, media_transport=LateMedia())
    delivered = []
    client.media(
        {"media": {"status": "pending", "ticket": key}, "request_key": key, "payload_sha256": sha},
        0,
        lambda *a: delivered.append(a),
    )
    try:
        assert entered.wait(3)
        client.close_media()
        assert client.media_process_fault
    finally:
        release.set()
        client._thread.join(3)
    assert not client.media_process_fault and not delivered
    client.close()


def test_End_ACK_timeout_cannot_be_reported_as_reset_success(make_native):
    obj, actor = make_native(1, 1)
    lease = register(actor)
    client = facade(obj, actor, lease)
    now = [0.0]
    client.settings["predict_timeout_s"] = 0.1
    client.client.clock = lambda: now[0]
    client.client.pause = lambda n: now.__setitem__(0, now[0] + n)
    end = actor.end
    actor.end = lambda proof: Reply(202, json_bytes({"acknowledged": False}))
    try:
        with pytest.raises(TimeoutError, match="original transaction deadline"):
            client.post(
                "http://cpu/predict",
                json={"_local_memory_command": "reset", "session_id": lease["session_id"]},
                timeout=5,
            )
        assert client._ended is False and client.client.lease is not None
    finally:
        actor.end = end
        client.close()


def test_foreign_optional_media_is_diagnostic_and_never_invalidates_actions(make_native):
    obj, actor = make_native(1, 1)
    lease = register(actor)
    client = facade(obj, actor, lease)
    delivered = []
    client.media(
        {"request_key": "b" * 64, "payload_sha256": "c" * 64, "media": {"status": "pending", "ticket": "d" * 64}},
        0,
        lambda *a: delivered.append(a),
    )
    assert not delivered and client.media_diagnostics()["predictions"]["0"]["status"] == "invalid_media_binding"
    client.close()
