"""Policy integration witnesses: prefix enters action path, failure never commits."""

from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.inference.local_memory_policy import PolicyLocalMemoryAdapter
from cosmos_framework.model.generator.mot.local_evidence import (
    CANONICAL_EVIDENCE_FEATURE_CONFIG,
    ContinualTTTLocalMemoryCore,
    LocalEvidenceEncoder,
    RecurrentLocalMemoryBackend,
)


def adapter(mode="auto"):
    runtime = SimpleNamespace(
        evidence_encoder=LocalEvidenceEncoder(feature_config=CANONICAL_EVIDENCE_FEATURE_CONFIG),
        ttt_core=ContinualTTTLocalMemoryCore(),
    )
    model = SimpleNamespace(
        config=SimpleNamespace(local_ttt_enabled=True), net=SimpleNamespace(local_memory_runtime=runtime)
    )
    service = SimpleNamespace(
        model=model,
        action_normalization="quantile",
        action_min=torch.full((10,), -1.0),
        action_range=torch.full((10,), 2.0),
    )
    return PolicyLocalMemoryAdapter(service, mode=mode)


def req(step=0, session="s"):
    evidence = [{"source_step": t, "visual_summary": [0.1] * 96, "executed_action": [0.2] * 10} for t in range(step)]
    return {
        "local_memory": {"session_id": session, "episode_id": "episode", "consumer_step": step, "evidence": evidence}
    }


def batch(n=1):
    return {"sequence_plan": [SimpleNamespace(has_local_memory=False) for _ in range(n)]}


def test_online_prefix_reaches_generation_and_both_loops_are_separate():
    policy = adapter()
    first = batch()
    output = policy.generate([req()], first, lambda: {"action": [torch.ones(16, 10)]})
    assert first["local_memory"] == [None]
    assert not output["_local_memory_status"][0]["prefix_present"]
    second = batch()

    def generate():
        assert not torch.is_grad_enabled()
        assert second["sequence_plan"][0].has_local_memory
        assert tuple(second["local_memory"][0].shape) == (1, 32)
        return {"action": [second["local_memory"][0].sum().expand(16, 10)]}

    result = policy.generate([req(3)], second, generate)
    assert result["_local_memory_status"][0]["prefix_present"]
    assert policy.memory.metadata()["steps"] == {"s": 3}


@pytest.mark.parametrize("failure", ["exception", "nan", "count"])
def test_failed_generation_keeps_previous_fast_state(failure):
    policy = adapter()
    policy.generate([req()], batch(), lambda: {"action": [torch.zeros(16, 10)]})
    state = policy.memory._records["s"]

    def generate():
        if failure == "exception":
            raise RuntimeError("synthetic generate failure")
        return {"action": [torch.full((16, 10), float("nan"))] if failure == "nan" else []}

    with pytest.raises((RuntimeError, FloatingPointError)):
        policy.generate([req(3)], batch(), generate)
    assert policy.memory._records["s"] is state
    assert not policy.memory._pending


def test_missing_evidence_version_and_disabled_mode_are_fail_closed():
    policy = adapter()
    with pytest.raises(ValueError, match="requires a local_memory"):
        policy.generate([{}], batch(), lambda: None)
    request = req()
    request["local_memory"]["evidence_version"] = "unreviewed"
    with pytest.raises(ValueError, match="version"):
        policy.generate([request], batch(), lambda: None)
    disabled = adapter("off")
    with pytest.raises(ValueError, match="disabled"):
        disabled.generate([req()], batch(), lambda: None)
    assert disabled.generate([{}], batch(), lambda: {"ok": True}) == {"ok": True}


def test_raw_libero_action_conversion_uses_the_training_axisangle_to_6d_rule():
    policy = adapter()
    raw = torch.tensor([0.1, -0.2, 0.3, 0.01, 0.02, -0.03, -1.0])
    from cosmos_framework.data.generator.action.datasets.libero_lerobot_dataset import LIBEROLeRobotDataset

    dataset = object.__new__(LIBEROLeRobotDataset)
    dataset._rotation_space = "6d"
    expected = dataset._build_frame_wise_action(raw.numpy()[None])[0]
    torch.testing.assert_close(policy._normalize_executed_action(raw, gripper_mode="pm_one"), expected)

    # NVIDIA zero_one dataset: env_gripper = 1 - 2 * raw_gripper.
    training_raw = raw.clone()
    training_raw[-1] = 0.75
    env_raw = training_raw.clone()
    env_raw[-1] = 1.0 - 2.0 * training_raw[-1]
    expected = dataset._build_frame_wise_action(training_raw.numpy()[None])[0]
    torch.testing.assert_close(
        policy._normalize_executed_action(env_raw, gripper_mode="zero_one"), expected
    )


def test_batched_sessions_can_commit_atomically_and_reset_independently():
    policy = adapter()
    policy.generate([req(session="a"), req(session="b")], batch(2), lambda: {"action": [torch.ones(16, 10)] * 2})
    policy.reset("a")
    assert policy.memory.metadata()["steps"] == {"b": 0}
    assert policy.info()["enabled"] and policy.info()["sessions"] == 1


def recent_adapter(mode="auto", history_horizon=2):
    runtime = SimpleNamespace(
        encoder=LocalEvidenceEncoder(feature_config=CANONICAL_EVIDENCE_FEATURE_CONFIG),
        recurrent_backend=RecurrentLocalMemoryBackend(),
    )
    model = SimpleNamespace(
        config=SimpleNamespace(
            local_ttt_enabled=False,
            local_history_enabled=True,
            local_history_backend="recurrent",
            local_history_canonical_evidence=True,
            local_history_horizon=history_horizon,
        ),
        net=SimpleNamespace(local_history_runtime=runtime),
    )
    service = SimpleNamespace(
        model=model,
        action_normalization="quantile",
        action_min=torch.full((10,), -1.0),
        action_range=torch.full((10,), 2.0),
    )
    return PolicyLocalMemoryAdapter(service, mode=mode)


def test_recent_history_checkpoint_uses_bounded_online_control():
    policy = recent_adapter(history_horizon=2)
    assert policy.info()["memory_kind"] == "bounded_recent_history"
    assert policy.info()["history_horizon"] == 2
    policy.generate([req()], batch(), lambda: {"action": [torch.zeros(16, 10)]})

    observed = batch()
    result = policy.generate(
        [req(3)],
        observed,
        lambda: {"action": [observed["local_memory"][0].sum().expand(16, 10)]},
    )
    assert tuple(observed["local_memory"][0].shape) == (1, 32)
    assert result["_local_memory_status"][0]["memory_kind"] == "bounded_recent_history"
    assert policy.memory.metadata()["retained_source_steps"]["s"] == [1, 2]


def window_adapter(history_horizon=2):
    config = SimpleNamespace(
        history_mode="window",
        local_history_horizon=history_horizon,
        local_ttt_enabled=False,
        local_history_enabled=False,
        local_history_backend="recurrent",
        local_history_canonical_evidence=False,
    )
    model = SimpleNamespace(config=config, net=SimpleNamespace())
    service = SimpleNamespace(
        model=model,
        action_normalization="quantile",
        action_min=torch.full((10,), -1.0),
        action_range=torch.full((10,), 2.0),
        cfg=SimpleNamespace(action_chunk_size=16),
    )
    service._input_video_key = lambda: "video"
    service._prep_policy_item = lambda request: {
        "video_padded": torch.zeros(3, 17, 8, 8),
        "padded_image_size": torch.tensor([8, 8, 8, 8]),
    }
    return PolicyLocalMemoryAdapter(service, mode="required")


def window_req(step=2):
    rows = []
    for source_step in range(max(0, step - 2), step):
        rows.append(
            {
                "source_step": source_step,
                "image": f"frame-{source_step}",
                "executed_action": [0.0] * 7,
                "gripper_mode": "pm_one",
            }
        )
    return {
        "local_memory": {
            "session_id": "window-session",
            "episode_id": "window-episode",
            "consumer_step": step,
            "reset": step == 0,
            "evidence_version": "causal_visual96_executed_action10_v1",
            "evidence_format": "libero_rgb_action7_v1",
            "evidence": rows,
        }
    }


def test_native_window_prefixes_clean_history_and_returns_only_target_actions():
    policy = window_adapter(history_horizon=2)
    assert policy.info()["memory_kind"] == "native_window"
    assert policy.info()["history_horizon"] == 2
    plan = SimpleNamespace(
        has_local_memory=False,
        condition_frame_indexes_vision=[0],
        condition_frame_indexes_action=[],
        action_start_frame_offset=1,
    )
    observed = {
        "video": [[torch.zeros(3, 17, 8, 8)]],
        "action": [[torch.zeros(16, 64)]],
        "image_size": torch.tensor([[8, 8, 8, 8]]),
        "sequence_plan": [plan],
    }

    def generate():
        assert len(observed["video"][0]) == 3
        assert [item.shape[1] for item in observed["video"][0]] == [1, 1, 17]
        assert observed["action"][0][0].shape == (18, 64)
        assert len(observed["image_size"]) == 3
        assert plan.condition_frame_indexes_vision == [0]
        assert plan.condition_frame_indexes_action == [0, 1]
        assert plan.action_start_frame_offset == 1
        assert plan.vision_item_source_frame_offsets == [0, 1, 2]
        assert not plan.has_local_memory
        return {"action": [torch.arange(18 * 10, dtype=torch.float32).reshape(18, 10)]}

    result = policy.generate([window_req(2)], observed, generate)

    assert result["action"][0].shape == (16, 10)
    assert result["_local_memory_status"][0]["memory_kind"] == "native_window"
    assert result["_local_memory_status"][0]["retained_source_steps"] == [0, 1]


@pytest.mark.parametrize("kind", ["none", "ttt", "gru", "window"])
def test_profile_timing_is_exposed_for_every_history_method(kind):
    if kind == "none":
        policy = adapter("off")
        request = {"profile_inference": True}
        result = policy.generate([request], batch(), lambda: {"action": [torch.zeros(16, 10)]})
        profile = result["_profile_timing"][0]
        assert profile["history_mode"] == "none"
        return

    if kind == "ttt":
        policy = adapter()
        policy.generate([req()], batch(), lambda: {"action": [torch.zeros(16, 10)]})
        request = req(2)
        request["profile_inference"] = True
        result = policy.generate([request], batch(), lambda: {"action": [torch.zeros(16, 10)]})
        history = result["_profile_timing"][0]["history_ms"]
        assert result["_profile_timing"][0]["history_mode"] == "ttt"
        for key in (
            "request_build_total_ms",
            "prepare_total_ms",
            "evidence_encode_ms",
            "kqv_project_ms",
            "inner_update_read_ms",
            "state_detach_ms",
            "commit_ms",
        ):
            assert key in history and history[key] >= 0
        return

    if kind == "gru":
        policy = recent_adapter(history_horizon=2)
        policy.generate([req()], batch(), lambda: {"action": [torch.zeros(16, 10)]})
        request = req(2)
        request["profile_inference"] = True
        result = policy.generate([request], batch(), lambda: {"action": [torch.zeros(16, 10)]})
        history = result["_profile_timing"][0]["history_ms"]
        assert result["_profile_timing"][0]["history_mode"] == "gru"
        for key in ("request_build_total_ms", "prepare_total_ms", "evidence_encode_ms", "gru_replay_ms"):
            assert key in history and history[key] >= 0
        return

    policy = window_adapter(history_horizon=2)
    request = window_req(2)
    request["profile_inference"] = True
    plan = SimpleNamespace(
        has_local_memory=False,
        condition_frame_indexes_vision=[0],
        condition_frame_indexes_action=[],
        action_start_frame_offset=1,
    )
    observed = {
        "video": [[torch.zeros(3, 17, 8, 8)]],
        "action": [[torch.zeros(16, 64)]],
        "image_size": torch.tensor([[8, 8, 8, 8]]),
        "sequence_plan": [plan],
    }
    result = policy.generate(
        [request],
        observed,
        lambda: {"action": [torch.zeros(18, 10)]},
    )
    history = result["_profile_timing"][0]["history_ms"]
    assert result["_profile_timing"][0]["history_mode"] == "window"
    for key in ("payload_total_ms", "history_preprocess_ms", "action_normalize_ms", "prefix_assembly_ms"):
        assert key in history and history[key] >= 0


def test_zero_intervention_keeps_prefix_shape_but_zeros_injected_content_and_commits_real_state():
    policy = adapter("zero")
    policy.generate([req()], batch(), lambda: {"action": [torch.zeros(16, 10)]})
    initial_state = tuple(value.clone() for value in policy.memory._records["s"].state)

    observed = batch()

    def generate():
        token = observed["local_memory"][0]
        assert token is not None
        assert tuple(token.shape) == (1, 32)
        assert torch.count_nonzero(token).item() == 0
        assert observed["sequence_plan"][0].has_local_memory
        return {"action": [torch.zeros(16, 10)]}

    result = policy.generate([req(3)], observed, generate)
    record = policy.memory._records["s"]
    assert record.token is not None
    assert torch.count_nonzero(record.token).item() > 0
    assert any(
        not torch.equal(actual, initial)
        for actual, initial in zip(record.state, initial_state, strict=True)
    )
    assert result["_local_memory_status"][0]["intervention"] == "zero"
    assert result["_local_memory_status"][0]["prefix_present"]


def test_init_intervention_reads_real_evidence_but_keeps_fast_state_at_w0():
    policy = adapter("init")
    policy.generate([req()], batch(), lambda: {"action": [torch.zeros(16, 10)]})
    initial_state = tuple(value.clone() for value in policy.memory._records["s"].state)

    observed = batch()

    def generate():
        token = observed["local_memory"][0]
        assert token is not None
        assert tuple(token.shape) == (1, 32)
        assert torch.count_nonzero(token).item() > 0
        assert observed["sequence_plan"][0].has_local_memory
        return {"action": [torch.zeros(16, 10)]}

    result = policy.generate([req(3)], observed, generate)
    record = policy.memory._records["s"]
    for actual, expected in zip(record.state, initial_state, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert result["_local_memory_status"][0]["intervention"] == "init"
    assert result["_local_memory_status"][0]["prefix_present"]


def test_zero_and_init_are_ttt_only_interventions():
    with pytest.raises(ValueError, match="TTT fast-weight checkpoint"):
        recent_adapter(mode="zero")
    with pytest.raises(ValueError, match="TTT fast-weight checkpoint"):
        recent_adapter(mode="init")
