"""B0 core 的 CPU 数值与参数合同。"""

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from cosmos_framework.model.generator.mot.cosmos3_vfm_network import Cosmos3VFMNetwork
from cosmos_framework.model.generator.mot.local_evidence import (
    ContinualTTTFastState,
    ContinualTTTLocalMemoryCore,
    EvidenceFeatureConfig,
    LocalEvidenceEncoder,
)
from cosmos_framework.model.generator.mot.memory_prefix import LocalMemoryRuntime


def test_frozen_configuration_and_parameter_inventory():
    encoder, core = LocalEvidenceEncoder(), ContinualTTTLocalMemoryCore()
    assert (core.evidence_dim, core.ttt_dim, core.fast_hidden_dim, core.local_dim) == (256, 64, 256, 32)
    assert (core.k_local, core.inner_lr, core.ttt_tbptt_steps) == (4, 0.1, 16)
    assert encoder.action_proj.in_features == 15
    assert encoder.visual_proj.in_features == 96
    assert encoder.feature_config == EvidenceFeatureConfig(False, False, False)
    assert not any(key.startswith(("state", "dt", "age")) for key in encoder.state_dict())
    parameters = dict(core.named_parameters())
    assert "slot_queries" in parameters
    assert all("w0_" + field in parameters for field in ContinualTTTFastState._fields)
    for value in core.initial_state(2):
        assert value.dtype == torch.float32
        assert not isinstance(value, torch.nn.Parameter)
        assert all(value is not parameter for parameter in parameters.values())
    assert not any(key.startswith("fast_") for key in core.state_dict())


@pytest.mark.parametrize("feature", ["state", "dt", "age"])
def test_legacy_features_rejected(feature):
    with pytest.raises(ValueError):
        LocalEvidenceEncoder(feature_config=EvidenceFeatureConfig(**{feature: True}))


def test_k4_scan_mask_and_slow_gradients():
    torch.manual_seed(1)
    encoder, core = LocalEvidenceEncoder(), ContinualTTTLocalMemoryCore()
    visual, action = torch.randn(2, 16, 96), torch.randn(2, 16, 15)
    valid = torch.ones(2, 16, dtype=torch.bool)
    valid[:, 0] = False
    visual[:, 0] = float("nan")
    action[:, 0] = float("nan")
    tokens, state, present = core.scan_segment_masked_encoded_many(encoder, visual, action, valid)
    assert tokens.shape == (2, 16, 4, 32)
    assert torch.equal(present, valid)
    assert torch.count_nonzero(tokens[:, 0]) == 0
    assert torch.isfinite(tokens).all()
    assert all(value.dtype == torch.float32 for value in state)
    # 合成 outer 标量仅验证梯度图，不接 Cosmos objective。
    tokens.square().sum().backward()
    for parameter in (*encoder.parameters(), *core.parameters()):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    assert torch.count_nonzero(core.slot_queries.grad) > 0
    assert torch.count_nonzero(core.w0_fast_in_weight.grad) > 0


def test_batched_update_matches_independent_rows_without_lr_dilution():
    torch.manual_seed(2)
    core = ContinualTTTLocalMemoryCore()
    evidence = torch.randn(3, 256)
    initial = core.initial_state(3)
    before = tuple(value.detach().clone() for value in initial)
    tokens, updated = core.step_many(evidence, initial)
    expected_tokens, expected_states = [], []
    for row in range(3):
        single = ContinualTTTFastState(*(value[row : row + 1] for value in initial))
        token, state = core.step_many(evidence[row : row + 1], single)
        expected_tokens.append(token)
        expected_states.append(state)
    torch.testing.assert_close(tokens, torch.cat(expected_tokens), atol=2e-6, rtol=2e-5)
    for index, value in enumerate(updated):
        torch.testing.assert_close(value, torch.cat([state[index] for state in expected_states]), atol=2e-6, rtol=2e-5)
        assert torch.equal(initial[index], before[index])
    # 独立 autograd 公式作为 lr 基准，避免仅比较同一实现的两条分支。
    work = ContinualTTTFastState(*(value[0].detach().requires_grad_(True) for value in initial))
    key, target = core.key_proj(evidence[0]), core.value_proj(evidence[0])
    hidden = torch.nn.functional.silu(torch.nn.functional.linear(key, work[0], work[1]))
    prediction = torch.nn.functional.linear(hidden, work[2], work[3])
    gradients = torch.autograd.grad((prediction - target).square().mean(), work)
    for value, reference, gradient in zip(updated, work, gradients, strict=True):
        torch.testing.assert_close(value[0], reference - 0.1 * gradient, atol=2e-6, rtol=2e-5)


def test_fast_state_remains_fp32_with_bfloat16_slow_parameters():
    core = ContinualTTTLocalMemoryCore().to(dtype=torch.bfloat16)
    state = core.initial_state(2)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        tokens, candidate = core.step_many(torch.randn(2, 256), state)
    assert tokens.dtype == torch.float32
    assert all(value.dtype == torch.float32 for value in candidate)


def test_invalid_state_and_action_dimension_rejected():
    encoder, core = LocalEvidenceEncoder(), ContinualTTTLocalMemoryCore()
    with pytest.raises(ValueError):
        encoder.encode_segment(torch.zeros(1, 2, 96), torch.zeros(1, 2, 64))
    with pytest.raises(ValueError):
        core.validate_state(ContinualTTTFastState(*(value.double() for value in core.initial_state(1))), 1)
    with torch.no_grad(), pytest.raises(RuntimeError):
        core.step_many(torch.zeros(1, 256), core.initial_state(1))


def test_ttt_telemetry_is_detached_and_drains_per_window():
    torch.manual_seed(7)
    core = ContinualTTTLocalMemoryCore()
    evidence = torch.randn(2, 256)
    state = core.initial_state(2)
    core.reset_telemetry()
    core.step_many(evidence, state)
    report = core.drain_telemetry()
    assert set(report) == {
        "ttt_inner_loss_sum",
        "ttt_inner_loss_max",
        "ttt_inner_loss_count",
        "ttt_fast_state_norm_sum",
        "ttt_fast_state_norm_max",
        "ttt_fast_state_norm_count",
        "ttt_fast_update_norm_sum",
        "ttt_fast_update_norm_max",
        "ttt_fast_update_norm_count",
    }
    assert report["ttt_inner_loss_count"].item() == 2
    assert report["ttt_inner_loss_sum"].ndim == 0
    assert report["ttt_fast_state_norm_sum"].ndim == 0
    assert report["ttt_fast_update_norm_sum"].ndim == 0
    assert all(not value.requires_grad for value in report.values())
    assert core.drain_telemetry() == {}


def test_k1_zero_init_and_k4_normal_init():
    torch.manual_seed(11)
    assert torch.count_nonzero(ContinualTTTLocalMemoryCore(k_local=1).slot_queries) == 0
    assert torch.count_nonzero(ContinualTTTLocalMemoryCore(k_local=4).slot_queries) > 0


def test_b1_telemetry_matches_legacy_scalar_values():
    torch.manual_seed(19)
    core = ContinualTTTLocalMemoryCore(evidence_dim=8, local_dim=4, ttt_dim=6, fast_hidden_dim=8)
    evidence = torch.randn(1, 8)
    state = core.initial_state(1)
    key, target = core.key_proj(evidence), core.value_proj(evidence)
    prediction = core._fast_mlp(key[:, None], state).squeeze(1)
    expected_inner = (prediction - target).square().mean().detach()
    _, updated = core.step_many(evidence, state)
    expected_state = sum(value.detach().square().sum() for value in updated).sqrt()
    expected_update = sum(
        (new.detach() - old.detach()).square().sum() for old, new in zip(state, updated, strict=True)
    ).sqrt()
    report = core.drain_telemetry()
    for name, expected in (
        ("inner_loss", expected_inner),
        ("fast_state_norm", expected_state),
        ("fast_update_norm", expected_update),
    ):
        torch.testing.assert_close(report[f"ttt_{name}_sum"], expected)
        torch.testing.assert_close(report[f"ttt_{name}_max"], expected)
        assert report[f"ttt_{name}_count"].item() == 1


@pytest.mark.parametrize("k_local", [1, 4])
def test_batched_scan_matches_scalar_rows_and_telemetry(k_local):
    torch.manual_seed(23)
    encoder = LocalEvidenceEncoder(evidence_dim=8)
    core = ContinualTTTLocalMemoryCore(
        evidence_dim=8, local_dim=4, ttt_dim=6, fast_hidden_dim=8, ttt_tbptt_steps=4, k_local=k_local
    )
    scalar_encoder, scalar_core = copy.deepcopy(encoder), copy.deepcopy(core)
    visual, action = torch.randn(4, 4, 96), torch.randn(4, 4, 15)
    valid = torch.tensor(
        [[False, True, True, True], [True, True, False, False], [False, True, False, False], [False] * 4]
    )
    tokens, state, present = core.scan_segment_masked_encoded_many(encoder, visual, action, valid)
    loss = tokens.square().sum() + sum(value.square().sum() for value in state) * 0.01
    loss.backward()
    scalar_tokens, scalar_states, scalar_present = [], [], []
    scalar_loss = 0
    for row in range(4):
        row_tokens, row_state, row_present = scalar_core.scan_segment_masked_encoded_many(
            scalar_encoder, visual[row : row + 1], action[row : row + 1], valid[row : row + 1]
        )
        scalar_tokens.append(row_tokens)
        scalar_states.append(row_state)
        scalar_present.append(row_present)
        scalar_loss = scalar_loss + row_tokens.square().sum() + sum(value.square().sum() for value in row_state) * 0.01
    scalar_loss.backward()
    torch.testing.assert_close(tokens, torch.cat(scalar_tokens), atol=2e-6, rtol=2e-5)
    assert torch.equal(present, torch.cat(scalar_present))
    for field, value in enumerate(state):
        torch.testing.assert_close(
            value, torch.cat([row_state[field] for row_state in scalar_states]), atol=2e-6, rtol=2e-5
        )
    for (_, parameter), (_, scalar_parameter) in zip(
        (*encoder.named_parameters(), *core.named_parameters()),
        (*scalar_encoder.named_parameters(), *scalar_core.named_parameters()),
        strict=True,
    ):
        torch.testing.assert_close(parameter.grad, scalar_parameter.grad, atol=2e-5, rtol=2e-4)
    for parameter in (
        encoder.visual_proj.weight,
        encoder.action_proj.weight,
        core.key_proj.weight,
        core.query_proj.weight,
        core.value_proj.weight,
        core.slot_queries,
        core.w0_fast_in_weight,
    ):
        assert parameter.grad is not None and bool(torch.count_nonzero(parameter.grad))
    batched_report, scalar_report = core.drain_telemetry(), scalar_core.drain_telemetry()
    for name in batched_report:
        torch.testing.assert_close(batched_report[name], scalar_report[name], atol=2e-5, rtol=2e-4)


def _local_net():
    net = nn.Module()
    net.config = SimpleNamespace(local_memory_enabled=True)
    net.local_memory_runtime = LocalMemoryRuntime(
        evidence_dim=8, local_dim=4, ttt_dim=6, fast_hidden_dim=8, ttt_tbptt_steps=4, k_local=1
    )
    return net


def _model_scan(net, visual, action, valid, state, *, continuation_mask=None):
    return Cosmos3VFMNetwork.scan_local_memory(net, visual, action, valid, state, continuation_mask=continuation_mask)


def test_model_owned_all_fresh_matches_core_default_state():
    torch.manual_seed(31)
    net = _local_net()
    reference = copy.deepcopy(net)
    visual, action, valid = torch.randn(2, 4, 96), torch.randn(2, 4, 15), torch.ones(2, 4, dtype=torch.bool)
    actual = _model_scan(net, visual, action, valid, None)
    expected = reference.local_memory_runtime.core.scan_segment_masked_encoded_many(
        reference.local_memory_runtime.encoder, visual, action, valid, None
    )
    torch.testing.assert_close(actual[0], expected[0], atol=0, rtol=0)
    assert torch.equal(actual[2], expected[2])
    for state, reference_state in zip(actual[1], expected[1], strict=True):
        torch.testing.assert_close(state, reference_state, atol=0, rtol=0)


def test_model_owned_mixed_state_matches_scalar_and_ignores_fresh_placeholder(monkeypatch):
    torch.manual_seed(37)
    base = _local_net()
    mixed, scalar, changed = (copy.deepcopy(base) for _ in range(3))
    visual, action = torch.randn(3, 4, 96), torch.randn(3, 4, 15)
    valid = torch.ones(3, 4, dtype=torch.bool)
    continued = ContinualTTTLocalMemoryCore.detach_state(base.local_memory_runtime.core.initial_state(1))
    continued = ContinualTTTFastState(*(value + 0.1 for value in continued))
    mask = torch.tensor([True, False, False])

    def state_with_placeholder(value):
        return ContinualTTTFastState(
            *(torch.cat((field, torch.full_like(field, value), torch.full_like(field, value))) for field in continued)
        )

    calls = []
    original = mixed.local_memory_runtime.core.initial_state

    def initial_state(batch):
        calls.append(batch)
        return original(batch)

    monkeypatch.setattr(mixed.local_memory_runtime.core, "initial_state", initial_state)
    tokens, state, present = _model_scan(
        mixed, visual, action, valid, state_with_placeholder(0.0), continuation_mask=mask
    )
    assert calls == [3] and torch.equal(present, valid)
    (tokens.square().sum() + sum(value.square().sum() for value in state) * 0.01).backward()
    scalar_tokens, scalar_states = [], []
    for row in range(3):
        row_tokens, row_state, _ = _model_scan(
            scalar,
            visual[row : row + 1],
            action[row : row + 1],
            valid[row : row + 1],
            continued if row == 0 else None,
        )
        scalar_tokens.append(row_tokens)
        scalar_states.append(row_state)
    scalar_loss = sum(
        row_tokens.square().sum() + sum(value.square().sum() for value in row_state) * 0.01
        for row_tokens, row_state in zip(scalar_tokens, scalar_states, strict=True)
    )
    scalar_loss.backward()
    torch.testing.assert_close(tokens, torch.cat(scalar_tokens), atol=2e-6, rtol=2e-5)
    for index, value in enumerate(state):
        torch.testing.assert_close(
            value, torch.cat([row_state[index] for row_state in scalar_states]), atol=2e-6, rtol=2e-5
        )
    for (_, parameter), (_, reference) in zip(mixed.named_parameters(), scalar.named_parameters(), strict=True):
        if reference.grad is not None:
            torch.testing.assert_close(parameter.grad, reference.grad, atol=2e-5, rtol=2e-4)
    assert torch.count_nonzero(mixed.local_memory_runtime.core.w0_fast_in_weight.grad)
    for field in ContinualTTTFastState._fields:
        torch.testing.assert_close(
            getattr(mixed.local_memory_runtime.core, "w0_" + field).grad,
            getattr(scalar.local_memory_runtime.core, "w0_" + field).grad,
            atol=2e-5,
            rtol=2e-4,
        )

    other_tokens, other_state, _ = _model_scan(
        changed, visual, action, valid, state_with_placeholder(9.0), continuation_mask=mask
    )
    (other_tokens.square().sum() + sum(value.square().sum() for value in other_state) * 0.01).backward()
    torch.testing.assert_close(other_tokens, tokens, atol=0, rtol=0)
    for (_, parameter), (_, reference) in zip(changed.named_parameters(), mixed.named_parameters(), strict=True):
        if reference.grad is not None:
            torch.testing.assert_close(parameter.grad, reference.grad, atol=0, rtol=0)


def test_model_owned_all_continuation_never_initializes_w0(monkeypatch):
    net = _local_net()
    visual, action, valid = torch.randn(2, 4, 96), torch.randn(2, 4, 15), torch.ones(2, 4, dtype=torch.bool)
    state = ContinualTTTLocalMemoryCore.detach_state(net.local_memory_runtime.core.initial_state(2))

    def forbidden(batch):
        raise AssertionError("all-continuation 不得读取 W0")

    monkeypatch.setattr(net.local_memory_runtime.core, "initial_state", forbidden)
    tokens, _, _ = _model_scan(net, visual, action, valid, state)
    tokens.square().sum().backward()
    assert all(
        getattr(net.local_memory_runtime.core, "w0_" + field).grad is None for field in ContinualTTTFastState._fields
    )
