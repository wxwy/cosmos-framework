"""B0 core 的 CPU 数值与参数合同。"""

import pytest
import torch

from cosmos_framework.model.generator.mot.local_evidence import (
    ContinualTTTFastState,
    ContinualTTTLocalMemoryCore,
    EvidenceFeatureConfig,
    LocalEvidenceEncoder,
)


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
