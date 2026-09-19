# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from cosmos_framework.inference.args import OmniSetupArgs
from cosmos_framework.scripts.action_policy_server_utils import apply_model_compile_setting


def _setup_args(*, use_torch_compile: bool) -> OmniSetupArgs:
    return OmniSetupArgs.model_construct(use_torch_compile=use_torch_compile)


def test_apply_model_compile_setting_honors_explicit_false() -> None:
    setup_args = _setup_args(use_torch_compile=True)

    resolved = apply_model_compile_setting(
        setup_args,
        {"model": {"config": {"compile": {"enabled": False}}}},
    )

    assert resolved is False
    assert setup_args.use_torch_compile is False


def test_apply_model_compile_setting_honors_explicit_true() -> None:
    setup_args = _setup_args(use_torch_compile=False)

    resolved = apply_model_compile_setting(
        setup_args,
        {"model": {"config": {"compile": {"enabled": True}}}},
    )

    assert resolved is True
    assert setup_args.use_torch_compile is True


def test_apply_model_compile_setting_leaves_default_when_missing() -> None:
    setup_args = _setup_args(use_torch_compile=True)

    resolved = apply_model_compile_setting(setup_args, {"model": {"config": {}}})

    assert resolved is None
    assert setup_args.use_torch_compile is True
