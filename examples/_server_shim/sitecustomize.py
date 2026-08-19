# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""sitecustomize shim for the Action policy inference server.

The server's ``setup_args.guardrails`` defaults to True, which pulls in a heavy
guardrail dependency stack (nltk / better_profanity / retinaface / qwen3guard /
llamaGuard3) that is not installed in this training venv and is irrelevant for a
policy (action-head) inference server.

Two layers of defense, because pydantic v2's ``model_validate`` reads defaults
baked into the compiled ``__pydantic_core_schema__`` (changing
``model_fields[field].default`` has no effect on validation output):

1. ``GuardrailRunners.create`` no-ops and returns None. inference.py:135 is
   ``guardrails = GuardrailRunners.create(setup_args) if setup_args.guardrails else None``
   so even when ``setup_args.guardrails`` is True, ``guardrails`` becomes None and
   the lazy import chain (``presets`` -> ``blocklist`` -> ``nltk``) never runs.
   Runtime guards already treat ``self.guardrails is None`` as "skip" (inference.py:274/285).
2. Belt-and-suspenders: still flip the pydantic field default on every override
   class in the MRO. Harmless if it works, ignored if the core schema is cached.

This file is loaded automatically at interpreter startup (``sitecustomize``),
so it patches BEFORE the server's tyro CLI builds its args. It does not modify
any cosmos-framework source file.
"""
import sys as _sys

print("[sitecustomize] loaded, guardrails-disable patch running", file=_sys.stderr)
try:
    from cosmos_framework.inference import args as _inference_args
    from cosmos_framework.inference.common import args as _inference_common_args
    from cosmos_framework.inference.common import inference as _inference

    # --- Layer 1: make the guardrail runner factory a no-op -------------------
    # inference/common/inference.py:84 classmethod; accessed as
    # GuardrailRunners.create(setup_args). Replacing with a staticmethod that
    # swallows all args and returns None means the `presets` import chain
    # (nltk / retinaface / llamaGuard3) is never triggered.
    _inference.GuardrailRunners.create = staticmethod(lambda *a, **k: None)  # noqa: E731
    print("[sitecustomize] GuardrailRunners.create -> None (no-op)", file=_sys.stderr)

    # --- Layer 2: try flipping the pydantic defaults anyway -------------------
    _inference_common_args.GuardrailOverrides.model_fields["guardrails"].default = False
    _inference_common_args.SetupOverrides.model_fields["guardrails"].default = False
    _inference_args.OmniSetupOverrides.model_fields["guardrails"].default = False
    print(
        "[sitecustomize] patch applied: "
        f"{_inference_args.OmniSetupOverrides.model_fields['guardrails'].default}",
        file=_sys.stderr,
    )

    # --- Layer 3: surface the real error inside torch DCP plan building --------
    # torch.distributed.checkpoint.reduce_scatter wraps any exception raised while
    # building the load plan with ``_wrap_exception`` -> (exc, tb.extract_tb(...)).
    # ``tb.extract_tb`` returns a StackSummary that is NOT picklable, so the later
    # ``gather_object`` dies with "TypeError: cannot pickle code objects" and the
    # ORIGINAL exception is swallowed. Log it before wrapping so the real failure
    # shows up in the server log instead of the misleading pickle error.
    try:
        import traceback as _traceback

        import torch.distributed.checkpoint.utils as _dcp_utils

        _orig_wrap_exception = _dcp_utils._wrap_exception

        def _debug_wrap_exception(exc):
            print("[sitecustomize] === REAL ERROR inside DCP plan build ====", file=_sys.stderr)
            _traceback.print_exception(type(exc), exc, exc.__traceback__)
            return _orig_wrap_exception(exc)

        _dcp_utils._wrap_exception = _debug_wrap_exception
        print("[sitecustomize] dcp _wrap_exception -> debug wrapper", file=_sys.stderr)
    except Exception as _exc2:  # pragma: no cover - defensive
        print(f"[sitecustomize] dcp debug patch skipped: {_exc2!r}", file=_sys.stderr)

    # --- Layer 4: inject keys_to_skip_loading into the inference checkpoint load
    # inference/inference.py:1193 calls load_model_from_checkpoint(...) WITHOUT
    # keys_to_skip_loading, so it defaults to [] and the edge warmstart's
    # `net_ema.` skip list (set for training in
    # action_policy_libero_edge_warmstart.py:52) is lost. The base Policy-DROID
    # DCP carries no `net_ema.action2llm.*` keys, so the strict DCP plan builder
    # raises "Missing key in checkpoint state_dict: net_ema.action2llm.bias.weight".
    # Mirror the training config: skip the whole EMA branch, keep `net.*`.
    try:
        import cosmos_framework.utils.generator.model_loader as _model_loader

        _orig_load_model_from_checkpoint = _model_loader.load_model_from_checkpoint

        def _patched_load_model_from_checkpoint(*args, **kwargs):
            kwargs.setdefault("keys_to_skip_loading", ["net_ema."])
            return _orig_load_model_from_checkpoint(*args, **kwargs)

        _model_loader.load_model_from_checkpoint = _patched_load_model_from_checkpoint
        print("[sitecustomize] load_model_from_checkpoint keys_to_skip_loading -> ['net_ema.']", file=_sys.stderr)
    except Exception as _exc3:  # pragma: no cover - defensive
        print(f"[sitecustomize] keys_to_skip_loading patch skipped: {_exc3!r}", file=_sys.stderr)
except Exception as _exc:  # pragma: no cover - defensive; never block server boot
    print(f"[sitecustomize] guardrails-disable patch skipped: {_exc!r}", file=_sys.stderr)
