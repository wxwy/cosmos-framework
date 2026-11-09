# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Opt-in B1 current-frame observer; no encoder, cache publisher or model loading.

The model calls the observer after its existing encoding and before latent crop.
An inactive observer is a no-op for all legacy inference and training. The trusted
C-profile caller supplies attested identities and owns the existing model executor;
successful scope exit creates only a private PolicyBoundary for the transaction.
"""

from __future__ import annotations

import struct
import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Callable

_CURRENT_CAPTURE: ContextVar[Any] = ContextVar("robocasa_policy_boundary_capture", default=None)


@contextmanager
def policy_current_encode_span(model):
    """Optional observer timing; default execution never touches CUDA events."""
    from cosmos_framework.inference.robocasa_feature_trace import get_trace

    trace = get_trace()
    if trace is None:
        yield
        return
    with trace.span("current_frame_vae", producer="policy"), trace.cuda_span("current_frame_vae", producer="policy"):
        yield


def observe_policy_current_encoding(model: Any, pixels: Any, latent: Any) -> None:
    """Observe one already-encoded tensor; never invoke an encoder or alter it."""
    capture = _CURRENT_CAPTURE.get()
    if capture is not None:
        from cosmos_framework.inference.robocasa_feature_trace import span

        with span("boundary_visual96_capture", producer="policy"):
            capture.observe(model, pixels, latent)


class _BoundaryCapture:
    def __init__(
        self,
        model: Any,
        *,
        source_rgb: bytes,
        identity: dict,
        lease: dict,
        request_key: str,
        chunk_id: str,
        source_step: int,
        image_size: int,
        deadline_ns: int,
        clock_domain: str,
        clock_ns: Callable[[], int],
    ) -> None:
        from cosmos_framework.inference.robocasa_vae_protocol import (
            RGB_BYTES,
            source_digest,
            validate_identity,
            validate_lease,
        )

        if type(source_rgb) is not bytes or len(source_rgb) != RGB_BYTES:
            raise ValueError("boundary source must be immutable original uint8 HWC RGB 256x512x3")
        if type(source_step) is not int or source_step < 0 or source_step % 16:
            raise ValueError("Stage3A boundary source_step must be a real nonnegative multiple of16")
        if type(deadline_ns) is not int or deadline_ns <= 0:
            raise ValueError("original absolute boundary deadline_ns required")
        for value in (request_key, chunk_id, clock_domain):
            if (
                type(value) is not str
                or not value
                or len(value) > 256
                or any(ord(character) < 32 for character in value)
            ):
                raise ValueError("bounded request/chunk/clock identity required")
        self._model = model
        self._rgb = source_rgb
        self._source_digest = source_digest(source_rgb)
        self._identity = validate_identity(identity)
        self._lease = validate_lease(lease)
        self._request_key, self._chunk_id, self._step = request_key, chunk_id, source_step
        self._image_size, self._deadline, self._clock_domain = image_size, deadline_ns, clock_domain
        self._clock = clock_ns
        self._entered = self._active = self._complete = False
        self._failure: BaseException | None = None
        self._events = 0
        self._visual: bytes | None = None
        self._boundary = None

    def _check_deadline(self) -> None:
        if self._clock() >= self._deadline:
            raise TimeoutError("original Policy boundary deadline exhausted")

    def _check_tokenizer(self) -> None:
        import torch

        tokenizer = getattr(self._model, "tokenizer_vision_gen", None)
        loaded = getattr(tokenizer, "model", None)
        module = getattr(loaded, "model", None)
        if (
            tokenizer is not self._tokenizer
            or loaded is not self._loaded
            or module is not self._module
            or getattr(tokenizer, "is_causal", None) is not True
            or getattr(tokenizer, "use_streaming_encode", None) is not False
            or getattr(tokenizer, "_keep_encoder_cache", None) is not False
            or getattr(tokenizer, "spatial_compression_factor", None) != 16
            or getattr(tokenizer, "temporal_compression_factor", None) != 4
            or getattr(loaded, "dtype", None) != torch.bfloat16
            or getattr(module, "training", None) is not False
            or not callable(getattr(module, "parameters", None))
        ):
            raise RuntimeError("boundary requires the same loaded frozen causal non-streaming Wan bf16 tokenizer")
        parameters = tuple(module.parameters())
        if not parameters or any(
            parameter.requires_grad or parameter.dtype != torch.bfloat16 for parameter in parameters
        ):
            raise RuntimeError("boundary requires frozen loaded Wan bf16 parameters")
        if self._parameter_ids != tuple(id(parameter) for parameter in parameters):
            raise RuntimeError("boundary model parameter identity changed during generation")
        aot = getattr(module, "_aot_chunk_fns", None)
        if self._identity["encoder_backend"] == "eager" and aot:
            raise RuntimeError("boundary eager identity cannot use installed AOT encoding")
        if self._identity["encoder_backend"] == "aot_b1":
            shape = self._prepared.padded_single_frame.shape
            key = (1, shape[-2] // 2, shape[-1] // 2, 0)
            if not isinstance(aot, dict) or key not in aot:
                raise RuntimeError("boundary aot_b1 identity requires its matching actual prime artifact")

    def __enter__(self):
        import numpy as np
        import torch

        from cosmos_framework.inference.robocasa_vae_encoder import (
            prepare_composite_frame,
            preprocess_profile_digest,
            visual96_contract_digest,
        )

        if self._entered:
            raise RuntimeError("Policy boundary capture cannot be reentered")
        self._entered = True
        if _CURRENT_CAPTURE.get() is not None:
            raise RuntimeError("nested Policy boundary capture is forbidden")
        self._check_deadline()
        if (
            self._identity["preprocess_profile_sha256"] != preprocess_profile_digest(self._image_size)
            or self._identity["visual96_contract_sha256"] != visual96_contract_digest()
        ):
            raise ValueError("boundary loaded preprocessing/Visual96 identity differs")
        array = np.frombuffer(self._rgb, dtype=np.uint8).reshape(256, 512, 3).copy()
        source = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        self._prepared = prepare_composite_frame(source, self._image_size)
        self._tokenizer = getattr(self._model, "tokenizer_vision_gen", None)
        self._loaded = getattr(self._tokenizer, "model", None)
        self._module = getattr(self._loaded, "model", None)
        parameters = getattr(self._module, "parameters", None)
        self._parameter_ids = tuple(id(parameter) for parameter in parameters()) if callable(parameters) else ()
        self._check_tokenizer()
        self._token = _CURRENT_CAPTURE.set(self)
        self._active = True
        return self

    def observe(self, model: Any, pixels: Any, latent: Any) -> None:
        import torch

        from cosmos_framework.inference.robocasa_vae_encoder import current_latent_to_visual96

        try:
            if not self._active or self._failure is not None:
                raise RuntimeError("Policy boundary capture is inactive or already failed")
            self._check_deadline()
            self._check_tokenizer()
            if model is not self._model or self._events:
                raise RuntimeError("Policy boundary requires exactly one encoding of its own B1 current frame")
            self._events += 1
            frame = self._prepared.padded_single_frame
            if (
                torch.is_grad_enabled()
                or not isinstance(pixels, torch.Tensor)
                or pixels.dtype != torch.float32
                or pixels.shape != (1, 3, 1, *frame.shape[-2:])
            ):
                raise ValueError("Policy boundary observer requires inference-only fp32 [1,3,1,H,W] input")
            expected = frame.unsqueeze(0).to(device=pixels.device, dtype=torch.float32)
            expected.div_(127.5).sub_(1.0)
            if not torch.equal(pixels, expected):
                raise ValueError("Policy boundary source RGB differs from the actually encoded current frame")
            if (
                not isinstance(latent, torch.Tensor)
                or latent.dtype != torch.float32
                or latent.device != pixels.device
                or latent.shape != (1, 48, 1, frame.shape[-2] // 16, frame.shape[-1] // 16)
                or not bool(torch.isfinite(latent).all())
            ):
                raise ValueError("Policy boundary requires the complete finite fp32 pre-crop current latent")
            visual = current_latent_to_visual96(latent[0, :, 0].detach()).detach().cpu()
            self._visual = struct.pack("<96f", *visual.tolist())
        except BaseException as error:
            self._visual = None
            self._failure = error
            raise

    def __exit__(self, exception_type, exception, traceback):
        self._active = False
        _CURRENT_CAPTURE.reset(self._token)
        if exception_type is not None:
            self._visual = None
            return False
        if self._failure is not None:
            raise RuntimeError("Policy boundary capture failed; candidate cannot be published") from self._failure
        if self._events != 1 or self._visual is None:
            raise RuntimeError("Policy boundary capture did not observe exactly one current encoding")
        self._check_deadline()
        self._check_tokenizer()
        from cosmos_framework.inference.robocasa_feature_evidence import PolicyBoundary

        self._boundary = PolicyBoundary.create(
            lease=self._lease,
            request_key=self._request_key,
            chunk_id=self._chunk_id,
            source_step=self._step,
            source_rgb_sha256=self._source_digest,
            identity=self._identity,
            deadline_ns=self._deadline,
            clock_domain=self._clock_domain,
            visual96=self._visual,
        )
        self._visual = None
        self._complete = True
        return False

    @property
    def boundary(self):
        if not self._complete:
            raise RuntimeError("Policy boundary is private/unavailable until successful scope exit")
        return self._boundary


def capture_policy_boundary(
    model: Any,
    *,
    source_rgb: bytes,
    identity: dict,
    lease: dict,
    request_key: str,
    chunk_id: str,
    source_step: int,
    deadline_ns: int,
    clock_domain: str,
    image_size: int = 256,
    clock_ns: Callable[[], int] = time.monotonic_ns,
) -> _BoundaryCapture:
    """Observe one existing C/B1 current encode; caller retains atomic commit ownership."""
    return _BoundaryCapture(
        model,
        source_rgb=source_rgb,
        identity=identity,
        lease=lease,
        request_key=request_key,
        chunk_id=chunk_id,
        source_step=source_step,
        image_size=image_size,
        deadline_ns=deadline_ns,
        clock_domain=clock_domain,
        clock_ns=clock_ns,
    )
