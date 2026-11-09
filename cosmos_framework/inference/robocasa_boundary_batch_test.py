"""Ordered native Policy members reuse their existing B1 encodes, never a B8 VAE."""

from copy import deepcopy

import pytest
import torch

from cosmos_framework.inference import robocasa_boundary_capture_test as original
from cosmos_framework.inference.robocasa_boundary_capture import capture_policy_boundaries

methods = original.methods


def contexts(count):
    return [
        dict(
            source_rgb=original._rgb(i),
            identity=original._identity(),
            lease=original._lease(i),
            request_key=f"request{i}",
            chunk_id=f"chunk{i}",
            source_step=16,
            deadline_ns=1000,
            clock_domain="cpu-clock",
            clock_ns=lambda: 100,
        )
        for i in range(count)
    ]


@pytest.mark.parametrize("count", range(1, 9))
def test_native_members_capture_identical_singleton_boundaries_without_extra_encoding(methods, count):
    model, calls = original._model(methods)
    expected = []
    with torch.no_grad():
        for i in range(count):
            with original._scope(model, variant=i, step=16) as single:
                original._policy_prefix(model, original._rgb(i))
            expected.append(single.boundary)
        calls.clear()
        with capture_policy_boundaries(model, contexts(count)) as group:
            with pytest.raises(RuntimeError, match="private"):
                _ = group.boundaries
            for i in range(count):
                original._policy_prefix(model, original._rgb(i))
        assert len(calls) == count
        assert all(c.shape[:3] == (1, 3, 1) for c in calls)
    for i, (single, batched) in enumerate(zip(expected, group.boundaries, strict=True)):
        assert batched.visual96 == single.visual96
        assert batched.header["lease"] == original._lease(i)
        assert batched.header["policy_request_key"] == f"request{i}"


@pytest.mark.parametrize("fault", ["missing", "extra", "swapped", "duplicate", "deadline", "generation"])
def test_one_bad_member_prevents_all_boundary_publication_and_restores_context(methods, fault):
    model, calls = original._model(methods)
    entries = contexts(2)
    if fault == "duplicate":
        entries[1]["lease"] = deepcopy(entries[0]["lease"])
        with pytest.raises(ValueError, match="duplicate"):
            capture_policy_boundaries(model, entries)
        return
    if fault == "deadline":
        entries[1]["clock_ns"] = lambda: 1001
    capture = capture_policy_boundaries(model, entries)
    with torch.no_grad(), pytest.raises((ValueError, RuntimeError, TimeoutError)):
        with capture:
            original._policy_prefix(model, original._rgb(1 if fault == "swapped" else 0))
            if fault != "missing":
                original._policy_prefix(model, original._rgb(1))
            if fault == "extra":
                original._policy_prefix(model)
            if fault == "generation":
                raise RuntimeError("native generation failed")
    with pytest.raises(RuntimeError, match="private"):
        _ = capture.boundaries
    with torch.no_grad(), original._scope(model) as single:
        original._policy_prefix(model)
    assert single.boundary.header["lease"] == original._lease()
