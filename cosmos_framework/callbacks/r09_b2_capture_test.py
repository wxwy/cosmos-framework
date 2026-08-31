import pytest
import torch

import cosmos_framework.callbacks.r09_b2_capture as capture_module
from cosmos_framework.callbacks.r09_b2_capture import R09B2NonMutatingCaptureCallback, clone_local_payload, isolated_rng, require_unchanged, snapshot_hash, take_isolation_snapshot


def _protected_values():
    return {
        "parameters": {"weight": torch.tensor([1.0])}, "buffers": {"mean": torch.tensor([1.0])},
        "optimizer": {"step": torch.tensor([1])}, "scheduler": {"epoch": 1},
        "batch_metadata": {"ordinal": 3, "epoch": 0, "microbatch": 1},
        "recurrent_state": {"hidden": torch.tensor([1.0])},
        "ttt_state": {name: torch.tensor([1.0]) for name in ("W", "pending_evidence", "last_evidence", "initialized", "segment_progress")},
    }


def _provider(values):
    return lambda: take_isolation_snapshot(**values)


def test_capture_payload_modes_do_not_mutate_source():
    source = [torch.tensor([1.0]), torch.tensor([2.0])]
    assert clone_local_payload(source, "normal")[0] is not source[0]
    assert [item.item() for item in clone_local_payload(source, "zero")] == [0.0, 0.0]
    assert [item.item() for item in clone_local_payload(source, "shuffle")] == [2.0, 1.0]
    assert [item.item() for item in source] == [1.0, 2.0]


def test_capture_rng_is_restored_after_exception():
    before = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError):
        with isolated_rng():
            torch.rand(3)
            raise RuntimeError("capture failure")
    assert torch.equal(torch.get_rng_state(), before)


def test_capture_rejects_invalid_mode_or_payload():
    with pytest.raises(ValueError):
        clone_local_payload([], "bad")
    with pytest.raises(TypeError):
        clone_local_payload([None], "normal")


def test_snapshot_hash_detects_tensor_and_metadata_mutation():
    state = {"weight": torch.tensor([1.0]), "ordinal": [0, 1]}
    before = snapshot_hash(state)
    state["weight"].add_(1)
    assert snapshot_hash(state) != before
    state["weight"].sub_(1)
    state["ordinal"][1] = 2
    assert snapshot_hash(state) != before


def test_callback_never_accesses_canonical_model():
    class Poison:
        def __getattr__(self, name):
            raise AssertionError(f"canonical model accessed: {name}")

    callback = R09B2NonMutatingCaptureCallback("zero", _provider(_protected_values()))
    source = torch.tensor([3.0])
    callback.on_training_step_end(Poison(), {"local_memory": [source], "b2_stream_ordinal": torch.tensor(4), "b2_stream_epoch": torch.tensor(0), "b2_stream_microbatch": torch.tensor(2)}, {}, torch.tensor(0.0))
    assert callback.last_capture[0].item() == 0.0
    assert source.item() == 3.0
    assert callback.last_ordinals == (4, 0, 2)


def test_callback_rejects_mutation_in_real_capture_path(monkeypatch):
    values = _protected_values()
    callback = R09B2NonMutatingCaptureCallback("normal", _provider(values))

    def mutate_then_clone(payload, mode):
        values["parameters"]["weight"].add_(1)
        return clone_local_payload(payload, mode)

    monkeypatch.setattr(capture_module, "clone_local_payload", mutate_then_clone)
    batch = {"local_memory": [torch.tensor([3.0])], "b2_stream_ordinal": torch.tensor(4), "b2_stream_epoch": torch.tensor(0), "b2_stream_microbatch": torch.tensor(2)}
    with pytest.raises(RuntimeError, match="parameters"):
        callback.on_training_step_end(object(), batch, {}, torch.tensor(0.0))


def test_callback_rejects_missing_microbatch_metadata():
    callback = R09B2NonMutatingCaptureCallback("normal", _provider(_protected_values()))
    with pytest.raises(ValueError, match="b2_stream_microbatch"):
        callback.on_training_step_end(object(), {"local_memory": [], "b2_stream_ordinal": torch.tensor(4), "b2_stream_epoch": torch.tensor(0)}, {}, torch.tensor(0.0))


@pytest.mark.parametrize("field", ["parameters", "buffers", "optimizer", "scheduler", "batch_metadata", "recurrent_state", "ttt_state"])
def test_isolation_snapshot_rejects_each_protected_mutation(field):
    values = _protected_values()
    before = take_isolation_snapshot(**values)
    if isinstance(values[field], dict):
        first = next(iter(values[field]))
        value = values[field][first]
        values[field][first] = value + 1 if isinstance(value, torch.Tensor) else value + 1
    after = take_isolation_snapshot(**values)
    with pytest.raises(RuntimeError, match="protected state"):
        require_unchanged(before, after)


def test_snapshot_rejects_noncanonical_ttt_schema():
    values = _protected_values()
    values["ttt_state"] = {"W": torch.tensor([1.0])}
    with pytest.raises(ValueError, match="TTT state"):
        take_isolation_snapshot(**values)
