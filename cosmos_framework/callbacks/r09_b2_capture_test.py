import pytest
import torch

from cosmos_framework.callbacks.r09_b2_capture import R09B2NonMutatingCaptureCallback, clone_local_payload, isolated_rng, snapshot_hash


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

    callback = R09B2NonMutatingCaptureCallback("zero")
    source = torch.tensor([3.0])
    callback.on_training_step_end(Poison(), {"local_memory": [source], "b2_stream_ordinal": torch.tensor(4)}, {}, torch.tensor(0.0))
    assert callback.last_capture[0].item() == 0.0
    assert source.item() == 3.0
