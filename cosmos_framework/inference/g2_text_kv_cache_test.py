import pytest
import torch

from cosmos_framework.model.generator.mot.inference_text_kv_memory import request_text_kv_caches


def test_new_request_seeds_only_text_KV_and_owns_storage_without_Local_references():
    source = tuple((torch.randn(1, 3, 2, 4), torch.randn(1, 3, 2, 4)) for _ in range(2))
    with torch.no_grad():
        first = request_text_kv_caches(source)
        second = request_text_kv_caches(source)
    for a, b, (key, value) in zip(first, second, source, strict=True):
        assert torch.equal(a.get()[0], key) and torch.equal(a.get()[1], value)
        assert a.get()[0].data_ptr() != key.data_ptr() != b.get()[0].data_ptr()
        assert a.prefix_key_input is a.prefix_key_output is a.prefix_value_output is None
        a.clear()
        assert b.is_initialized
        assert torch.equal(b.get()[0], key)
        b.clear()
        assert not b.is_initialized


@pytest.mark.parametrize("bad", ["shape", "dtype", "batch", "gradient"])
def test_bad_prepared_KV_rejected(bad):
    key = torch.ones(1, 3, 2, 4)
    value = key.clone()
    if bad == "shape":
        value = value[:, :1]
    elif bad == "dtype":
        value = value.double()
    elif bad == "batch":
        key, value = key.repeat(2, 1, 1, 1), value.repeat(2, 1, 1, 1)
    elif bad == "gradient":
        key.requires_grad_(True)
    with torch.no_grad(), pytest.raises(ValueError, match="fixed"):
        request_text_kv_caches(((key, value),))


def test_no_implicit_training_cache_or_raw_reasoner_conversion():
    with pytest.raises(ValueError, match="inference only"):
        request_text_kv_caches(((torch.ones(1, 3, 2, 4), torch.ones(1, 3, 2, 4)),))
    # Raw reasoner [S,H,D] is not the normalized/positioned [1,S,H,D] GEN ABI.
    with torch.no_grad(), pytest.raises(ValueError, match="GEN text"):
        request_text_kv_caches(((torch.ones(3, 2, 4), torch.ones(3, 2, 4)),))
