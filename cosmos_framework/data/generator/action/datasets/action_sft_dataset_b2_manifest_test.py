import pytest
import torch

from cosmos_framework.data.generator.action.datasets.action_sft_dataset import B2ManifestAwareIterableDataset


class _Dataset:
    def __getitem__(self, index: int):
        return {
            "task_index": torch.tensor(3),
            "episode_index": torch.tensor(11),
            "start_frame": torch.tensor(index),
        }


def _record(ordinal: int, start_frame: int | None = None):
    return {
        "ordinal": ordinal,
        "epoch": 0,
        "dataset_flat_index": ordinal,
        "task_index": 3,
        "episode_index": 11,
        "start_frame": ordinal if start_frame is None else start_frame,
    }


def test_b2_manifest_stream_preserves_requested_ordinals():
    records = [{**_record(ordinal), "suite": "libero_spatial"} for ordinal in (0, 1)]
    samples = list(B2ManifestAwareIterableDataset(_Dataset(), records, "libero_spatial"))
    assert [int(sample["b2_stream_ordinal"]) for sample in samples] == [0, 1]
    assert [int(sample["b2_stream_epoch"]) for sample in samples] == [0, 0]
    assert [int(sample["b2_dataset_flat_index"]) for sample in samples] == [0, 1]


def test_b2_manifest_stream_preserves_global_ordinals_for_one_suite():
    records = [{**_record(ordinal), "suite": "libero_spatial"} for ordinal in (3, 7)]
    samples = list(B2ManifestAwareIterableDataset(_Dataset(), records, "libero_spatial"))
    assert [int(sample["b2_stream_ordinal"]) for sample in samples] == [3, 7]


def test_b2_manifest_stream_rejects_duplicate_ordinals():
    with pytest.raises(ValueError, match="unique and strictly increasing"):
        B2ManifestAwareIterableDataset(_Dataset(), [{**_record(3), "suite": "libero_spatial"}, {**_record(3), "suite": "libero_spatial"}], "libero_spatial")


def test_b2_manifest_stream_rejects_identity_mismatch():
    with pytest.raises(ValueError, match="identity mismatch"):
        list(B2ManifestAwareIterableDataset(_Dataset(), [{**_record(0, start_frame=4), "suite": "libero_spatial"}], "libero_spatial"))


def test_b2_manifest_stream_rejects_wrong_suite():
    with pytest.raises(ValueError, match="must all belong"):
        B2ManifestAwareIterableDataset(_Dataset(), [{**_record(0), "suite": "libero_goal"}], "libero_spatial")
