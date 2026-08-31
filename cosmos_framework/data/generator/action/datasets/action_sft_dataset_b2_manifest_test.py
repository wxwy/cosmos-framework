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
        "dataset_flat_index": ordinal,
        "task_index": 3,
        "episode_index": 11,
        "start_frame": ordinal if start_frame is None else start_frame,
    }


def test_b2_manifest_stream_preserves_requested_ordinals():
    samples = list(B2ManifestAwareIterableDataset(_Dataset(), [_record(0), _record(1)]))
    assert [int(sample["b2_stream_ordinal"]) for sample in samples] == [0, 1]
    assert [int(sample["b2_dataset_flat_index"]) for sample in samples] == [0, 1]


def test_b2_manifest_stream_rejects_identity_mismatch():
    with pytest.raises(ValueError, match="identity mismatch"):
        list(B2ManifestAwareIterableDataset(_Dataset(), [_record(0, start_frame=4)]))
