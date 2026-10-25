"""Truthful compatibility snapshots for new Corrected Phase5 jobs only.

The retained Stage-A dataloader is not the grouped trainer's sample producer.
Its action geometry is mirrored solely for generic config/inference consumers.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

_DATASET_PATH = ("dataloader", "datasets", "robocasa", "dataset")
_CHUNK_KEYS = ("chunk_length", "num_action_per_chunk")
_DATASET_KEYS = (
    "fps",
    "camera_set",
    "use_state",
    "use_base_action",
    "base_encoding",
    "action_normalization",
    "mode",
    "max_action_dim",
)


def _legacy_fields(config: Any) -> tuple[Mapping, list[tuple[str, Mapping, str]]]:
    loader = (
        config.get("dataloader_train") if isinstance(config, Mapping) else getattr(config, "dataloader_train", None)
    )
    dataset = loader
    for key in _DATASET_PATH:
        if not isinstance(dataset, Mapping) or key not in dataset:
            raise ValueError("Corrected Phase5 requires the canonical legacy RoboCasa dataset node")
        dataset = dataset[key]
    if not isinstance(dataset, Mapping) or "chunk_length" not in dataset:
        raise ValueError("legacy RoboCasa chunk_length missing")
    fields = []

    def visit(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                name = f"{path}.{key}"
                if key in _CHUNK_KEYS:
                    fields.append((name, value, key))
                else:
                    visit(item, name)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]")

    visit(loader, "dataloader_train")
    prefix = "dataloader_train." + ".".join(_DATASET_PATH)
    if any(path not in {f"{prefix}.{key}" for key in _CHUNK_KEYS} for path, _, _ in fields):
        raise ValueError("ambiguous action chunk fields outside the legacy RoboCasa dataset node")
    values = [node[key] for _, node, key in fields]
    if any(type(value) is not int or value <= 0 for value in values) or len(set(values)) != 1:
        raise ValueError("invalid or conflicting legacy action chunk fields")
    return dataset, fields


def synchronize_legacy_chunk_length(config: Any, contract: Any) -> None:
    """Replace the known inherited Stage-A32 mirror, never a conflicting recipe."""
    dataset, fields = _legacy_fields(config)
    if contract.chunk_length != 16 or contract.action_horizon != 16 or contract.observation_frames != 17:
        raise ValueError("Corrected Phase5 requires H16/chunk16/17 frames")
    if dataset["chunk_length"] not in (16, 32):
        raise ValueError("unsupported inherited legacy action chunk length")
    # Validate the other inherited compatibility fields before modifying anything.
    for key in _DATASET_KEYS:
        expected = getattr(contract, key)
        actual = dataset.get(key)
        if actual != expected or (key != "fps" and type(actual) is not type(expected)):
            raise ValueError(f"legacy dataset contract mismatch: {key}")
    for _, node, key in fields:
        node[key] = contract.chunk_length


def describe_effective_data_source(
    config: Any, contract: Any, *, ttt_steps: int, local_memory_enabled: bool = True
) -> dict[str, Any]:
    """Record real producer provenance separately from the unused legacy mirror."""
    _, fields = _legacy_fields(config)
    metadata = {
        "schema": "corrected_phase5_effective_data_v1",
        "producer_chain": [
            "cosmos_framework.data.generator.action.datasets.robocasa_exact_window_cached_sft."
            "get_action_robocasa_exact_window_cached_sft_dataset",
            "cosmos_framework.model.generator.mot.robocasa_exact_window_local.ExactWindowSegmentProducer",
            "cosmos_framework.trainer.local_memory_grouped.GroupedLocalMemoryTrainer.bind_grouped_stream",
        ],
        "action_horizon": contract.action_horizon,
        "chunk_length": contract.chunk_length,
        "observation_frames": contract.observation_frames,
        "raw_action_dim": contract.action_dim,
        "state_dim": contract.state_dim,
        "fps": contract.fps,
        "local_ttt_tbptt_steps": ttt_steps,
        "legacy_dataloader_role": "config_and_generic_inference_compatibility_only_not_training_producer",
        "legacy_dataset_contract": {key: getattr(contract, key) for key in _DATASET_KEYS},
        "legacy_chunk_fields": {path: node[key] for path, node, key in fields},
    }
    if not local_memory_enabled:
        metadata["producer_chain"][-1] = (
            "cosmos_framework.trainer.matched_no_ttt.MatchedNoTTTTrainer.bind_grouped_stream"
        )
        metadata.update(local_memory_enabled=False, local_ttt_tbptt_steps=None, consumer_block_steps=ttt_steps)
    validate_effective_snapshot(config, metadata)
    return metadata


def validate_effective_snapshot(config: Any, metadata: Mapping) -> None:
    """Validate live config or deserialized YAML before publishing canonical files."""
    dataset, fields = _legacy_fields(config)
    if (
        metadata["schema"] != "corrected_phase5_effective_data_v1"
        or any(
            metadata[key] != value
            for key, value in (
                ("action_horizon", 16),
                ("chunk_length", 16),
                ("observation_frames", 17),
                ("raw_action_dim", 15),
                ("state_dim", 15),
                ("fps", 20.0),
            )
        )
        or (
            metadata.get("local_memory_enabled", True) is not False
            and (type(metadata["local_ttt_tbptt_steps"]) is not int or metadata["local_ttt_tbptt_steps"] <= 0)
        )
        or (
            metadata.get("local_memory_enabled", True) is False
            and (metadata["local_ttt_tbptt_steps"] is not None or metadata.get("consumer_block_steps") != 16)
        )
        or any(node[key] != 16 for _, node, key in fields)
        or metadata["legacy_chunk_fields"] != {path: node[key] for path, node, key in fields}
    ):
        raise ValueError("effective Exact-Window geometry/snapshot mismatch")
    for key in _DATASET_KEYS:
        expected = metadata["legacy_dataset_contract"][key]
        if dataset.get(key) != expected or (key != "fps" and type(dataset.get(key)) is not type(expected)):
            raise ValueError(f"effective legacy dataset contract mismatch: {key}")


def validate_saved_yaml(path: Any, metadata: Mapping) -> None:
    import yaml

    validate_effective_snapshot(yaml.safe_load(path.read_text()), metadata)
