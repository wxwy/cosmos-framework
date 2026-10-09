"""CPU acceptance for persisted strict RoboCasa source/cache index and warm data parity."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from cosmos_framework.data.generator.action.datasets.robocasa_exact_window_cache import (
    RoboCasaExactWindowCacheCatalog,
    RoboCasaExactWindowEpisodeReader,
)
from cosmos_framework.data.generator.action.datasets.robocasa_exact_window_cached_sft import (
    get_action_robocasa_exact_window_cached_sft_dataset,
)
from cosmos_framework.data.generator.action.datasets.robocasa_exact_window_source import (
    CacheDrivenFlatWindowIndex,
    RoboCasaExactWindowSourceReader,
)
from cosmos_framework.data.generator.action.datasets.robocasa_exact_window_source_test import _cache, _source
from cosmos_framework.data.generator.action.datasets.robocasa_verified_index import (
    AbsoluteRowLookup,
    VerifiedExactWindowIndex,
)


@pytest.fixture
def verified(
    tmp_path: Path,
) -> tuple[Path, Path, Path, RoboCasaExactWindowCacheCatalog, RoboCasaExactWindowSourceReader]:
    cache_root, source_root, index_root = tmp_path / "cache", tmp_path / "source", tmp_path / "verified"
    _cache(cache_root)
    _source(source_root)
    # Official raw15 adapter expects a well-formed quaternion in state16.
    data_file = source_root / "data/chunk-000/file-000.parquet"
    source_table = pq.read_table(data_file)
    states = source_table["observation.state"].to_pylist()
    for state in states:
        state[10:14] = [0.0, 0.0, 0.0, 1.0]
    source_table = source_table.set_column(
        source_table.schema.get_field_index("observation.state"),
        "observation.state",
        pa.array(states),
    )
    pq.write_table(source_table, data_file)
    # Mirror the corrected Edge tokenizer canvas (192x320 / spatial factor 16).
    manifest_file = cache_root / "dataset_manifest.json"
    manifest = json.loads(manifest_file.read_text())
    manifest["latent_shape"] = [5, 48, 12, 20]
    manifest_file.write_text(json.dumps(manifest))
    payload_file = cache_root / "tasks/Pick_Mug/episodes/episode_000000.pt"
    payload = torch.load(payload_file, weights_only=True)
    for window in payload["windows"].values():
        window["latent"] = torch.ones((5, 48, 12, 20), dtype=torch.float32)
    torch.save(payload, payload_file)
    catalog = RoboCasaExactWindowCacheCatalog(cache_root)
    cold = RoboCasaExactWindowSourceReader(catalog, source_root)
    built = VerifiedExactWindowIndex.build(index_root, cache_catalog=catalog, source_reader=cold)
    assert built.source_binding_digest == cold.source_binding_digest
    return cache_root, source_root, index_root, catalog, cold


def test_cold_and_verified_warm_source_window_parity(verified) -> None:
    cache_root, source_root, index_root, old_catalog, cold = verified
    index = VerifiedExactWindowIndex.open(index_root, cache_root=cache_root, source_root=source_root)
    with patch.object(
        RoboCasaExactWindowCacheCatalog,
        "_corpus_digest",
        side_effect=AssertionError("warm load recomputed 2M-window corpus digest"),
    ):
        catalog = RoboCasaExactWindowCacheCatalog(cache_root, verified_index=index)
    with patch.object(
        RoboCasaExactWindowEpisodeReader,
        "_episode",
        side_effect=AssertionError("warm constructor loaded cached .pt"),
    ):
        warm = RoboCasaExactWindowSourceReader(
            catalog,
            source_root,
            identity_scanner=lambda *_: (_ for _ in ()).throw(AssertionError("warm scanned Parquet")),
            verified_index=index,
        )
    assert warm.init_timings_ms["verified_index_hit"] is True
    assert warm.source_binding_digest == cold.source_binding_digest
    assert catalog.corpus_digest == old_catalog.corpus_digest
    assert len(warm.index) == len(cold.index) == 3
    for index_number in range(len(warm.index)):
        assert warm.index[index_number] == cold.index[index_number]
        old = cold.read_at(index_number)
        new = warm.read_at(index_number)
        assert (old.key, old.start_frame, old.global_row_indices) == (
            new.key, new.start_frame, new.global_row_indices
        )
        assert old.ai_caption == new.ai_caption and old.task_class == new.task_class
        torch.testing.assert_close(old.action12, new.action12)
        torch.testing.assert_close(old.state16, new.state16)


def test_no_eager_flat_tuple_expansion_and_compatible_shuffle_block(verified) -> None:
    _, _, _, catalog, _ = verified
    compact = CacheDrivenFlatWindowIndex(catalog)
    assert not hasattr(compact, "_windows") and not hasattr(compact, "_blocks")
    assert len(compact._offsets) == len(catalog.episodes)
    assert [compact[index] for index in range(len(compact))] == [
        (catalog.episodes[0].key, 0),
        (catalog.episodes[0].key, 1),
        (catalog.episodes[0].key, 2),
    ]
    assert compact[-1] == compact[2]
    assert compact.get_shuffle_blocks() == (
        tuple((catalog.episodes[0].key, start) for start in range(3)),
    )
    with pytest.raises(IndexError):
        _ = compact[3]


def test_warm_index_requires_exact_cache_and_source_fingerprint(verified) -> None:
    cache_root, source_root, index_root, catalog, cold = verified
    assert cold.source_binding_digest
    path = cache_root / catalog.episodes[0].relative_path
    witness = path.stat()
    os.utime(path, ns=(witness.st_atime_ns, witness.st_mtime_ns + 10_000_000))
    with pytest.raises(ValueError, match="发生变化"):
        VerifiedExactWindowIndex.open(index_root, cache_root=cache_root, source_root=source_root)


def test_warm_index_rejects_source_data_drift(verified) -> None:
    cache_root, source_root, index_root, _, cold = verified
    parquet = source_root / next(iter(cold._bound.values())).data_file
    with parquet.open("ab") as output:
        output.write(b"mutated")
    with pytest.raises(ValueError, match="发生变化"):
        VerifiedExactWindowIndex.open(index_root, cache_root=cache_root, source_root=source_root)


def test_warm_index_rejects_row_map_corruption_and_missing_index(verified) -> None:
    cache_root, source_root, index_root, _, _ = verified
    with pytest.raises(FileNotFoundError):
        VerifiedExactWindowIndex.open(
            index_root / "missing", cache_root=cache_root, source_root=source_root
        )
    rows = index_root / "absolute_row_mapping.npy"
    with rows.open("ab") as out:
        out.write(b"extra")
    with pytest.raises(ValueError, match="映射哈希"):
        VerifiedExactWindowIndex.open(index_root, cache_root=cache_root, source_root=source_root)


def test_read_only_absolute_lookup_rejects_missing_and_invalid_rows(verified) -> None:
    cache_root, source_root, index_root, _, _ = verified
    loaded = VerifiedExactWindowIndex.open(index_root, cache_root=cache_root, source_root=source_root)
    rows: AbsoluteRowLookup = loaded.open_rows()
    assert len(rows) == 19
    assert rows[100] == 0 and rows[118] == 18
    with pytest.raises(KeyError):
        _ = rows[999]
    with pytest.raises(KeyError):
        _ = rows["100"]  # type: ignore[index]
    with pytest.raises(FileExistsError):
        VerifiedExactWindowIndex.build(index_root, cache_catalog=None, source_reader=None)


def test_factory_catalog_identity_is_shared_in_warm_path(verified) -> None:
    cache_root, source_root, index_root, _, _ = verified
    index = VerifiedExactWindowIndex.open(index_root, cache_root=cache_root, source_root=source_root)
    catalog = RoboCasaExactWindowCacheCatalog(cache_root, verified_index=index)
    with patch(
        "cosmos_framework.data.generator.action.datasets.robocasa_exact_window_cached_sft.ActionTransformPipeline",
        return_value=lambda data, _resolution: data,
    ):
        dataset = get_action_robocasa_exact_window_cached_sft_dataset(
            cache_root=cache_root,
            source_root=source_root,
            tokenizer_config={"test": "tokenizer bypassed by unit test"},
            catalog=catalog,
            verified_index=index,
        )
    assert dataset._dataset.catalog is catalog
    assert dataset._dataset.source_reader.catalog is catalog
    assert dataset._dataset.source_reader.verified_index is index


def test_compact_index_uses_episode_offsets_not_global_window_python_tuples() -> None:
    from types import SimpleNamespace

    counts = (140, 98, 200)
    episodes = tuple(SimpleNamespace(key=f"ep{n}", window_count=count) for n, count in enumerate(counts))
    index = CacheDrivenFlatWindowIndex(
        SimpleNamespace(
            episodes=episodes,
            stats=SimpleNamespace(exact_window_count=sum(counts)),
        )
    )
    assert index._offsets == (0, 140, 238)
    assert len(index) == 438
    assert (index[0], index[139], index[140], index[237], index[238], index[437]) == (
        ("ep0", 0),
        ("ep0", 139),
        ("ep1", 0),
        ("ep1", 97),
        ("ep2", 0),
        ("ep2", 199),
    )


def test_cold_build_loads_cache_episode_payload_only_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, source_root = tmp_path / "cache", tmp_path / "source"
    _cache(root)
    _source(source_root)
    catalog = RoboCasaExactWindowCacheCatalog(root)
    original = torch.load
    reads: list[Path] = []

    def counted_load(filename, *args, **kwargs):
        if str(filename).endswith(".pt"):
            reads.append(Path(filename))
        return original(filename, *args, **kwargs)

    monkeypatch.setattr(torch, "load", counted_load)
    reader = RoboCasaExactWindowSourceReader(catalog, source_root)
    assert len(reader.index) == 3
    assert reads == [root / catalog.episodes[0].relative_path]


def test_full_raw_sample_and_formal_digest_match_across_cold_and_warm(verified, tmp_path: Path) -> None:
    from cosmos_framework.model.generator.mot.robocasa_exact_window_local import (
        ExactWindowLocalCatalog,
        ExactWindowRankPlanner,
    )
    from examples.psm_wma_robocasa_corrected_phase5 import config_digest
    from examples.psm_wma_robocasa_corrected_phase5_test import _config

    cache_root, source_root, index_root, cold_catalog, _ = verified
    index = VerifiedExactWindowIndex.open(index_root, cache_root=cache_root, source_root=source_root)
    warm_catalog = RoboCasaExactWindowCacheCatalog(cache_root, verified_index=index)

    def make_dataset(catalog, verified_index=None):
        with patch(
            "cosmos_framework.data.generator.action.datasets.robocasa_exact_window_cached_sft.ActionTransformPipeline",
            return_value=lambda data, _resolution: data,
        ):
            return get_action_robocasa_exact_window_cached_sft_dataset(
                cache_root=cache_root,
                source_root=source_root,
                tokenizer_config={"test": "bypass in CPU fixture"},
                catalog=catalog,
                verified_index=verified_index,
            )

    cold = make_dataset(cold_catalog)
    warm = make_dataset(warm_catalog, index)
    assert cold._dataset.catalog is cold_catalog
    assert warm._dataset.catalog is warm_catalog
    for number in (0, 1, 2):
        lhs = cold._dataset[number]
        rhs = warm._dataset[number]
        assert (lhs["task_class"], lhs["episode_index"], lhs["start_frame"]) == (
            rhs["task_class"], rhs["episode_index"], rhs["start_frame"]
        )
        for name in ("action", "video_latent", "global_row_indices", "window_frame_indices"):
            torch.testing.assert_close(lhs[name], rhs[name])

    cold_local, warm_local = ExactWindowLocalCatalog(cold, ttt_tbptt_steps=16), ExactWindowLocalCatalog(
        warm, ttt_tbptt_steps=16
    )
    assert cold_local.episodes == warm_local.episodes
    _, config, b_stream, active_ga = _config(tmp_path)
    witnesses = {"edge_config_sha256": "edge", "base_model_metadata_sha256": "droid"}
    assert config_digest(cold_local, config, b_stream=b_stream, active_ga=active_ga, witnesses=witnesses) == (
        config_digest(warm_local, config, b_stream=b_stream, active_ga=active_ga, witnesses=witnesses)
    )
    left_plan = ExactWindowRankPlanner(cold_local, rank=0, world_size=1, b_stream=1, active_ga=1)
    right_plan = ExactWindowRankPlanner(warm_local, rank=0, world_size=1, b_stream=1, active_ga=1)
    assert left_plan.plan_window(left_plan.initial_frontier()) == right_plan.plan_window(right_plan.initial_frontier())


def test_warm_index_rejects_metadata_content_change(verified) -> None:
    cache_root, source_root, index_root, _, _ = verified
    info = source_root / "meta" / "info.json"
    info.write_bytes(info.read_bytes() + b" ")
    with pytest.raises(ValueError, match="发生变化"):
        VerifiedExactWindowIndex.open(index_root, cache_root=cache_root, source_root=source_root)


def test_warm_index_rejects_schema_version_change(verified) -> None:
    cache_root, source_root, index_root, _, _ = verified
    receipt = index_root / "receipt.json"
    payload = json.loads(receipt.read_text())
    payload["schema"] = "unsupported_v0"
    receipt.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="schema"):
        VerifiedExactWindowIndex.open(index_root, cache_root=cache_root, source_root=source_root)
