import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from cosmos_framework.data.generator.action.datasets.libero_lerobot_dataset import LIBEROLeRobotDataset


def test_jsonl_per_episode_layout(tmp_path: Path) -> None:
    meta = tmp_path / "meta"
    data = tmp_path / "data" / "chunk-000"
    meta.mkdir()
    data.mkdir(parents=True)
    (meta / "info.json").write_text(
        json.dumps(
            {
                "fps": 20,
                "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            }
        ),
        encoding="utf-8",
    )
    (meta / "episodes.jsonl").write_text(
        json.dumps({"episode_index": 0, "tasks": ["test task"], "length": 20}) + "\n",
        encoding="utf-8",
    )
    (meta / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": "test task"}) + "\n",
        encoding="utf-8",
    )
    table = pa.table(
        {
            "index": list(range(20)),
            "episode_index": [0] * 20,
            "task_index": [0] * 20,
            "timestamp": [index / 20 for index in range(20)],
            "action": [[0.0] * 7 for _ in range(20)],
        }
    )
    pq.write_table(table, data / "episode_000000.parquet")

    dataset = LIBEROLeRobotDataset(
        root=str(tmp_path),
        chunk_length=16,
        action_normalization=None,
        split="full",
    )

    assert dataset.domain_id == 5
    assert dataset.action_dim == 10
    assert len(dataset) == 4
    assert dataset._tasks == {0: "test task"}
    assert dataset._video_path(dataset._episodes[0], "observation.images.image") == (
        tmp_path / "videos/chunk-000/observation.images.image/episode_000000.mp4"
    )
