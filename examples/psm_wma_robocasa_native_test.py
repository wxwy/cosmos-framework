"""P1 入口的 CPU 合同测试；临时数据仅用于预检，不冒充训练证据。"""

from __future__ import annotations

import ast
import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import psm_wma_robocasa_native as native
import tomllib


class NativeGlueTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="psm_wma_v3_glue_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dataset = self.root / "data/CloseFridge/20250822/lerobot"
        self.meta = self.dataset / "meta"
        (self.meta / "episodes/chunk-000").mkdir(parents=True)
        (self.meta / "info.json").write_text(json.dumps({"codebase_version": "v3.0"}))
        (self.meta / "tasks.parquet").touch()
        (self.meta / "episodes/chunk-000/file-000.parquet").touch()
        self.eval_dataset = self.root / "original/CloseFridge/20250822/lerobot"
        (self.eval_dataset / "meta").mkdir(parents=True)
        (self.eval_dataset / "meta/info.json").write_text('{"codebase_version": "v2.1"}')
        (self.eval_dataset / "extras").mkdir()
        (self.eval_dataset / "extras/dataset_meta.json").write_text(
            json.dumps({"env_args": {"env_name": "CloseFridge"}})
        )
        self.base = self.root / "base DCP"
        (self.base / "model").mkdir(parents=True)
        (self.base / "model/.metadata").touch()
        self.vae = self.root / "Wan2.2_VAE.pth"
        self.vae.touch()
        self.output = self.root / "output with spaces"
        self.job = self.output / "psm_wma_v3/native_robocasa/smoke"
        self.common = [
            "--output-root",
            str(self.output),
            "--dataset-root",
            str(self.root / "data"),
            "--base-checkpoint",
            str(self.base),
            "--vae",
            str(self.vae),
        ]

    def args(self, stage, *extra):
        return native.parser().parse_args([stage, *self.common, *extra])

    def checkpoint(self, **changes):
        import yaml

        dataset = dict(
            use_base_action=True,
            base_encoding="raw",
            camera_set="left_wrist",
            use_state=True,
            fps=20,
            chunk_length=32,
            action_normalization=None,
        )
        dataset.update(changes)
        config = {
            "dataloader_train": {"dataloader": {"datasets": {"robocasa": {"dataset": dataset}}}},
            "model": {"config": {"tokenizer": {"encode_exact_durations": [33]}}},
        }
        self.job.mkdir(parents=True, exist_ok=True)
        (self.job / "config.yaml").write_text(yaml.safe_dump(config))
        checkpoint = self.job / "checkpoints/iter_000000003"
        for group in ("model", "trainer"):
            (checkpoint / group).mkdir(parents=True, exist_ok=True)
            (checkpoint / group / ".metadata").touch()
        return checkpoint

    def test_schema_and_official_contract(self):
        sys.path.insert(0, str(native.REPO))
        self.addCleanup(sys.path.remove, str(native.REPO))
        from cosmos_framework.configs.toml_config.sft_config import SFTExperimentConfig
        from cosmos_framework.configs.toml_config.toml_config_helper import build_hydra_overrides

        raw = tomllib.loads(native.RECIPE.read_text())
        validated = SFTExperimentConfig.model_validate(raw)
        self.assertEqual(validated.job.experiment, "action_policy_robocasa_nano")
        overrides = build_hydra_overrides(raw)
        for expected in (
            "trainer.grad_accum_iter=1",
            "dataloader_train.max_samples_per_batch=1",
            "model.config.parallelism.data_parallel_shard_degree=1",
        ):
            self.assertIn(expected, overrides)
        self.assertFalse(any("datasets." in item for item in overrides))
        source = (
            native.REPO
            / "cosmos_framework/configs/base/experiment/action/posttrain_config/action_policy_robocasa_nano.py"
        )
        expected = dict(
            use_base_action=True,
            base_encoding="raw",
            camera_set="left_wrist",
            use_state=True,
            fps=20,
            chunk_length=32,
            action_normalization=None,
        )
        found = {
            node.arg: ast.literal_eval(node.value)
            for node in ast.walk(ast.parse(source.read_text()))
            if isinstance(node, ast.keyword) and node.arg in expected and isinstance(node.value, ast.Constant)
        }
        self.assertEqual(found, expected)

    def test_train_is_one_rank_and_keeps_contract(self):
        command, env = native.build_command(self.args("train", "--steps", "1"))
        self.assertIn("torch.distributed.run", command)
        self.assertIn("--nproc-per-node=1", command)
        self.assertIn("trainer.max_iter=1", command)
        self.assertIn(f"{native.DATASET_NODE}.task_names=[CloseFridge]", command)
        self.assertFalse(any("base_encoding=" in item or "chunk_length=" in item for item in command))
        self.assertEqual(env["BASE_CHECKPOINT_PATH"], str(self.base))
        self.assertNotIn("HF_HUB_OFFLINE", env)
        self.assertNotIn("TRANSFORMERS_OFFLINE", env)

    def test_config_never_uses_torchrun(self):
        command, _ = native.build_command(self.args("config"))
        self.assertIn("--dryrun", command)
        self.assertNotIn("torch.distributed.run", command)

    def test_v21_is_rejected(self):
        (self.meta / "info.json").write_text('{"codebase_version": "v2.1"}')
        with self.assertRaisesRegex(ValueError, "需要 LeRobot v3.0"):
            native.build_command(self.args("train"))

    def test_ambiguous_task_date_is_rejected(self):
        (self.dataset.parents[1] / "20990101/lerobot").mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "恰有一个"):
            native.build_command(self.args("train"))

    def test_missing_metadata_is_rejected(self):
        (self.meta / "tasks.parquet").rename(self.meta / "incomplete.parquet")
        with self.assertRaisesRegex(ValueError, "tasks.parquet"):
            native.build_command(self.args("config"))

    def test_hf_directory_is_not_accepted_as_dcp(self):
        with self.assertRaisesRegex(ValueError, "基座 DCP"):
            native.build_command(self.args("train", "--base-checkpoint", str(self.root)))

    def test_existing_checkpoint_prevents_accidental_resume(self):
        self.checkpoint()
        with self.assertRaisesRegex(ValueError, "自动续训"):
            native.build_command(self.args("train"))

    def test_server_uses_training_checkpoint_without_fake_ema_directory(self):
        checkpoint = self.checkpoint()
        command, _ = native.build_command(self.args("server"))
        self.assertIn(str(checkpoint), command)
        for flag, value in (("--raw-action-dim", "15"), ("--action-chunk-size", "32"), ("--fps", "20")):
            self.assertEqual(command[command.index(flag) + 1], value)
        self.assertNotIn("--action-normalization", command)
        self.assertNotIn("--action-stats-path", command)
        self.assertFalse((checkpoint / "model_ema").exists())

    def test_server_rejects_ego_config_in_raw_entrypoint(self):
        self.checkpoint(base_encoding="ego")
        with self.assertRaisesRegex(ValueError, "base_encoding"):
            native.build_command(self.args("server"))

    def test_export_does_not_implicitly_launch_gpu_verify(self):
        self.checkpoint()
        command, _ = native.build_command(self.args("export"))
        self.assertIn("cosmos_framework.scripts.export_model", command)
        self.assertNotIn("--verify", command)

    def test_eval_uses_separate_interpreter_and_one_episode(self):
        command, env = native.build_command(
            self.args(
                "eval",
                "--eval-dataset",
                str(self.eval_dataset),
                "--sim-python",
                "/sim env/bin/python",
                "--port",
                "8912",
            )
        )
        self.assertEqual(command[0], "/sim env/bin/python")
        self.assertIn("http://127.0.0.1:8912", command)
        self.assertEqual(command[command.index("--num-test-episodes") + 1], "1")
        self.assertEqual(command[command.index("--base-encoding") + 1], "raw")
        self.assertIn("--use-state", command)
        self.assertIn("--use-base-action", command)
        self.assertIn("MUJOCO_GL", env)

    def test_eval_missing_extras_or_wrong_task_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "环境元信息"):
            native.build_command(self.args("eval", "--eval-dataset", str(self.root)))
        with self.assertRaisesRegex(ValueError, "评测任务不匹配"):
            native.build_command(self.args("eval", "--eval-dataset", str(self.eval_dataset), "--task", "OpenDrawer"))

    def test_print_command_does_not_execute(self):
        with (
            mock.patch.object(sys, "argv", ["native", "train", *self.common, "--print-command"]),
            mock.patch.object(native.subprocess, "run") as run,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(native.main(), 0)
        run.assert_not_called()

    def test_subprocess_failure_is_propagated(self):
        for value in (None, "0", "1"):
            caller_env = {} if value is None else {"HF_HUB_OFFLINE": value, "TRANSFORMERS_OFFLINE": value}
            with (
                self.subTest(offline=value),
                mock.patch.dict(native.os.environ, caller_env, clear=True),
                mock.patch.object(sys, "argv", ["native", "train", *self.common]),
                mock.patch.object(native.subprocess, "run", return_value=subprocess.CompletedProcess([], 23)) as run,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(native.main(), 23)
                child_env = run.call_args.kwargs["env"]
                for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
                    if value is None:
                        self.assertNotIn(key, child_env)
                    else:
                        self.assertEqual(child_env[key], value)


if __name__ == "__main__":
    unittest.main()
