"""Stage A Edge raw15 CPU 合同测试；临时数据不冒充训练证据。"""

from __future__ import annotations

import ast
import contextlib
import io
import json
import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
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
        self.dcp_shapes = {
            "net.action2llm.fc.weight": (32, 131072),
            "net.action2llm.bias.weight": (32, 2048),
            "net.llm2action.fc.weight": (32, 131072),
            "net.llm2action.bias.weight": (32, 64),
            "net.action_modality_embed": (2048,),
            "net.vae2llm.weight": (2048, 192),
            "net.llm2vae.weight": (192, 2048),
        }
        self.write_metadata()
        (self.base / "checkpoint.json").write_text(
            json.dumps({"policy": dict(action_chunk_size=32, conditioning_fps=15.0, domain_name="droid_lerobot")})
        )
        self.edge = self.root / "Cosmos3-Edge-Policy-DROID"
        self.edge.mkdir()
        self.edge_model = dict(
            action_gen=True,
            vision_gen=True,
            max_action_dim=64,
            num_embodiment_domains=32,
            tokenizer=dict(encode_exact_durations=[33]),
            vlm_config=dict(model_name="nvidia/Cosmos3-Edge-Policy-DROID"),
        )
        (self.edge / "config.json").write_text(
            json.dumps({"model": {"config": self.edge_model}, "text_config": {"hidden_size": 2048}})
        )
        for name in (
            "tokenizer.json",
            "tokenizer_config.json",
            "processor_config.json",
            "preprocessor_config.json",
            "video_preprocessor_config.json",
            "chat_template.jinja",
        ):
            (self.edge / name).touch()
        env_patch = mock.patch.dict(native.os.environ, {"EDGE_POLICY_CHECKPOINT": str(self.edge)})
        env_patch.start()
        self.addCleanup(env_patch.stop)
        self.vae = self.root / "Wan2.2_VAE.pth"
        self.vae.touch()
        self.output = self.root / "output with spaces"
        self.job = self.output / "psm_wma_v3/edge_robocasa/smoke"
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

    def write_metadata(self):
        metadata = SimpleNamespace(
            state_dict_metadata={key: SimpleNamespace(size=size) for key, size in self.dcp_shapes.items()}
        )
        (self.base / "model/.metadata").write_bytes(pickle.dumps(metadata))

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
            "model": {
                "config": {
                    **self.edge_model,
                    "vlm_config": {"tokenizer": {"tokenizer_type": str(self.edge), "repository": None}},
                }
            },
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
        self.assertEqual(validated.job.experiment, "action_policy_robocasa_edge")
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
        self.assertEqual(env["EDGE_POLICY_CHECKPOINT"], str(self.edge))
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
            caller_env = {"EDGE_POLICY_CHECKPOINT": str(self.edge)}
            if value is not None:
                caller_env.update(HF_HUB_OFFLINE=value, TRANSFORMERS_OFFLINE=value)
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

    def test_missing_empty_or_remote_tokenizer_is_rejected(self):
        for value in (None, "", "  ", "nvidia/Cosmos3-Edge-Policy-DROID"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                native.check_edge_checkpoint(value)

    def test_nano_and_incomplete_heads_are_rejected(self):
        for key in ("net.action2llm.fc.weight", "net.llm2action.fc.weight", "net.action_modality_embed"):
            with self.subTest(key=key):
                shape = self.dcp_shapes.pop(key)
                self.write_metadata()
                with self.assertRaisesRegex(ValueError, "Edge DCP"):
                    native.check_droid_dcp(self.base)
                self.dcp_shapes[key] = shape
        self.dcp_shapes["net.action2llm.fc.weight"] = (32, 262144)
        self.write_metadata()
        with self.assertRaisesRegex(ValueError, "action2llm"):
            native.check_droid_dcp(self.base)

    def test_base_policy_metadata_is_required(self):
        (self.base / "checkpoint.json").write_text('{"policy": {"domain_name": "robocasa_lerobot"}}')
        with self.assertRaisesRegex(ValueError, "Edge-Policy-DROID"):
            native.check_droid_dcp(self.base)

    def test_real_composed_edge_contract_and_upstream_not_mutated(self):
        from omegaconf import OmegaConf

        from cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_robocasa_nano import (
            action_policy_robocasa_nano,
        )
        from cosmos_framework.configs.base.experiment.sft.models.edge_model_config import EDGE_MODEL_CONFIG
        from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml

        command, env = native.build_command(self.args("config"))
        with mock.patch.dict(native.os.environ, env):
            cfg = load_experiment_from_toml(native.RECIPE, extra_overrides=command[command.index("--") + 1 :])
            model = cfg.model.config
            ds = cfg.dataloader_train.dataloader.datasets.robocasa.dataset
            self.assertTrue(model.action_gen and model.vision_gen)
            self.assertEqual((model.max_action_dim, model.num_embodiment_domains), (64, 32))
            self.assertEqual(list(model.tokenizer.encode_exact_durations), [33])
            self.assertEqual(model.vlm_config.tokenizer.tokenizer_type, str(self.edge))
            self.assertIsNone(model.vlm_config.tokenizer.repository)
            self.assertIsNone(model.vlm_config.tokenizer.revision)
            self.assertFalse(model.diffusion_expert_config.load_weights_from_pretrained)
            self.assertFalse(model.vlm_config.pretrained_weights.enabled)
            self.assertFalse(model.ema.enabled)
            self.assertEqual(list(cfg.checkpoint.keys_to_skip_loading), ["net_ema."])
            self.assertFalse(cfg.checkpoint.load_training_state)
            self.assertFalse(cfg.checkpoint.load_ema_to_reg)
            self.assertTrue(cfg.checkpoint.strict_resume)
            self.assertEqual(
                list(cfg.optimizer.weight_decay_skip_patterns),
                [
                    r"action2llm\.(fc|bias)\.weight$",
                    r"llm2action\.(fc|bias)\.weight$",
                ],
            )
            self.assertEqual(cfg.checkpoint.load_path, str(self.base))
            self.assertEqual((cfg.trainer.grad_accum_iter, cfg.dataloader_train.max_samples_per_batch), (1, 1))
            expected = dict(
                use_base_action=True,
                base_encoding="raw",
                camera_set="left_wrist",
                use_state=True,
                fps=20,
                chunk_length=32,
                action_normalization=None,
            )
            for key, value in expected.items():
                self.assertEqual(ds[key], value)
                self.assertEqual(
                    action_policy_robocasa_nano.dataloader_train.dataloader.datasets.robocasa.dataset[key], value
                )
            self.assertEqual(ds.max_action_dim, 64)
            self.assertEqual(ds.tokenizer_config.tokenizer_type, str(self.edge))
            self.assertEqual(EDGE_MODEL_CONFIG["vlm_config"]["tokenizer"]["repository"], "nvidia/Cosmos3-Edge")
            self.assertIsNone(EDGE_MODEL_CONFIG["tokenizer"]["encode_exact_durations"])
            # 生产 YAML 往返，验证 server 接受同一份合成配置。
            from cosmos_framework.utils.lazy_config import LazyConfig
            from cosmos_framework.utils.serialization import to_yaml

            path = self.root / "composed.yaml"
            try:
                to_yaml(cfg, str(path))
            except Exception:
                LazyConfig.save_yaml(cfg, str(path))
            native.check_raw15_config(path)
        tokenizer = OmegaConf.create({"path": "${oc.env:EDGE_POLICY_CHECKPOINT}"})
        with mock.patch.dict(native.os.environ, {}, clear=True), self.assertRaises(Exception):
            OmegaConf.to_container(tokenizer, resolve=True)

    def test_raw15_and_ego20_use_existing_dataset_width(self):
        from cosmos_framework.data.generator.action.datasets.robocasa_lerobot_dataset import RoboCasaLeRobotDataset

        dataset = object.__new__(RoboCasaLeRobotDataset)
        dataset._use_base_action = True
        for encoding, width in (("raw", 15), ("ego", 20)):
            dataset._base_encoding = encoding
            self.assertEqual(dataset.action_dim, width)

    def test_current_dcp_planner_keeps_droid_heads(self):
        import torch

        from cosmos_framework.checkpoint.dcp import CustomLoadPlanner

        state = {key: torch.zeros(1) for key in self.dcp_shapes}
        state["net.action_pos_embed"] = torch.zeros(1)
        state["net_ema.action2llm.fc.weight"] = torch.zeros(1)
        planner = CustomLoadPlanner(keys_to_skip_loading=["net_ema."], allow_partial_load=False)
        kept = planner._skip_keys_if_found(state)
        self.assertEqual(set(kept), set(state) - {"net_ema.action2llm.fc.weight"})
        self.assertIn("net_ema.action2llm.fc.weight", state)

    def test_raw15_loss_masks_tail_and_inference_unpads(self):
        import torch

        from cosmos_framework.data.generator.action.utils.action_processing import ActionProcessor
        from cosmos_framework.model.generator.algorithm.loss.flow_matching import compute_flow_matching_loss

        pred = torch.ones(2, 64, requires_grad=True)
        with torch.no_grad():
            pred[:, 15:] = 1000
        rf = SimpleNamespace(train_time_weight=lambda t, tensor_kwargs: torch.ones_like(t))
        loss, _ = compute_flow_matching_loss(
            [pred],
            [torch.zeros_like(pred)],
            [torch.zeros(2, 1)],
            torch.zeros(1, 1),
            True,
            rf,
            dict(device="cpu", dtype=torch.float32),
            raw_action_dim=[torch.tensor(15)],
        )
        self.assertEqual(loss.item(), 1.0)
        loss.backward()
        self.assertEqual(pred.grad[:, 15:].count_nonzero().item(), 0)
        self.assertGreater(pred.grad[:, :15].count_nonzero().item(), 0)
        self.assertEqual(ActionProcessor._unpad_action(pred, 15).shape, (2, 15))


@unittest.skipUnless(
    native.os.environ.get("EDGE_POLICY_CHECKPOINT") and native.os.environ.get("BASE_CHECKPOINT_PATH"),
    "本地资产核验需显式提供 EDGE_POLICY_CHECKPOINT / BASE_CHECKPOINT_PATH",
)
class LocalEdgeAssetsTest(unittest.TestCase):
    def test_local_hf_and_dcp_metadata_contract(self):
        root = native.check_edge_checkpoint(native.os.environ["EDGE_POLICY_CHECKPOINT"])
        native.check_droid_dcp(Path(native.os.environ["BASE_CHECKPOINT_PATH"]))
        policy = json.loads((root / "checkpoint.json").read_text())["policy"]
        self.assertEqual(policy, dict(action_chunk_size=32, conditioning_fps=15.0, domain_name="droid_lerobot"))

    def test_local_processor_without_hub_fallback(self):
        from cosmos_framework.data.generator.processors import build_processor_lazy

        with mock.patch(
            "cosmos_framework.utils.checkpoint_db.CheckpointDirHf.download",
            side_effect=AssertionError("本地 Edge tokenizer 不得调用 Hub 下载"),
        ):
            processor = build_processor_lazy(
                tokenizer_type=native.os.environ["EDGE_POLICY_CHECKPOINT"],
                repository=None,
                revision=None,
            )
        self.assertEqual(processor.tokenizer.convert_tokens_to_ids("<|vision_start|>"), 20)
        self.assertEqual(processor.tokenizer.convert_tokens_to_ids("<|vision_end|>"), 21)


if __name__ == "__main__":
    unittest.main()
