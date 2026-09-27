"""B2-C harness 的 CPU preflight、原生 callback 与失败闭锁测试。"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from cosmos_framework.model.generator.mot.memory_prefix import LocalMemoryRuntime
from cosmos_framework.utils.generator.optimizer import _build_params_with_metadata
from examples import psm_wma_robocasa_local_s1 as s1


def _paths(tmp_path: Path) -> s1.SmokePaths:
    run = tmp_path / "smoke"
    return s1.SmokePaths(
        run / "checkpoints/iter_000000001",
        run / "config.yaml",
        tmp_path / "cache/CloseFridge/20250816/lerobot/ep_000000.h5",
        tmp_path / "robocasa",
        tmp_path / "edge",
        tmp_path / "vae.pth",
        tmp_path / "output",
    )


def _monkeypatch_defaults(monkeypatch: pytest.MonkeyPatch, paths: s1.SmokePaths) -> None:
    for key, value in (
        ("DEFAULT_CHECKPOINT", paths.checkpoint),
        ("DEFAULT_CONFIG", paths.config),
        ("DEFAULT_CACHE", paths.cache),
        ("DEFAULT_DATASET_ROOT", paths.dataset_root),
        ("DEFAULT_EDGE", paths.edge),
        ("DEFAULT_VAE", paths.vae),
    ):
        monkeypatch.setattr(s1, key, value)


def test_cli_requires_unique_output_and_frozen_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _paths(tmp_path)
    _monkeypatch_defaults(monkeypatch, paths)
    args = s1.parser().parse_args(["--preflight", "--output", str(paths.output)])
    resolved = s1.paths_from_args(args)
    s1.validate_frozen_paths(resolved)
    assert resolved == paths
    assert s1.T == 16 and s1.TASK == "CloseFridge" and s1.EPISODE_INDEX == s1.CURSOR == 0
    args = s1.parser().parse_args(["--output", str(paths.output), "--cache", str(tmp_path / "other.h5")])
    with pytest.raises(ValueError, match="冻结资产"):
        s1.validate_frozen_paths(s1.paths_from_args(args))
    with pytest.raises(SystemExit):
        s1.parser().parse_args(["--preflight"])


def test_stage_a_contract_and_dcp_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from examples import psm_wma_robocasa_native as stage_a

    paths = _paths(tmp_path)
    monkeypatch.setattr(stage_a, "check_edge_checkpoint", lambda value: Path(value))
    for file in (
        paths.config,
        paths.checkpoint / "model/.metadata",
        paths.cache,
        paths.vae,
        paths.edge / "config.json",
        paths.dataset_root / "CloseFridge/20250816/lerobot/meta/info.json",
    ):
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("{}")
    info = paths.dataset_root / "CloseFridge/20250816/lerobot/meta/info.json"
    info.write_text(json.dumps({"codebase_version": "v3.0"}))
    config = {
        "dataloader_train": {
            "dataloader": {
                "datasets": {
                    "robocasa": {
                        "dataset": {
                            "use_base_action": True,
                            "base_encoding": "raw",
                            "camera_set": "left_wrist",
                            "use_state": True,
                            "fps": 20,
                            "chunk_length": 32,
                            "action_normalization": None,
                            "root": str(paths.dataset_root),
                            "task_names": ["CloseFridge"],
                            "max_action_dim": 64,
                        }
                    }
                }
            }
        },
        "optimizer": {"optimizer_type": "FusedAdam", "lr": 5e-5},
        "model": {
            "config": {
                "tokenizer": {"encode_exact_durations": [33], "vae_path": str(paths.vae)},
                "max_action_dim": 64,
                "num_embodiment_domains": 32,
                "vlm_config": {"tokenizer": {"tokenizer_type": str(paths.edge)}},
                "activation_checkpointing": {"mode": "selective"},
                "compile": {"enabled": False},
                "ema": {"enabled": False},
                "lbl": {"coeff_gen": None, "coeff_und": None},
            }
        },
    }
    import yaml

    paths.config.write_text(yaml.safe_dump(config))
    keys = {
        "net.action2llm.fc.weight",
        "net.llm2action.fc.weight",
        "net.action_modality_embed",
        "net.vae2llm.weight",
        "net.llm2vae.weight",
    }
    monkeypatch.setattr(
        s1,
        "FileSystemReader",
        lambda path: SimpleNamespace(read_metadata=lambda: SimpleNamespace(state_dict_metadata=dict.fromkeys(keys))),
    )
    assert s1.read_stage_a_contract(paths)["dcp_keys"] == 5
    config["model"]["config"]["tokenizer"]["encode_exact_durations"] = [17]
    paths.config.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="Edge/资源"):
        s1.read_stage_a_contract(paths)


def test_warm_start_only_allows_local_and_disabled_ema(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = nn.Module()
    model.net = nn.Module()
    model.net.host = nn.Linear(2, 2)
    model.net.local_memory2llm = nn.Linear(2, 2)
    source = {
        name: SimpleNamespace(size=value.shape)
        for name, value in model.state_dict().items()
        if "local_memory" not in name
    }
    monkeypatch.setattr(
        s1,
        "FileSystemReader",
        lambda path: SimpleNamespace(read_metadata=lambda: SimpleNamespace(state_dict_metadata=source)),
    )
    report = s1.validate_warm_start_keys(model, tmp_path)
    assert all("local_memory" in name for name in report["missing_local"])
    source.pop("net.host.weight")
    with pytest.raises(ValueError, match="缺少 host"):
        s1.validate_warm_start_keys(model, tmp_path)
    source["net.host.weight"] = SimpleNamespace(size=(2, 2))
    source["net.unexpected"] = SimpleNamespace(size=(1,))
    with pytest.raises(ValueError, match="意外 host"):
        s1.validate_warm_start_keys(model, tmp_path)


def test_stage_a_dcp_load_writes_host_and_preserves_local(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import torch.distributed.checkpoint as dcp

    model = nn.Module()
    model.net = nn.Module()
    model.net.host = nn.Linear(2, 2)
    model.net.local_memory2llm = nn.Linear(2, 2)
    before_local = model.net.local_memory2llm.weight.detach().clone()
    monkeypatch.setattr(s1, "validate_warm_start_keys", lambda owner, checkpoint: {"host_keys": 2})
    monkeypatch.setattr(s1, "FileSystemReader", lambda path: str(path))

    def fake_load(*, state_dict, storage_reader, planner, no_dist):
        assert storage_reader == str(tmp_path / "model") and no_dist
        assert planner.keys_to_skip_loading == ["local_memory"]
        state_dict["net.host.weight"] = torch.full_like(state_dict["net.host.weight"], 7)

    monkeypatch.setattr(dcp, "load", fake_load)
    assert s1.load_stage_a_host(model, tmp_path) == {"host_keys": 2}
    torch.testing.assert_close(model.net.host.weight, torch.full_like(model.net.host.weight, 7))
    torch.testing.assert_close(model.net.local_memory2llm.weight, before_local)


def test_episode0_official_raw15_overlap_and_b1_segment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _paths(tmp_path)
    import yaml

    paths.config.parent.mkdir(parents=True)
    paths.config.write_text(
        yaml.safe_dump(
            {
                "dataloader_train": {
                    "dataloader": {
                        "datasets": {
                            "robocasa": {
                                "dataset": {
                                    "tokenizer_config": {
                                        "_target_": "cosmos_framework.data.generator.processors.build_processor_lazy",
                                        "tokenizer_type": str(paths.edge),
                                    },
                                    "cfg_dropout_rate": 0.1,
                                    "max_action_dim": 64,
                                    "append_viewpoint_info": True,
                                    "append_duration_fps_timestamps": True,
                                    "append_resolution_info": True,
                                    "append_idle_frames": True,
                                    "format_prompt_as_json": True,
                                    "resolution": None,
                                }
                            }
                        }
                    }
                }
            }
        )
    )
    from cosmos_framework.data.generator.action.utils import transforms
    from cosmos_framework.data.generator.augmentors import text_tokenizer

    class FakeProcessor:
        def tokenize_text(self, caption, *, is_video, use_system_prompt):
            assert not is_video and not use_system_prompt
            return list(caption.encode())

    monkeypatch.setattr(text_tokenizer, "lazy_instantiate", lambda config: FakeProcessor())

    def resize(self, data, resolution):
        assert resolution is None
        data["image_size"] = (2, 4)
        return data

    monkeypatch.setattr(transforms.VideoResize, "__call__", resize)
    transform, resolution = s1.build_stage_a_action_transform(paths)
    assert resolution is None and transform.max_action_dim == 64
    assert transform.prompt_json_formatter is not None and transform.text_tokenizer.cfg_dropout_rate == 0.1
    assert transform.text_tokenizer._processor.__class__ is FakeProcessor
    raw12 = torch.arange(48 * 12, dtype=torch.float32).reshape(48, 12) / 100
    arm = torch.cat((raw12[:, 5:8], torch.ones(48, 6), raw12[:, 11:12]), dim=-1)
    raw15 = torch.cat((raw12[:, :5], arm), dim=-1)
    corrupt = {"enabled": False}

    class FakeRows:
        def __getitem__(self, key):
            assert key == slice(0, 48)
            return {"episode_index": [0] * 48, "frame_index": list(range(48)), "action": list(raw12)}

    class FakeDataset:
        def __init__(self, **kwargs):
            assert kwargs["split"] == "full" and kwargs["camera_set"] == "left_wrist"
            assert kwargs["use_base_action"] and kwargs["base_encoding"] == "raw"
            self._all_shard_roots = [kwargs["root"]]
            self._episode_records = [(0, 0, 16, 0)]
            self._episode_cum_ends = [16]

        def _get_dataset(self, source):
            assert source == 0
            return SimpleNamespace(meta=SimpleNamespace(episodes={"length": [48]}), hf_dataset=FakeRows())

        def _resolve_index(self, index):
            return 0, index, 0, index

        def _build_frame_wise_action(self, raw):
            return torch.cat((raw[:, 5:8], torch.ones(len(raw), 6), raw[:, 11:12]), dim=-1)

        def __getitem__(self, index):
            action = torch.cat((torch.zeros(1, 15), raw15[index : index + 32]))
            if corrupt["enabled"] and index == 3:
                action[1, 0] += 1
            payload = {
                "action": action,
                "video": torch.zeros(3, 33, 2, 4),
                "ai_caption": "CloseFridge",
                "conditioning_fps": torch.tensor(20),
                "mode": "wam",
                "domain_id": torch.tensor(30),
                "viewpoint": "concat_view",
                "idle_frames": torch.tensor(0),
                "additional_view_description": "The left half is a third-person view. The right half is a wrist view.",
            }
            assert "text_token_ids" not in payload and "sequence_plan" not in payload
            return payload

    from cosmos_framework.data.generator.action.datasets import robocasa_lerobot_dataset

    monkeypatch.setattr(robocasa_lerobot_dataset, "RoboCasaLeRobotDataset", FakeDataset)

    class FakeReader:
        def __init__(self, path, *, expected_episode_id, expected_source_frames):
            assert path == paths.cache and expected_episode_id == "ep_000000" and expected_source_frames == 48
            self.episode_id, self.source_frames = expected_episode_id, expected_source_frames

        def visual_summary(self, step):
            return step, torch.zeros(96)

    monkeypatch.setattr(s1, "RoboCasaLatentReader", FakeReader)
    segment, identity, evidence = s1.load_episode_segment(paths)
    assert segment.consumer_step.tolist() == [list(range(16))]
    assert segment.evidence_source_step.tolist() == [[-1, *range(15)]]
    assert identity.episode_id == "ep_000000"
    assert {key: evidence[key] for key in ("frames", "raw_action_dim", "payloads", "transform_seed")} == {
        "frames": 48,
        "raw_action_dim": 15,
        "payloads": 16,
        "transform_seed": 0,
    }
    assert evidence["consumer0"]["ai_caption"]["actions"][0]["description"] == "CloseFridge."
    assert evidence["consumer0"]["text_token_count"] > 0
    assert evidence["consumer0"]["action_shape"] == [33, 64]
    assert evidence["consumer0"]["action_raw_shape"] == [33, 15]
    for step, payload in enumerate(segment.consumer_payload[0]):
        assert payload["text_token_ids"].dtype == torch.long and payload["text_token_ids"].numel()
        assert payload["sequence_plan"].has_action and payload["sequence_plan"].has_text
        torch.testing.assert_close(payload["action_raw"][1:], raw15[step : step + 32])
        assert payload["action"].shape == (33, 64) and payload["raw_action_dim"] == 15
        assert "video_latent" not in payload
    repeated, _, repeated_evidence = s1.load_episode_segment(paths)
    assert repeated_evidence["consumer0"]["text_token_sha256"] == evidence["consumer0"]["text_token_sha256"]
    assert repeated_evidence["consumer0"]["text_token_count"] == evidence["consumer0"]["text_token_count"]
    assert all(payload["ai_caption"] for payload in repeated.consumer_payload[0])
    corrupt["enabled"] = True
    with pytest.raises(ValueError, match="重叠 raw15"):
        s1.load_episode_segment(paths)


def test_untransformed_raw_payload_reproduces_missing_text_tokens() -> None:
    raw = torch.zeros(33, 15)
    payload = {"action": raw, "video": torch.zeros(3, 33, 2, 4), "ai_caption": "CloseFridge"}
    assert "text_token_ids" not in payload and "sequence_plan" not in payload
    with pytest.raises(ValueError, match="text_token_ids"):
        s1._validate_native_payload(payload, raw, 0)


def test_local_config_overlay_and_optimizer_inventory(monkeypatch: pytest.MonkeyPatch) -> None:
    config = SimpleNamespace(
        model=SimpleNamespace(
            config=SimpleNamespace(
                ema=SimpleNamespace(enabled=True),
                compile=SimpleNamespace(enabled=True),
                activation_checkpointing=SimpleNamespace(mode="selective"),
            )
        ),
        optimizer=SimpleNamespace(optimizer_type="FusedAdam", lr=5e-5, betas=[0.9, 0.99], eps=1e-8, weight_decay=0.05),
    )
    s1.overlay_local_config(config)
    assert config.model.config.local_memory_enabled and config.model.config.local_memory_ttt_tbptt_steps == 16
    assert config.model.config.local_memory_k_local == 4 and config.model.config.local_memory_action_dim == 15
    assert not config.model.config.ema.enabled and not config.model.config.compile.enabled
    assert config.optimizer.keys_to_select == ["local_memory"] and config.optimizer.lr == 5e-5

    model = nn.Module()
    model.net = nn.Module()
    model.net.local_memory_runtime = LocalMemoryRuntime()
    model.net.local_memory2llm = nn.Linear(32, 2048)
    model.net.local_memory_modality_embed = nn.Parameter(torch.ones(2048))
    model.net.host = nn.Linear(2048, 1)
    model.external_host = nn.Linear(2, 2)

    def fake_build_optimizer(owner, *, keys_to_select, lr, **kwargs):
        assert keys_to_select == ["local_memory"] and lr == 5e-5
        selected = _build_params_with_metadata(
            owner,
            keys_to_select=keys_to_select,
            lr_multipliers={},
            base_lr=lr,
            base_weight_decay=0.05,
            disable_weight_decay_for_1d_params=False,
        )
        return SimpleNamespace(optimizers=[torch.optim.SGD([parameter for parameter, _ in selected], lr=lr)])

    from cosmos_framework.utils.generator import optimizer as optimizer_module

    monkeypatch.setattr(optimizer_module, "build_optimizer", fake_build_optimizer)
    optimizer = s1.local_optimizer(model, config)
    assert sum(parameter.numel() for group in optimizer.param_groups for parameter in group["params"]) == 165_312
    assert all(not parameter.requires_grad for parameter in model.net.host.parameters())
    assert all(not parameter.requires_grad for parameter in model.external_host.parameters())


def test_stage_a_yaml_uses_lazy_config_loader(tmp_path: Path) -> None:
    from cosmos_framework.utils.lazy_config import LazyConfig

    config_path = tmp_path / "stage_a.yaml"
    config_path.write_text(
        "model:\n  config:\n    ema: {enabled: true}\n    compile: {enabled: true}\n"
        "    activation_checkpointing: {mode: selective}\noptimizer:\n  lr: 0.00005\n"
    )
    config = LazyConfig.load(str(config_path))
    s1.overlay_local_config(config)
    assert config.model.config.local_memory_enabled
    assert config.model.config.local_memory_ttt_tbptt_steps == 16
    assert config.optimizer.keys_to_select == ["local_memory"]


def test_pair_lock_rejects_gitlink_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    values = iter(("a" * 40, "b" * 40, f"160000 commit {'c' * 40}\tcosmos-framework"))
    monkeypatch.setattr(s1.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=next(values)))
    with pytest.raises(ValueError, match="Gitlink"):
        s1.lock_implementation_pair()
    values = iter(("a" * 40, "b" * 40, f"160000 commit {'b' * 40}\tcosmos-framework", ""))
    assert s1.lock_implementation_pair() == {"root": "a" * 40, "child": "b" * 40, "gitlink": "b" * 40}
    values = iter(("a" * 40, "b" * 40, f"160000 commit {'b' * 40}\tcosmos-framework", " M examples/x.py"))
    with pytest.raises(ValueError, match="干净"):
        s1.lock_implementation_pair()


def test_cuda_memory_record_schema_and_no_gpu_guard(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    for name, value in (
        ("memory_allocated", 1),
        ("memory_reserved", 2),
        ("max_memory_allocated", 3),
        ("max_memory_reserved", 4),
    ):
        monkeypatch.setattr(torch.cuda, name, lambda value=value: value)
    trace = []
    s1.record_cuda(trace, "consumer_forward", 7)
    assert trace == [
        {
            "phase": "consumer_forward",
            "consumer": 7,
            "allocated_bytes": 1,
            "reserved_bytes": 2,
            "peak_allocated_bytes": 3,
            "peak_reserved_bytes": 4,
        }
    ]
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(RuntimeError, match="单 rank"):
        s1.execute_cuda(_paths(tmp_path), [], {}, None, None)


def test_native_consumer_uses_joint_batch_abi_and_packs_text() -> None:
    from cosmos_framework.data.generator.sequence_packing.sequence import PackedSequenceBuilder
    from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel

    payload = {
        "video": torch.zeros(3, 33, 2, 4),
        "action": torch.zeros(33, 64),
        "action_raw": torch.zeros(33, 15),
        "text_token_ids": torch.tensor([1, 2, 3]),
        "conditioning_fps": torch.tensor(20),
        "raw_action_dim": 15,
    }
    batch = s1.collate_native_consumer(payload)
    for key, shape in (
        ("video", (3, 33, 2, 4)),
        ("action", (33, 64)),
        ("action_raw", (33, 15)),
        ("text_token_ids", (3,)),
    ):
        assert len(batch[key]) == len(batch[key][0]) == 1
        assert batch[key][0][0].shape == shape
    assert len(batch["conditioning_fps"]) == 1 and batch["conditioning_fps"][0].shape == (1,)
    text_ids = OmniMoTModel._load_and_tokenize_text_data(object(), batch, iteration=0)
    assert text_ids == [[1, 2, 3]]
    builder = PackedSequenceBuilder()
    assert builder.pack_text_tokens(text_ids[0], {"eos_token_id": 42, "start_of_generation": 43}, True) == 5
    assert builder.text_ids == [1, 2, 3, 42, 43]


def test_local_witness_uses_cpu_dtensor_shard(tmp_path: Path) -> None:
    import torch.distributed as dist
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.tensor import DTensor, Shard

    assert not dist.is_initialized()
    dist.init_process_group("gloo", store=dist.FileStore(str(tmp_path / "store"), 1), rank=0, world_size=1)
    try:
        mesh = DeviceMesh("cpu", [0])

        def shard(value: torch.Tensor) -> DTensor:
            return DTensor.from_local(value, mesh, (Shard(0),))

        names = (
            "net.local_memory_runtime.encoder.visual_proj.weight",
            "net.local_memory_runtime.core.slot_queries",
            "net.local_memory_runtime.core.w0_fast_in_weight",
            "net.local_memory2llm.weight",
            "net.local_memory_modality_embed",
        )
        parameters = {name: SimpleNamespace(grad=shard(torch.ones(2))) for name in names}
        parameters["net.host.weight"] = SimpleNamespace(grad=None)
        model = SimpleNamespace(named_parameters=lambda: iter(parameters.items()))
        witness = s1._witness(model)
        assert set(witness) == set(names)
        assert all(item["nonzero"] == 2 and item["norm"] == pytest.approx(2**0.5) for item in witness.values())
        local = s1._local_witness_tensor(parameters[names[0]].grad)
        assert isinstance(local, torch.Tensor) and not isinstance(local, DTensor)
        torch.testing.assert_close(local, torch.ones(2))
        assert s1._local_witness_tensor(local) is local
        before = s1._local_witness_tensor(shard(torch.ones(2))).clone()
        assert torch.equal(before, s1._local_witness_tensor(shard(torch.ones(2))))
        assert not torch.equal(before, s1._local_witness_tensor(shard(torch.zeros(2))))

        parameters[names[0]].grad = shard(torch.zeros(2))
        with pytest.raises(ValueError, match="全零"):
            s1._witness(model)
        parameters[names[0]].grad = shard(torch.tensor([1.0, float("nan")]))
        with pytest.raises(ValueError, match="非有限"):
            s1._witness(model)
        parameters[names[0]].grad = shard(torch.ones(2))
        parameters["net.host.weight"].grad = shard(torch.ones(2))
        with pytest.raises(ValueError, match="host 参数出现梯度"):
            s1._witness(model)
    finally:
        dist.destroy_process_group()


def test_native_callback_single_sample_rgb_and_leaf(monkeypatch: pytest.MonkeyPatch) -> None:
    events = []
    monkeypatch.setattr(s1, "record_cuda", lambda trace, phase, index=None: events.append((phase, index)))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    from cosmos_framework.utils import misc

    monkeypatch.setattr(misc, "to", lambda batch, *, device: batch)

    class FakeModel:
        def training_step(self, batch, iteration, *, _local_memory_prefixes):
            assert iteration == 0 and len(batch["video"]) == len(batch["video"][0]) == 1
            assert batch["text_token_ids"][0][0].tolist() == [1, 2, 3]
            leaf = _local_memory_prefixes[0]
            return {}, torch.tensor(0.0) if leaf is None else leaf.square().sum()

    callback = s1.native_callback(FakeModel(), [], {})
    payload = {
        "video": torch.zeros(3, 33, 2, 4),
        "action": torch.zeros(33, 64),
        "action_raw": torch.zeros(33, 15),
        "text_token_ids": torch.tensor([1, 2, 3]),
        "ai_caption": "CloseFridge",
    }
    assert callback(payload, None, 0).loss == 0
    leaf = torch.ones(4, 32, requires_grad=True)
    result = callback(payload, leaf, 1)
    result.loss.backward()
    assert events == [("consumer_forward", 0), ("consumer_forward", 1)]
    callback(payload, leaf, 2)
    assert events[-2:] == [("consumer_backward", 1), ("consumer_forward", 2)]
    assert leaf.grad is not None and result.output["loss"].grad_fn is None
    with pytest.raises(ValueError, match="RGB 权威"):
        callback({**payload, "video_latent": torch.zeros(1)}, leaf, 1)

    class AuxiliaryModel(FakeModel):
        def training_step(self, batch, iteration, *, _local_memory_prefixes):
            return {"aux_loss_gen": torch.tensor(0.1)}, torch.tensor(1.0)

    with pytest.raises(ValueError, match="auxiliary"):
        s1.native_callback(AuxiliaryModel(), [], {})(payload, leaf, 1)


def test_main_failure_writes_result_without_cuda(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _paths(tmp_path)
    _monkeypatch_defaults(monkeypatch, paths)
    monkeypatch.setattr(s1, "read_stage_a_contract", lambda paths: (_ for _ in ()).throw(ValueError("metadata failed")))
    assert s1.main(["--preflight", "--output", str(paths.output)]) == 1
    result = json.loads((paths.output / "result.json").read_text())
    assert result["status"] == "FAIL" and result["error"]["message"] == "metadata failed"
    assert json.loads((paths.output / "cuda_memory_trace.json").read_text()) == []
    assert s1.main(["--preflight", "--output", str(paths.output)]) == 1


def test_oom_writes_phase_and_memory_without_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _paths(tmp_path)
    _monkeypatch_defaults(monkeypatch, paths)
    monkeypatch.setattr(s1, "read_stage_a_contract", lambda paths: {"dcp_keys": 549})
    monkeypatch.setattr(s1, "load_episode_segment", lambda paths: (None, None, {"frames": 429, "payloads": 16}))
    monkeypatch.setattr(s1, "lock_implementation_pair", lambda: {"root": "r", "child": "c", "gitlink": "c"})
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)

    def fail_runtime(paths, trace, state, segment, identity):
        state.update(phase="native_backward", consumer=7)
        raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    monkeypatch.setattr(s1, "execute_cuda", fail_runtime)
    monkeypatch.setattr(
        s1, "record_cuda", lambda trace, phase, consumer=None: trace.append({"phase": phase, "consumer": consumer})
    )
    assert s1.main(["--output", str(paths.output)]) == 1
    result = json.loads((paths.output / "result.json").read_text())
    assert result["status"] == "FAIL"
    assert result["error"] == {
        "type": "OutOfMemoryError",
        "message": "CUDA out of memory",
        "phase": "native_backward",
        "consumer": 7,
    }
    assert json.loads((paths.output / "cuda_memory_trace.json").read_text()) == [{"phase": "failure", "consumer": 7}]
