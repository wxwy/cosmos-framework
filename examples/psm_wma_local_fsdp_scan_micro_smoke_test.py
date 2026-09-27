"""R1-B tiny FSDP harness 的 CPU/static 准入测试；不启动 CUDA。"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.distributed.fsdp import MixedPrecisionPolicy

from examples import psm_wma_local_fsdp_scan_micro_smoke as micro


def test_cli_preflight_is_cpu_only_and_writes_schema(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: (_ for _ in ()).throw(AssertionError("CUDA called")))
    output = tmp_path / "preflight"
    assert micro.main(["--preflight", "--output", str(output)]) == 0
    result = json.loads((output / "result.json").read_text())
    assert result["status"] == "PASS" and result["preflight"] is True
    assert result["local_parameter_count"] == 165_312
    assert result["model"] == "TinyLocalOwner"
    assert result["parallel_dims"] == {
        "world_size": 1,
        "dp_shard": 1,
        "dp_replicate": 1,
        "cp": 1,
        "cfgp": 1,
        "enable_inference_mode": False,
        "dp_enabled": True,
    }
    assert result["process_group_destroyed"] is False
    assert json.loads((output / "cuda_memory_trace.json").read_text()) == []


def test_output_must_be_unique_and_failure_has_separate_json(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    (output / "original.txt").write_text("untouched")
    assert micro.main(["--preflight", "--output", str(output)]) == 1
    assert sorted(path.name for path in output.iterdir()) == ["original.txt"]
    rejected = list(tmp_path.glob("existing.rejected_*/result.json"))
    assert len(rejected) == 1
    result = json.loads(rejected[0].read_text())
    assert result["status"] == "FAIL"
    assert result["error"]["type"] == "FileExistsError"


@pytest.mark.parametrize("key,value", [("WORLD_SIZE", "2"), ("RANK", "1"), ("LOCAL_RANK", "1")])
def test_rank_guard_rejects_before_cuda_or_pair_lookup(tmp_path, monkeypatch, key, value):
    monkeypatch.setenv(key, value)
    monkeypatch.setattr(micro, "lock_implementation_pair", lambda: (_ for _ in ()).throw(AssertionError("pair")))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: (_ for _ in ()).throw(AssertionError("CUDA")))
    with pytest.raises(ValueError, match=key):
        micro.run_gpu(tmp_path, {}, [])


def test_tiny_owner_uses_exact_model_owned_runtime_and_frozen_dimensions():
    owner = micro.TinyLocalOwner()
    runtime = owner.local_memory_runtime
    assert micro.local_inventory(owner) == 165_312
    assert runtime.encoder.evidence_dim == 256
    assert runtime.encoder.action_proj.in_features == 15
    assert runtime.core.local_dim == 32 and runtime.core.ttt_dim == 64
    assert runtime.core.fast_hidden_dim == 256 and runtime.core.inner_lr == 0.1
    assert runtime.core.ttt_tbptt_steps == 16 and runtime.core.k_local == 4
    assert owner.local_memory2llm.weight.shape == (2048, 32)
    assert owner.local_memory_modality_embed.shape == (2048,)
    visual, action, valid = micro.inputs(torch.device("cpu"))
    assert visual.shape == (1, 16, 96) and visual.dtype == torch.float32
    assert action.shape == (1, 16, 15) and action.dtype == torch.float32
    assert valid.tolist() == [[False] + [True] * 15]
    assert owner.scan_local_memory.__self__ is owner
    tokens, state, present = owner.scan_local_memory(visual, action, valid, None)
    assert tokens.shape == (1, 16, 4, 32) and torch.equal(present, valid)
    assert all(value.dtype == torch.float32 and not isinstance(value, torch.nn.Parameter) for value in state)


def test_fsdp_recipe_uses_training_mesh_and_bf16_fp32_policy(monkeypatch):
    owner = micro.TinyLocalOwner()
    dims = micro.make_parallel_dims()
    assert dims.dp_enabled and not dims.enable_inference_mode
    calls = []
    mesh = object()
    monkeypatch.setattr(micro, "fsdp_mesh", lambda value: mesh if value is dims else None)

    def shard(module, *, mesh, mp_policy):
        calls.append(("shard", module, mesh, mp_policy))
        return module

    monkeypatch.setattr(micro, "fully_shard", shard)
    monkeypatch.setattr(
        micro, "register_fsdp_forward_method", lambda module, name: calls.append(("register", module, name))
    )
    assert micro.wrap_local_owner(owner, dims) is owner
    assert len(calls) == 2 and calls[0][:3] == ("shard", owner, mesh)
    policy = calls[0][3]
    assert isinstance(policy, MixedPrecisionPolicy)
    assert policy.param_dtype == torch.bfloat16
    assert policy.reduce_dtype == torch.float32
    assert policy.cast_forward_inputs is False
    assert calls[1] == ("register", owner, "scan_local_memory")
    assert owner._local_memory_scan_fsdp_registered is True
    assert 'dims.build_meshes("cuda")' in inspect.getsource(micro.run_gpu)


def test_script_has_no_full_model_assets_or_forbidden_shortcuts():
    source = Path(micro.__file__).read_text()
    for forbidden in (
        "Cosmos3-Edge",
        "RoboCasa",
        "Wan2.2",
        "FileSystemReader",
        "to_local(",
        "ignored_params",
        "inference_mode(",
        "no_grad(",
    ):
        assert forbidden not in source
    assert "LocalMemoryRuntime()" in source
    assert 'register_fsdp_forward_method(owner, "scan_local_memory")' in source


def test_pair_lock_requires_root_environment(monkeypatch):
    monkeypatch.delenv(micro.ROOT_ENV, raising=False)
    with pytest.raises(ValueError, match=micro.ROOT_ENV):
        micro.lock_implementation_pair()


def test_pair_lock_checks_root_gitlink_and_clean_child(tmp_path, monkeypatch):
    monkeypatch.setenv(micro.ROOT_ENV, str(tmp_path))
    outputs = {
        (str(tmp_path), "rev-parse"): "rootsha",
        (str(Path(micro.__file__).resolve().parents[1]), "rev-parse"): "childsha",
        (str(tmp_path), "ls-tree"): "160000 commit childsha\tcosmos-framework",
        (str(Path(micro.__file__).resolve().parents[1]), "status"): "",
    }

    def git(command, **kwargs):
        return SimpleNamespace(stdout=outputs[(command[2], command[3])])

    monkeypatch.setattr(micro.subprocess, "run", git)
    assert micro.lock_implementation_pair() == {"root": "rootsha", "child": "childsha", "gitlink": "childsha"}
    outputs[(str(tmp_path), "ls-tree")] = "160000 commit wrong\tcosmos-framework"
    with pytest.raises(ValueError, match="Gitlink"):
        micro.lock_implementation_pair()


def test_runtime_world_size_failure_writes_json_without_cuda(tmp_path, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: (_ for _ in ()).throw(AssertionError("CUDA called")))
    output = tmp_path / "world_failure"
    assert micro.main(["--output", str(output)]) == 1
    result = json.loads((output / "result.json").read_text())
    assert result["status"] == "FAIL"
    assert result["error"]["type"] == "ValueError" and "WORLD_SIZE" in result["error"]["message"]
    assert result["process_group_destroyed"] is False
    assert json.loads((output / "cuda_memory_trace.json").read_text()) == []
