# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json
from dataclasses import asdict, replace
from hashlib import sha256
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_model_state_dict
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor

from cosmos_framework.checkpoint.safe_local_witness import (
    SAFE_LOCAL_COMPLETE,
    SAFE_LOCAL_PROFILE,
    component_witness,
    file_witness,
)
from cosmos_framework.data.generator.action.utils.static_task_text import StaticTaskTextRecipe
from cosmos_framework.inference.g2_native_policy import NativeG2R1Authority
from cosmos_framework.model.generator.mot.g2_contracts import (
    G2SampleIdentity,
    GlobalMemoryConfig,
    g2_run_recipe_identity,
)


def _write(path, value):
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")))
    return file_witness(path)["sha256"]


@pytest.fixture
def training_job(tmp_path):
    model = torch.nn.Linear(6, 4, bias=False)
    model.weight.data.copy_(torch.arange(24, dtype=torch.float32).view(4, 6))
    checkpoint = tmp_path / "checkpoints" / "iter_000000100"
    model_path = checkpoint / "model"
    dcp.save(get_model_state_dict(model), checkpoint_id=model_path)

    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_bytes(b'{"stub": "local tokenizer bytes"}\n')
    text = StaticTaskTextRecipe(17, 20.0, 192, 320, file_witness(tokenizer)["sha256"])
    run_recipe = tmp_path / "g2-materialized.json"
    recipe = {
        "alignment_version": "g2_causal_latest_spatial_v1",
        "g1_config_path": "/unused/for/inference/frozen_g1.yaml",
        "text": asdict(text),
        "view_recipe_id": None,
        "training": {
            "profile": "R1",
            "world_size": 4,
            "text_recipe_identity": text.identity,
            "state15_recipe_identity": sha256(b"state15").hexdigest(),
        },
        "global_memory": asdict(GlobalMemoryConfig()),
    }
    recipe_sha = _write(run_recipe, recipe)
    config = tmp_path / "config.yaml"
    config.write_text("model: frozen\n")
    semantic = {
        "training_profile": "g2_R1",
        "g2_recipe_identity": g2_run_recipe_identity(recipe),
        "checkpoint": {"safety_profile": SAFE_LOCAL_PROFILE},
        "contract": {"raw_action_dim": 15, "state_dim": 15, "h_pred": 16, "consumer_frames": 17},
    }
    receipt = tmp_path / "v3_job_identity.json"
    marker = checkpoint / SAFE_LOCAL_COMPLETE
    model_receipt = component_witness(model_path)
    _write(
        marker,
        {
            "schema": SAFE_LOCAL_PROFILE,
            "complete": True,
            "checkpoint": checkpoint.name,
            "iteration": 100,
            "world_size": 4,
            "components": {"model": model_receipt},
        },
    )
    _write(
        receipt,
        {
            "schema": "corrected_phase5_f0_v1",
            "diagnostic_only_do_not_resume": False,
            "semantic": semantic,
            "semantic_digest": sha256(json.dumps(semantic, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "canonical_config_sha256": {"config.yaml": file_witness(config)["sha256"]},
        },
    )
    return SimpleNamespace(
        model=model,
        run_recipe=run_recipe,
        model_path=model_path,
        config=config,
        marker=marker,
        receipt=receipt,
        recipe_sha=recipe_sha,
        metadata_sha=model_receipt["metadata"]["sha256"],
        text=text,
        tokenizer=tokenizer,
    )


def _open(job, **overrides):
    arguments = {
        "run_recipe": job.run_recipe,
        "model_path": job.model_path,
        "config_path": job.config,
        "metadata_sha256": job.metadata_sha,
    }
    arguments.update(overrides)
    return NativeG2R1Authority.open(**arguments)


def test_native_r1_pins_identities_and_reuses_the_two_native_text_stages(training_job):
    job = training_job
    bound = _open(job)
    tokenizer = object()
    model = job.model.eval()
    model.vlm_tokenizer = tokenizer
    model.vlm_processor = SimpleNamespace(
        tokenizer=tokenizer, name=str(job.tokenizer.parent), tokenize_text=lambda *a, **kw: [2, 3]
    )
    model.config = SimpleNamespace(
        g2_text_enabled=True,
        global_memory_enabled=False,
        diffusion_expert_config=SimpleNamespace(enable_fps_modulation=True),
    )
    model.net = SimpleNamespace(
        config=SimpleNamespace(g2_text_enabled=True, global_memory_enabled=False, local_memory_enabled=True)
    )
    model.precision = torch.float32
    model.prepare_g2_text = lambda _: "existing native UND"
    model.prepare_g2_gen_text_kv = lambda _: "existing native GEN"

    hooks = bound.hooks(model)

    assert hooks.model is model and hooks.use_float_positions
    assert hooks.prepare_text(None) == "existing native UND"
    assert hooks.prepare_gen_text_kv(None) == "existing native GEN"
    assert hooks.versions["reader_version"] == "disabled:r1"
    assert hooks.versions["tokenizer_identity"] == job.text.tokenizer_identity
    assert bound.versions["run_recipe_sha256"] == job.recipe_sha
    assert bound.versions["checkpoint_metadata_sha256"] == job.metadata_sha
    assert bound.text_recipe.identity == job.text.identity
    assert bound.text_recipe.identity != replace(job.text, fps=20).identity

    sample = G2SampleIdentity(
        episode_id="fixture",
        scene_id="scene",
        subtask_id="PickPlace",
        decision_frame=16,
        logical_state_frame=0,
        data_version="real_source_version",
        alignment_version=bound.alignment_version,
        state15_recipe_id=bound.state15_recipe_id,
        task_binding_id="binding",
    )
    bound.validate_sample(sample, frames=17, fps=20, height=192, width=320)
    bound.validate_state([0] * 14 + [1.0])
    with pytest.raises(ValueError, match="State15/alignment"):
        bound.validate_sample(replace(sample, state15_recipe_id="bad"), frames=17, fps=20, height=192, width=320)
    with pytest.raises(ValueError, match="canvas/FPS"):
        bound.validate_sample(sample, frames=17, fps=20, height=224, width=320)
    for bad in (None, [[0] * 15], [False] * 15, [float("nan")] * 15, [1e50] * 15, [10**400] * 15):
        with pytest.raises(ValueError, match="State15"):
            bound.validate_state(bad)

    model.net.global_memory_reader = torch.nn.Linear(1, 1)
    with pytest.raises(ValueError, match="Global disabled"):
        bound.hooks(model)
    del model.net.global_memory_reader
    job.tokenizer.write_text("changed")
    with pytest.raises(ValueError, match="tokenizer"):
        bound.hooks(model)


@pytest.mark.parametrize(
    "failure, message",
    [
        ("metadata_pin", "pinned model metadata"),
        ("metadata_bytes", "publication witness"),
        ("config", "canonical training manifest"),
        ("checkpoint_marker", "pinned model metadata"),
        ("diagnostic", "formal training job"),
        ("recipe", "formal training job"),
        ("shard", "truncated DCP shard"),
    ],
)
def test_native_r1_rejects_incompatible_or_incomplete_checkpoint(training_job, failure, message):
    job = training_job
    kwargs = {}
    if failure == "metadata_pin":
        kwargs["metadata_sha256"] = "a" * 64
    elif failure == "metadata_bytes":
        (job.model_path / ".metadata").write_bytes(b"invalid pickle that must not be opened")
    elif failure == "config":
        job.config.write_text("model: edited\n")
    elif failure == "checkpoint_marker":
        record = json.loads(job.marker.read_text())
        record["complete"] = False
        _write(job.marker, record)
    elif failure == "diagnostic":
        record = json.loads(job.receipt.read_text())
        record["diagnostic_only_do_not_resume"] = True
        _write(job.receipt, record)
    elif failure == "recipe":
        recipe = json.loads(job.run_recipe.read_text())
        recipe["alignment_version"] = "wrong"
        _write(job.run_recipe, recipe)
    else:
        next(job.model_path.glob("*.distcp")).write_bytes(b"truncated")
    with pytest.raises(ValueError, match=message):
        _open(job, **kwargs)


def test_native_r1_checks_model_inputs_again_after_loading(training_job):
    bound = _open(training_job)
    shard = next(training_job.model_path.glob("*.distcp"))
    content = shard.read_bytes()
    replacement = shard.with_suffix(".replacement")
    replacement.write_bytes(content)
    replacement.replace(shard)
    with pytest.raises(ValueError, match="changed during startup"):
        bound.check_unchanged()


def _save_four_rank_model(rank, checkpoint, rendezvous):
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", world_size=4, rank=rank)
    try:
        mesh = init_device_mesh("cpu", (4,))
        model = torch.nn.Linear(6, 4, bias=False)
        model.weight = torch.nn.Parameter(
            distribute_tensor(torch.arange(24, dtype=torch.float32).view(4, 6), mesh, [Shard(0)])
        )
        dcp.save(get_model_state_dict(model), checkpoint_id=checkpoint)
    finally:
        dist.destroy_process_group()


def test_four_rank_dtensor_model_dcp_loads_as_single_rank_full_tensor(tmp_path):
    checkpoint = tmp_path / "four_rank_model"
    torch.multiprocessing.start_processes(
        _save_four_rank_model, args=(checkpoint, tmp_path / "rendezvous"), nprocs=4, start_method="fork"
    )
    assert len(component_witness(checkpoint)["files"]) == 4
    dist.init_process_group("gloo", init_method=f"file://{tmp_path / 'single'}", world_size=1, rank=0)
    try:
        model = torch.nn.Linear(6, 4, bias=False)
        dcp.load(get_model_state_dict(model), storage_reader=dcp.FileSystemReader(checkpoint))
        torch.testing.assert_close(model.weight, torch.arange(24, dtype=torch.float32).view(4, 6), rtol=0, atol=0)
    finally:
        dist.destroy_process_group()
