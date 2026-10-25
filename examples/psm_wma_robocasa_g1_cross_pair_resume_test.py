"""CPU-only Owner G1 override; synthetic files, real Torch and unchanged restore AST.

Frozen YAML SHA is patched only to the tiny test fixture. No real G1/DCP/model.
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import importlib.util
import json
import os
import pickle
import random
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from shutil import copytree
from types import ModuleType
from types import SimpleNamespace as NS

import pytest
import torch
from torch import nn

from examples import psm_wma_robocasa_corrected_phase5 as phase5
from examples import psm_wma_robocasa_phase5_f0_test as f0

CHILD = Path(__file__).resolve().parents[1]
RAW_DIGEST = hashlib.sha256(b"synthetic phase5 digest, not real G1 evidence").hexdigest()


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _local_types():
    """Execute unchanged CPU classes/functions, isolate only heavy import wiring."""
    segment = _load("g1_cpu_segment", CHILD / "cosmos_framework/model/generator/mot/local_memory_segment.py")
    evidence = _load("g1_cpu_evidence", CHILD / "cosmos_framework/model/generator/mot/local_evidence.py")
    module = ModuleType("g1_cpu_restore_types")
    module.__dict__.update(
        dataclass=dataclasses.dataclass,
        torch=torch,
        nn=nn,
        hashlib=hashlib,
        json=json,
        random=random,
        deepcopy=deepcopy,
        Callback=object,
        **{n: getattr(segment, n) for n in ("SegmentIdentity", "SegmentProvenance", "RankLocalSegmentScheduler")},
        **{n: getattr(evidence, n) for n in ("ContinualTTTFastState", "ContinualTTTLocalMemoryCore")},
    )
    sys.modules[module.__name__] = module
    for relative, names in [
        (
            "model/generator/mot/robocasa_exact_window_local.py",
            {
                "_positive",
                "_digest",
                "ExactWindowSlotFrontier",
                "ExactWindowCatalogFrontier",
                "ExactWindowRankPlanner",
                "ExactWindowSegmentRequest",
                "ExactWindowGroupedPlan",
            },
        ),
        ("model/generator/mot/local_memory_segment_adapter.py", {"LocalMemorySegmentSidecar"}),
        ("model/generator/mot/local_memory_grouped_window.py", {"GroupedLiveState"}),
        (
            "trainer/local_memory_grouped_resume.py",
            {
                "_profile",
                "snapshot_grouped_local_state",
                "restore_grouped_local_state",
                "GroupedLocalMemoryStateCallback",
            },
        ),
    ]:
        path = CHILD / "cosmos_framework" / relative
        nodes = [
            n
            for n in ast.parse(path.read_text()).body
            if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names
        ]
        assert {n.name for n in nodes} == names
        future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
        exec(
            compile(ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[])), str(path), "exec"),
            module.__dict__,
        )
    module.FORMAT = "psm_v3_corrected_grouped_local_v2"
    module.Encoder = evidence.LocalEvidenceEncoder
    return module


LOCAL = _local_types()


class CPUWindow:
    plan = None
    _candidate = None

    def __init__(self, model, planner):
        self.model, self.planner = model, planner
        self._live = LOCAL.GroupedLiveState(
            planner.initial_frontier(), LOCAL.LocalMemorySegmentSidecar(), LOCAL.RankLocalSegmentScheduler()
        )

    @property
    def live(self):
        return self._live


def _state(rank, digest):
    core = LOCAL.ContinualTTTLocalMemoryCore(evidence_dim=6, ttt_dim=4, fast_hidden_dim=8, local_dim=3, k_local=4)
    model = NS(net=NS(local_memory_runtime=NS(encoder=LOCAL.Encoder(evidence_dim=6), core=core)))
    episodes, index = [], 0
    while len(episodes) < 8:
        uid = f"rank{rank}_episode{index}"
        if int(hashlib.sha256(uid.encode()).hexdigest(), 16) % 8 == rank:
            episodes.append(
                NS(uid=uid, task_class="SyntheticTask", source_digest="source", segment_count=4, window_count=64)
            )
        index += 1
    catalog = NS(
        episodes=tuple(episodes),
        task_classes=("SyntheticTask",),
        manifest_digest="manifest",
        cache_corpus_digest="corpus",
        source_binding_digest="binding",
        ttt_tbptt_steps=16,
    )
    planner = LOCAL.ExactWindowRankPlanner(catalog, rank=rank, world_size=8, b_stream=8, active_ga=2)
    window = CPUWindow(model, planner)
    slots, committed, saved = [], {}, {}
    for index, episode in enumerate(episodes):
        slot = rank * 8 + index
        identity = LOCAL.SegmentIdentity(slot, episode.uid, episode.task_class, 0, 0, episode.source_digest)
        provenance = LOCAL.SegmentProvenance("manifest", digest, "source", 0)
        values = LOCAL.ContinualTTTFastState(*(v.detach().clone() + rank + index for v in core.initial_state(1)))
        committed[slot], saved[slot] = identity, (identity, provenance, values)
        slots.append(LOCAL.ExactWindowSlotFrontier(episode.uid, 0, 1, 1))
    frontier = LOCAL.ExactWindowCatalogFrontier(0, tuple(sorted(e.uid for e in episodes)), tuple(slots))
    window._live = LOCAL.GroupedLiveState(
        frontier, LOCAL.LocalMemorySegmentSidecar(), LOCAL.RankLocalSegmentScheduler()
    )
    window.live.sidecar._records, window.live.scheduler._committed = saved, committed
    return window, LOCAL.snapshot_grouped_local_state(window, iteration=8900, config_digest=digest)


def _witness(path):
    return {
        str(p.relative_to(path)): (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
        for p in path.rglob("*")
        if p.is_file()
    }


@pytest.fixture
def job(tmp_path, monkeypatch):
    args = f0._args(tmp_path)
    args.phase, args.job_name, args.formal_execution_until = "resume", phase5.G1_JOB, 9000
    args.owner_authorized_g1_cross_pair_resume = True
    path = phase5.job_namespace(args)
    path.mkdir(parents=True)
    config = f0._config(args)
    old_yaml = {"dataloader_train": deepcopy(config.dataloader_train)}
    old_yaml["dataloader_train"]["dataloader"]["datasets"]["robocasa"]["dataset"]["chunk_length"] = 32
    (path / "config.yaml").write_text(json.dumps(old_yaml))
    (path / "config.pkl").write_bytes(b"tiny frozen config, never unpickled by code")
    hashes = {n: phase5._file_sha256(path / n) for n in ("config.yaml", "config.pkl")}
    monkeypatch.setattr(phase5, "G1_CONFIG_YAML_SHA256", hashes["config.yaml"])
    runtime = {
        "root": args.expected_root,
        "child": args.expected_child,
        "native_batch_layout": phase5.NATIVE_BATCH_LAYOUT,
        "training": {
            "model": {"mask": "raw15", "precision": "bf16", "tokenizer": {"duration": 17}},
            "optimizer": {"lr": 5e-5},
            "scheduler": {"cycle_lengths": [30000], "warm_up_steps": [500]},
            "trainer": {"seed": 0},
            "model_parallel": {"world_size": 8},
            "data_setting": {},
        },
        "checkpoint": {"save_iter": 100, "strict_resume": True},
        "contract": dict(t=16, b=8, ga=2, k=4, world_size=8, h_pred=16, raw_action_dim=15, state_dim=15),
        "authority": {"cache": "manifest", "corpus": "corpus", "source": "binding"},
        "source_paths": {"source": str(tmp_path / "source")},
        "rng": {"seed_by_rank": True, "deterministic_algorithms": False},
    }
    frozen = {**deepcopy(runtime), "root": phase5.G1_FROZEN_ROOT, "child": phase5.G1_FROZEN_CHILD}
    record = {
        "schema": "corrected_phase5_f0_v1",
        "namespace": str(path),
        "root": phase5.G1_FROZEN_ROOT,
        "child": phase5.G1_FROZEN_CHILD,
        "native_batch_layout": phase5.NATIVE_BATCH_LAYOUT,
        "semantic": frozen,
        "semantic_digest": phase5._json_digest(frozen),
        "diagnostic_only_do_not_resume": False,
        "canonical_config_sha256": hashes,
    }
    (path / phase5.JOB_IDENTITY).write_text(json.dumps(record))
    digest = phase5._json_digest({"phase5_digest": RAW_DIGEST, "training_identity": frozen})
    launch = path / "launches/frozen_fresh"
    launch.mkdir(parents=True)
    for name in hashes:
        (launch / name).write_bytes((path / name).read_bytes())
    (launch / "launch.json").write_text(
        json.dumps({"phase": "fresh", "semantic_digest": record["semantic_digest"], "config_digest": digest})
    )
    checkpoint = path / "checkpoints" / phase5.G1_CHECKPOINT
    for component in ("model", "optim", "scheduler", "trainer"):
        folder = checkpoint / component
        folder.mkdir(parents=True)
        (folder / ".metadata").write_bytes(b"synthetic metadata, not DCP")
    (checkpoint / "dataloader").mkdir()
    windows, states = [], []
    for rank in range(8):
        window, state = _state(rank, digest)
        windows.append(window)
        states.append(state)
        with (checkpoint / "dataloader" / f"rank_{rank}.pkl").open("wb") as stream:
            pickle.dump(state, stream)
    (checkpoint.parent / "latest_checkpoint.txt").write_text(phase5.G1_CHECKPOINT)
    (path / "launch.lock").touch()
    report = f0._report(args, runtime)
    report.update(
        checkpoint_load_path=str(checkpoint),
        cache_manifest_sha256="manifest",
        cache_corpus_digest="corpus",
        source_binding_digest="binding",
        lock={
            "root": args.expected_root,
            "child": args.expected_child,
            "gitlink": args.expected_child,
            "allowed_root_noncode_changes": 0,
        },
    )
    return NS(
        args=args,
        path=path,
        checkpoint=checkpoint,
        report=report,
        record=record,
        digest=digest,
        windows=windows,
        states=states,
        config=config,
        launch=launch,
    )


def test_explicit_g1_namespace_allows_only_source_pair_change(job):
    before = _witness(job.path)
    assert phase5.check_job_namespace(job.args, job.report["training_identity"]) == job.record
    assert _witness(job.path) == before


def test_historical_digest_matches_launch_all_ranks_and_original_local_restore(job):
    before = _witness(job.path)
    audit = phase5._g1_resume_compatibility(job.args, job.report, RAW_DIGEST)
    assert audit["config_digest"] == job.digest and len(audit["rank_sidecars"]) == 8
    assert audit["runtime_source"]["root"] != audit["frozen_source"]["root"]
    new_digest = phase5._json_digest(
        {"phase5_digest": RAW_DIGEST, "training_identity": job.report["training_identity"]}
    )
    assert new_digest != job.digest
    for old, state in zip(job.windows, job.states, strict=True):
        restored = CPUWindow(old.model, old.planner)
        initial = restored.live
        with pytest.raises(ValueError):
            LOCAL.restore_grouped_local_state(restored, state, iteration=8900, config_digest=new_digest)
        assert restored.live is initial
        LOCAL.restore_grouped_local_state(restored, state, iteration=8900, config_digest=audit["config_digest"])
        assert restored.live.frontier == state["frontier"] and restored.live.scheduler._committed == state["scheduler"]
        for identity, provenance, fast in restored.live.sidecar.snapshot():
            assert provenance.config_digest == job.digest
            for value, expected in zip(fast, state["sidecar"][identity.slot_id][2], strict=True):
                torch.testing.assert_close(value, expected)
        assert (
            restored.planner.plan_window(restored.live.frontier).members
            == old.planner.plan_window(old.live.frontier).members
        )
    assert _witness(job.path) == before
    print("G1_COMPATIBILITY_WITNESS " + json.dumps(audit, sort_keys=True))


def test_named_g1_resume_without_legacy_owner_flag_or_current_root_pin(job):
    # The Owner's G1 standing recovery rule is not tied to a one-off
    # cross-pair flag, nor to the caller's stale expected Root HEAD.
    job.args.owner_authorized_g1_cross_pair_resume = False
    job.args.expected_root = "OLD_HEAD_NOT_REQUIRED"
    assert phase5._owner_g1_cross_pair_resume(job.args) is True
    assert phase5.check_job_namespace(job.args, job.report["training_identity"]) == job.record
    # Report identity must bind the *observed* runtime Root, not the
    # deliberately stale CLI string.
    report = deepcopy(job.report)
    report["lock"]["root"] = "c" * 40
    identity = deepcopy(report["training_identity"])
    identity["root"] = "c" * 40
    report["training_identity"] = identity
    assert phase5._g1_resume_compatibility(job.args, report, RAW_DIGEST)["runtime_source"]["root"] == "c" * 40


@pytest.mark.parametrize("iteration", [8900, 9000, 12000])
def test_g1_repeated_resume_uses_latest_checkpoint_and_rank_iterations(job, iteration):
    # Synthetic component presence and 8rank Local state. This proves
    # admission is no longer restricted to iter8900, not real CUDA DCP load.
    job.args.owner_authorized_g1_cross_pair_resume = False
    checkpoint = job.checkpoint
    if iteration != 8900:
        checkpoint = checkpoint.parent / f"iter_{iteration:09d}"
        copytree(job.checkpoint, checkpoint)
        for rank in range(8):
            sidecar = checkpoint / "dataloader" / f"rank_{rank}.pkl"
            state = pickle.loads(sidecar.read_bytes())
            state["iteration"] = iteration
            sidecar.write_bytes(pickle.dumps(state))
        (checkpoint.parent / "latest_checkpoint.txt").write_text(checkpoint.name)
    report = deepcopy(job.report)
    report["checkpoint_load_path"] = str(checkpoint)
    assert phase5._resume_checkpoint(job.args) == checkpoint
    audit = phase5._g1_resume_compatibility(job.args, report, RAW_DIGEST)
    assert audit["iteration"] == iteration
    assert audit["checkpoint"] == str(checkpoint)
    assert len(audit["rank_sidecars"]) == 8
    if iteration != 8900:
        damaged = checkpoint / "dataloader/rank_7.pkl"
        state = pickle.loads(damaged.read_bytes())
        state["iteration"] = 8900
        damaged.write_bytes(pickle.dumps(state))
        with pytest.raises(ValueError, match="rank7|Sidecar|sidecar|迭代"):
            phase5._g1_resume_compatibility(job.args, report, RAW_DIGEST)


@pytest.mark.parametrize("change", ["fresh", "other_job", "ga", "t", "b", "k", "world_size", "schedule"])
def test_override_scope_is_not_a_general_bypass(job, change):
    if change == "fresh":
        job.args.phase = "fresh"
    elif change == "other_job":
        job.args.job_name = "another_job"
    elif change == "schedule":
        job.args.warmup = 499
    else:
        setattr(job.args, change, getattr(job.args, change) + 1)
    before = _witness(job.path)
    with pytest.raises((ValueError, FileNotFoundError)):
        phase5.check_job_namespace(job.args, job.report["training_identity"])
    assert _witness(job.path) == before


@pytest.mark.parametrize(
    "section,key",
    [
        ("model", "mask"),
        ("model", "precision"),
        ("model", "tokenizer"),
        ("optimizer", "lr"),
        ("scheduler", "cycle_lengths"),
        ("trainer", "seed"),
        ("model_parallel", "world_size"),
        ("data_setting", "new"),
    ],
)
def test_training_semantic_drift_reports_exact_path(job, section, key):
    changed = deepcopy(job.report["training_identity"])
    changed["training"][section][key] = "changed"
    with pytest.raises(ValueError, match=f"semantic.training.{section}.{key}"):
        phase5.check_job_namespace(job.args, changed)


@pytest.mark.parametrize(
    "section,key",
    [
        ("contract", "h_pred"),
        ("contract", "ga"),
        ("contract", "b"),
        ("rng", "deterministic_algorithms"),
        ("authority", "cache"),
        ("authority", "source"),
        ("source_paths", "source"),
        ("checkpoint", "save_iter"),
    ],
)
def test_all_non_source_sections_remain_strict(job, section, key):
    changed = deepcopy(job.report["training_identity"])
    changed[section][key] = "changed"
    with pytest.raises(ValueError, match=f"semantic.{section}.{key}"):
        phase5.check_job_namespace(job.args, changed)


@pytest.mark.parametrize(
    "damage", ["root", "child", "semantic_root", "semantic_hash", "yaml", "pkl", "layout", "namespace", "diagnostic"]
)
def test_frozen_authority_damage_is_rejected(job, damage):
    record = deepcopy(job.record)
    if damage in ("yaml", "pkl"):
        (job.path / ("config." + damage)).write_bytes(b"changed")
    elif damage == "semantic_root":
        record["semantic"]["root"] = "c" * 40
        record["semantic_digest"] = phase5._json_digest(record["semantic"])
    elif damage == "semantic_hash":
        record["semantic_digest"] = "0" * 64
    else:
        record[{"layout": "native_batch_layout", "diagnostic": "diagnostic_only_do_not_resume"}.get(damage, damage)] = (
            "changed"
        )
    (job.path / phase5.JOB_IDENTITY).write_text(json.dumps(record))
    with pytest.raises(ValueError):
        phase5.check_job_namespace(job.args, job.report["training_identity"])


@pytest.mark.parametrize(
    "damage",
    [
        "phase5_digest",
        "launch_digest",
        "missing_launch",
        "launch_snapshot",
        "sidecar_digest",
        "provenance",
        "iteration",
        "rank",
        "cache",
        "missing_rank",
        "missing_optim",
        "wrong_checkpoint",
    ],
)
def test_digest_and_full_checkpoint_continuity_fail_closed(job, damage):
    raw = RAW_DIGEST
    if damage == "phase5_digest":
        raw = "0" * 64
    elif damage == "missing_launch":
        (job.launch / "launch.json").unlink()
    elif damage == "launch_snapshot":
        (job.launch / "config.yaml").write_text("changed")
    elif damage == "launch_digest":
        record = json.loads((job.launch / "launch.json").read_text())
        record["config_digest"] = "0" * 64
        (job.launch / "launch.json").write_text(json.dumps(record))
    elif damage == "missing_optim":
        (job.checkpoint / "optim/.metadata").unlink()
    elif damage == "missing_rank":
        (job.checkpoint / "dataloader/rank_7.pkl").unlink()
    elif damage == "wrong_checkpoint":
        (job.checkpoint.parent / "latest_checkpoint.txt").write_text("iter_000008901")
    else:
        state = deepcopy(job.states[7])
        if damage == "sidecar_digest":
            state["config_digest"] = "0" * 64
        if damage == "iteration":
            state["iteration"] = 8899
        if damage == "rank":
            state["profile"] = (0, *state["profile"][1:])
        if damage == "cache":
            state["cache_manifest_sha256"] = "changed"
        if damage == "provenance":
            key = next(iter(state["sidecar"]))
            identity, provenance, values = state["sidecar"][key]
            state["sidecar"][key] = identity, dataclasses.replace(provenance, config_digest="0" * 64), values
        with (job.checkpoint / "dataloader/rank_7.pkl").open("wb") as stream:
            pickle.dump(state, stream)
    before = _witness(job.path)
    with pytest.raises((ValueError, FileNotFoundError)):
        phase5._g1_resume_compatibility(job.args, job.report, raw)
    assert _witness(job.path) == before


@pytest.mark.parametrize("damage", ["cursor", "fast_nan", "fast_shape", "provenance"])
def test_original_restore_validation_is_not_disabled(job, damage):
    source = job.windows[0]
    window = CPUWindow(source.model, source.planner)
    state = deepcopy(job.states[0])
    key = next(iter(state["sidecar"]))
    if damage == "cursor":
        state["frontier"] = dataclasses.replace(
            state["frontier"],
            slots=(dataclasses.replace(state["frontier"].slots[0], cursor=2), *state["frontier"].slots[1:]),
        )
    else:
        identity, provenance, values = state["sidecar"][key]
        if damage == "provenance":
            provenance = dataclasses.replace(provenance, config_digest="0" * 64)
        elif damage == "fast_nan":
            values[0].flatten()[0] = float("nan")
        else:
            values = (values[0][..., :-1], *values[1:])
        state["sidecar"][key] = identity, provenance, values
    before = window.live
    with pytest.raises(ValueError):
        LOCAL.restore_grouped_local_state(window, state, iteration=8900, config_digest=job.digest)
    assert window.live is before


def test_append_only_launch_records_new_source_and_keeps_all_old_assets(job, monkeypatch):
    report = deepcopy(job.report)
    report["owner_cross_pair_resume"] = phase5._g1_resume_compatibility(job.args, report, RAW_DIGEST)
    report["config_digest"] = job.digest
    monkeypatch.setattr(phase5, "verify_root_child_lock", lambda args: report["lock"])
    before = _witness(job.path)
    writer = phase5.JobLifecycle(job.args, report)
    try:
        writer.write_config(job.config, f0.Serializer)
    finally:
        writer.close()
    after = _witness(job.path)
    assert all(after[name] == value for name, value in before.items())
    new = set(after) - set(before)
    assert new and all(name.startswith(f"launches/{writer.launch_id}/") for name in new)
    record = json.loads((job.path / "launches" / writer.launch_id / "launch.json").read_text())
    assert (
        record["owner_cross_pair_resume"] == report["owner_cross_pair_resume"] and record["config_digest"] == job.digest
    )
    assert (
        json.loads((job.path / "config.yaml").read_text())["dataloader_train"]["dataloader"]["datasets"]["robocasa"][
            "dataset"
        ]["chunk_length"]
        == 32
    )
    print(
        "G1_ASSET_IMMUTABILITY "
        + json.dumps(
            {"before": before, "after_original": {k: after[k] for k in before}, "new_launch": writer.launch_id},
            sort_keys=True,
        )
    )


def test_all_rank_readonly_agreement_uses_runtime_source_as_well_as_old_digest(job, monkeypatch):
    report = deepcopy(job.report)
    report["owner_cross_pair_resume"] = phase5._g1_resume_compatibility(job.args, report, RAW_DIGEST)
    report["config_digest"] = job.digest
    config = NS(validate=lambda: None, freeze=lambda: None)
    monkeypatch.setattr(phase5, "preflight", lambda args: (report, config, None, None))
    for corrupt in (False, True):
        calls = []

        def gather(output, value):
            calls.append(value)
            output[:] = [value] * 8
            if corrupt and len(calls) == 2:
                output[7] = "other-runtime-source"

        monkeypatch.setattr(phase5, "torch", NS(distributed=NS(all_gather_object=gather)), raising=False)
        if corrupt:
            with pytest.raises(ValueError, match="identities disagree"):
                phase5._agree_preflight(job.args)
        else:
            assert phase5._agree_preflight(job.args)[0]["config_digest"] == job.digest
        assert len(calls) == 2


def test_launcher_explicit_flag_only_reaches_named_resume_without_running_torch(tmp_path):
    fake = tmp_path / "capture-python"
    capture = tmp_path / "argv.json"
    fake.write_text(
        f"#!{sys.executable}\nimport json,sys\nfrom pathlib import Path\nPath({str(capture)!r}).write_text(json.dumps(sys.argv[1:]))\n"
    )
    fake.chmod(0o700)
    env = dict(os.environ, PYTHON=str(fake))
    for key in (
        "OUTPUT_ROOT",
        "SOURCE_ROOT",
        "CACHE_ROOT",
        "DATASET_INDEX_ROOT",
        "EDGE_POLICY_CHECKPOINT",
        "WAN_VAE_PATH",
        "BASE_CHECKPOINT_PATH",
        "EXPECTED_ROOT",
        "EXPECTED_CHILD",
    ):
        env[key] = "synthetic_" + key
    env["JOB_NAME"] = phase5.G1_JOB
    launcher = CHILD.parent / "scripts/train_v3_packed128_staged.sh"
    flag = "--owner-authorized-g1-cross-pair-resume"
    proc = subprocess.run(["bash", str(launcher), "resume", "9000", flag], env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert json.loads(capture.read_text()).count(flag) == 1

    # The permanent G1 recovery contract must work with neither a one-off
    # override flag nor an EXPECTED_ROOT environment pin.
    implicit = dict(env)
    implicit.pop("EXPECTED_ROOT")
    capture.unlink()
    resumed = subprocess.run(
        ["bash", str(launcher), "resume", "12000"], env=implicit, capture_output=True, text=True
    )
    assert resumed.returncode == 0, resumed.stderr
    argv = json.loads(capture.read_text())
    assert flag not in argv
    assert argv[argv.index("--expected-root") + 1] == "G1_ROOT_HEAD_NOT_PINNED"

    for argv, name in [
        (["fresh", "9000", flag], phase5.G1_JOB),
        (["resume", "9000", flag], "other"),
        (["resume", "9000", "--unknown"], phase5.G1_JOB),
    ]:
        capture.unlink(missing_ok=True)
        env["JOB_NAME"] = name
        assert subprocess.run(["bash", str(launcher), *argv], env=env, capture_output=True).returncode == 2
        assert not capture.exists()


@pytest.mark.parametrize(
    "fault",
    [
        "none",
        "wrong_root",
        "wrong_child",
        "wrong_gitlink",
        "dirty_child",
        "dirty_root_source",
        "wrong_lineage",
        "short_expected",
    ],
)
def test_runtime_source_lock_remains_strict(job, monkeypatch, fault):
    calls = []
    if fault == "short_expected":
        # Deliberately stale/short caller Root pin must not block G1.
        job.args.expected_root = "a" * 8

    def run(command, **kwargs):
        command = tuple(command)
        calls.append(command)
        repo, args = command[2], command[3:]
        if args == ("rev-parse", "HEAD"):
            value = job.args.expected_root if Path(repo) == job.args.root_worktree else job.args.expected_child
            if fault in ("wrong_root", "short_expected") and Path(repo) == job.args.root_worktree:
                value = "c" * 40
            if fault == "wrong_child" and Path(repo) != job.args.root_worktree:
                value = "c" * 40
            return NS(stdout=value + "\n")
        if args[:2] == ("ls-tree", "HEAD"):
            return NS(
                stdout="160000 commit "
                + ("d" * 40 if fault == "wrong_gitlink" else job.args.expected_child)
                + "\tcosmos-framework\n"
            )
        if args == ("status", "--porcelain"):
            return NS(stdout=" M examples/changed.py\n" if fault == "dirty_child" else "")
        if args[0] == "status":
            return NS(stdout=" M scripts/unsafe.py\0" if fault == "dirty_root_source" else "")
        assert args == ("merge-base", "--is-ancestor", phase5.G1_RUNTIME_BASE_CHILD, job.args.expected_child)
        if fault == "wrong_lineage":
            raise subprocess.CalledProcessError(1, command)
        return NS(stdout="")

    monkeypatch.setattr(phase5.subprocess, "run", run)
    if fault in {"none", "wrong_root", "short_expected"}:
        lock = phase5.verify_root_child_lock(job.args)
        assert lock["child"] == lock["gitlink"] == job.args.expected_child
        assert lock["runtime_base_child"] == phase5.G1_RUNTIME_BASE_CHILD
        if fault != "none":
            assert lock["root"] == "c" * 40
    else:
        with pytest.raises((ValueError, subprocess.CalledProcessError)):
            phase5.verify_root_child_lock(job.args)
    assert calls


@pytest.mark.parametrize("damage", ["sidecar_changed", "report_digest", "missing_audit"])
def test_launch_revalidates_continuity_before_any_snapshot_write(job, monkeypatch, damage):
    report = deepcopy(job.report)
    report["owner_cross_pair_resume"] = phase5._g1_resume_compatibility(job.args, report, RAW_DIGEST)
    report["config_digest"] = job.digest
    if damage == "sidecar_changed":
        state = deepcopy(job.states[0])
        state["sidecar"] = {}
        with (job.checkpoint / "dataloader/rank_0.pkl").open("wb") as stream:
            pickle.dump(state, stream)
    elif damage == "report_digest":
        report["config_digest"] = "0" * 64
    else:
        del report["owner_cross_pair_resume"]
    monkeypatch.setattr(phase5, "verify_root_child_lock", lambda args: report["lock"])
    before = _witness(job.path)
    writer = phase5.JobLifecycle(job.args, report)
    try:
        with pytest.raises((ValueError, KeyError)):
            writer.write_config(job.config, f0.Serializer)
    finally:
        writer.close()
    assert _witness(job.path) == before


def test_real_preflight_wires_historical_digest_to_report_and_snapshot_producer(job, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    args = job.args
    args.vae.parent.mkdir(exist_ok=True)
    args.vae.touch()
    args.snapshot10 = True
    config = NS(model=NS(config=NS(tokenizer={}, vlm_config=NS(tokenizer={}))))
    contract = NS(vae_encode_contract={})
    catalog = NS(
        manifest_digest="manifest",
        cache_corpus_digest="corpus",
        source_binding_digest="binding",
        raw=NS(contract=contract),
        episodes=("synthetic",),
    )
    dataset = NS(_dataset=NS(source_reader=NS(init_timings_ms={})))
    plan = NS(members=(tuple(range(10)),), candidate_frontier=None)
    producer_digests = []

    def producer(catalog, *, config_digest, **kwargs):
        producer_digests.append(config_digest)
        return NS(produce=lambda request: NS(consumer_payload=((dict(cached_latent_required=True),),)))

    boundaries = dict(
        _load_runtime=lambda: None,
        verify_root_child_lock=lambda args: job.report["lock"],
        check_edge_checkpoint=lambda path: None,
        check_droid_dcp=lambda path: None,
        _load_config=lambda args: config,
        VerifiedExactWindowIndex=NS(open=lambda *a, **kw: None),
        RoboCasaExactWindowCacheCatalog=lambda *a, **kw: NS(manifest_sha256="manifest"),
        CorrectedRoboCasaPolicyContract=NS(from_cache_catalog=lambda cache: contract),
        overlay_config=lambda *a, **kw: None,
        get_action_robocasa_exact_window_cached_sft_dataset=lambda **kw: dataset,
        ExactWindowLocalCatalog=lambda *a, **kw: catalog,
        _validate_runtime_tokenizer=lambda *a: None,
        model_witnesses=lambda *a: {},
        config_digest=lambda *a, **kw: RAW_DIGEST,
        ExactWindowRankPlanner=lambda *a, **kw: NS(initial_frontier=lambda: None, plan_window=lambda frontier: plan),
        ExactWindowSegmentProducer=producer,
        collate_grouped_native_batch=lambda *a: None,
        describe_effective_data_source=lambda *a, **kw: job.report["effective_data_source"],
        training_identity=lambda *a: job.report["training_identity"],
    )
    for name, value in boundaries.items():
        monkeypatch.setattr(phase5, name, value, raising=False)
    before = _witness(job.path)
    report, _, _, _ = phase5.preflight(args)
    assert report["config_digest"] == job.digest
    assert report["owner_cross_pair_resume"]["config_digest"] == job.digest
    assert producer_digests == [job.digest]
    assert _witness(job.path) == before


def test_original_callback_stages_state_without_clearing_validation(job):
    trainer = NS(_grouped_planner=object(), _grouped_config_digest=job.digest)
    callback = LOCAL.GroupedLocalMemoryStateCallback(trainer)
    callback.load_state_dict(job.states[0])
    assert trainer._pending_grouped_resume["config_digest"] == job.digest
    assert trainer._pending_grouped_resume is not job.states[0]
    window = CPUWindow(job.windows[0].model, job.windows[0].planner)
    LOCAL.restore_grouped_local_state(window, trainer._pending_grouped_resume, iteration=8900, config_digest=job.digest)
    assert window.live.frontier == job.states[0]["frontier"]
