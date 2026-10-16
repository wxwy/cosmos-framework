"""Real CPU DCP/Gloo + synthetic tiny tensors. No model assets, CUDA or training.

Run directly or with pytest --noconftest and an external empty pytest config.
Only Cosmos config/log/object-store imports are isolated; Torch/DCP/NumPy are real.
CUDA RNG bytes are an explicit CPU stand-in, not CUDA correctness evidence.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import pickle
import random
import subprocess
import sys
import tempfile
import time
import traceback
import unittest
from contextlib import ExitStack
from datetime import timedelta
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace as NS
from unittest.mock import MagicMock, patch

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp

CHILD = Path(__file__).resolve().parents[2]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class LocalIO:
    @staticmethod
    def exists(path, **kw):
        return Path(path).exists()

    @staticmethod
    def load(path, **kw):
        return pickle.loads(Path(path).read_bytes()) if str(path).endswith(".pkl") else Path(path).read_text()

    @staticmethod
    def dump(value, path, **kw):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(pickle.dumps(value) if kw.get("file_format") == "pkl" else value.encode())


def _modules():
    modules = {}

    def mod(name, **values):
        module = ModuleType(name)
        module.__dict__.update(values)
        modules[name] = module
        return module

    mod("cosmos_framework", __path__=[str(CHILD / "cosmos_framework")])
    mod("cosmos_framework.checkpoint", __path__=[str(CHILD / "cosmos_framework/checkpoint")])
    mod("cosmos_framework.model", __path__=[])
    mod("cosmos_framework.model._base", ImaginaireModel=torch.nn.Module)
    mod(
        "cosmos_framework.utils",
        __path__=[],
        callback=NS(CallBackGroup=object),
        distributed=NS(is_rank0=lambda: dist.get_rank() == 0),
        log=MagicMock(),
        misc=NS(timer=lambda *args, **kw: lambda fn: fn),
    )
    mod("cosmos_framework.utils.config", CheckpointConfig=NS, JobConfig=NS, ObjectStoreConfig=NS)
    mod("cosmos_framework.utils.flags", INTERNAL=True)
    mod("cosmos_framework.utils.easy_io", easy_io=LocalIO)
    mod("cosmos_framework.utils.generator", __path__=[str(CHILD / "cosmos_framework/utils/generator")])
    mod(
        "cosmos_framework.checkpoint.s3_filesystem",
        S3StorageReader=type("S3Reader", (), {}),
        S3StorageWriter=type("S3Writer", (), {}),
    )
    # Import real rand packing and real background rendezvous modules under the scoped package map.
    with patch.dict(sys.modules, modules):
        rand = _load(
            "cosmos_framework.utils.generator.rand_state", CHILD / "cosmos_framework/utils/generator/rand_state.py"
        )
        _load("cosmos_framework.checkpoint.background_store", CHILD / "cosmos_framework/checkpoint/background_store.py")
        _load("cosmos_framework.checkpoint.base", CHILD / "cosmos_framework/checkpoint/base.py")
        module = _load("cosmos_framework.checkpoint.dcp", CHILD / "cosmos_framework/checkpoint/dcp.py")
    # Production uses CUDA RNG on GPU hosts. Tests substitute only its external get/set bytes.
    cuda_bytes = torch.arange(32, dtype=torch.uint8)

    def get_rng():
        with patch.object(torch.cuda, "get_rng_state", return_value=cuda_bytes.clone()):
            return rand.get_rand_state_dict()

    def set_rng(state):
        nonlocal cuda_bytes

        def set_cuda(value):
            nonlocal cuda_bytes
            if value.numel() != 32 or value.dtype != torch.uint8:
                raise ValueError("synthetic CUDA byte shape")
            cuda_bytes = value.clone()

        with patch.object(torch.cuda, "set_rng_state", side_effect=set_cuda):
            rand.set_rand_state_dict(state)

    module.get_rand_state_dict = get_rng
    module.set_rand_state_dict = set_rng
    return module


class LocalState:
    checkpoint_component = "dataloader"

    def has_checkpoint_state(self):
        return True

    def state_dict(self):
        return {"rank": dist.get_rank(), "local_fast": torch.tensor([dist.get_rank(), 5.0])}

    def load_state_dict(self, state):
        self.restored = state


class Callbacks:
    def __init__(self):
        self._callbacks = [LocalState()]
        self.successes = []

    def on_save_checkpoint_start(self, *args):
        pass

    def on_save_checkpoint(self, *args, **kw):
        pass

    def on_save_checkpoint_success(self, **kw):
        self.successes.append(kw["iteration"])

    def on_save_checkpoint_end(self, **kw):
        pass

    def on_load_checkpoint_start(self, *args):
        pass

    def on_load_checkpoint(self, *args, **kw):
        pass

    def on_load_checkpoint_end(self, *args, **kw):
        pass


def _configuration(root, **overrides):
    store = NS(enabled=False, credentials=None, bucket=None)
    values = dict(
        save_to_object_store=store,
        load_from_object_store=store,
        strict_resume=True,
        load_path=None,
        load_training_state=False,
        only_load_scheduler_state=False,
        verbose=False,
        keys_not_to_resume=[],
        keys_to_skip_loading=[],
        broadcast_via_filesystem=False,
        load_ema_to_reg=False,
        dcp_async_mode_enabled=False,
        dcp_load_dedup=False,
    )
    values.update(overrides)
    return NS(**values), NS(path_local=str(root), path="synthetic/f1")


def _objects():
    torch.manual_seed(91)
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    # Initialize optimizer state without calling a training/optimizer update.
    for parameter in model.parameters():
        optimizer.state[parameter] = {
            "step": torch.tensor(1.0),
            "exp_avg": torch.zeros_like(parameter),
            "exp_avg_sq": torch.zeros_like(parameter),
        }
    return (
        model,
        optimizer,
        torch.optim.lr_scheduler.StepLR(optimizer, step_size=10),
        NS(state_dict=lambda: {}, load_state_dict=lambda state: None),
    )


def _sample():
    return [random.random(), float(np.random.random()), float(torch.rand(()))]


def _reader(root):
    """Independent read-only process: standard files/JSON + no Checkpointer helpers."""
    code = """import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); name=(root/'latest_checkpoint.txt').read_text().strip(); checkpoint=root/name
marker=json.loads((checkpoint/'complete.v1.json').read_text())
assert marker['complete'] and marker['checkpoint']==name
for rank,item in marker['sidecars'].items():
 p=checkpoint/'dataloader'/('rank_'+rank+'.pkl'); b=p.read_bytes()
 assert len(b)==item['size'] and hashlib.sha256(b).hexdigest()==item['sha256']
for component,item in marker['components'].items():
 p=checkpoint/component; b=(p/'.metadata').read_bytes()
 assert hashlib.sha256(b).hexdigest()==item['metadata']['sha256']
 for name,size in item['files'].items(): assert (p/name).stat().st_size==size
print(json.dumps({'checkpoint':marker['checkpoint'],'world_size':marker['world_size'],'verified':True}))
"""
    result = subprocess.run([sys.executable, "-c", code, str(root)], capture_output=True, text=True, timeout=15)
    if result.returncode:
        raise AssertionError(result.stderr)
    return json.loads(result.stdout)


def _worker(rank, root, init_file, scenario, event, release):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method="file://" + init_file, rank=rank, world_size=2, timeout=timedelta(seconds=20)
    )
    try:
        module = _modules()
        config, job = _configuration(Path(root))
        callbacks = Callbacks()
        saver = module.SafeLocalDistributedCheckpointer(config, job, callbacks, coordination_timeout_seconds=10)
        objects = _objects()
        random.seed(40 + rank)
        np.random.seed(40 + rank)
        torch.manual_seed(40 + rank)
        if scenario != "first_failure":
            saver.save(*objects, iteration=1)
        previous = (
            (Path(saver.save_dirname) / "latest_checkpoint.txt").read_bytes() if scenario != "first_failure" else None
        )
        if scenario == "resume_success":
            expected_model = {key: value.clone() for key, value in objects[0].state_dict().items()}
            expected_scheduler = copy.deepcopy(objects[2].state_dict())
            expected_draw = _sample()
            with torch.no_grad():
                for parameter in objects[0].parameters():
                    parameter.add_(100)
            for value in objects[1].state.values():
                value["step"].fill_(99)
                value["exp_avg"].fill_(99)
                value["exp_avg_sq"].fill_(99)
            objects[2].last_epoch = 99
            iteration = saver.load(*objects)
            actual_draw = _sample()
            model_matches = all(
                torch.equal(expected_model[key], value) for key, value in objects[0].state_dict().items()
            )
            optimizer_matches = all(
                value["step"].item() == 1 and not value["exp_avg"].any().item() and not value["exp_avg_sq"].any().item()
                for value in objects[1].state.values()
            )
            local_rank = callbacks._callbacks[0].restored["rank"]
            status = {
                "success": iteration == 1
                and model_matches
                and optimizer_matches
                and objects[2].state_dict() == expected_scheduler
                and actual_draw == expected_draw
                and local_rank == rank
                and saver.rng_restored_from_same_job is True,
                "rank": rank,
                "local_rank": local_rank,
                "model_matches": model_matches,
                "optimizer_matches": optimizer_matches,
                "expected_draw": expected_draw,
                "actual_draw": actual_draw,
            }
            (Path(root) / f"rank_{rank}.json").write_text(json.dumps(status))
            return
        if scenario == "source_disagreement":
            if rank == 1:
                saver._read_latest_checkpoint_file = lambda: None
            try:
                saver.keys_to_resume_during_load()
                status = {"success": True, "rank": rank}
            except RuntimeError as exc:
                status = {
                    "success": False,
                    "rank": rank,
                    "error": str(exc),
                    "latest_unchanged": True,
                    "traceback": traceback.format_exc(),
                }
            (Path(root) / f"rank_{rank}.json").write_text(json.dumps(status))
            return
        if scenario in {"rng_failure", "rng_wrong_rank"}:
            reader = dcp.FileSystemReader(str(Path(saver.save_dirname) / "iter_000000001/trainer"))
            source = Path(saver.save_dirname) / "iter_000000001"
            if scenario == "rng_wrong_rank":
                source = Path(root) / "invalid_fixture/iter_000000001"
                invalid_rng = saver._rank_rng_state()
                if rank == 1:
                    invalid_rng["rank"] = torch.tensor(0, dtype=torch.int64)
                dcp.save(
                    {"grad_scaler": {}, "iteration": 1, f"rng_state_{rank}": invalid_rng},
                    storage_writer=dcp.FileSystemWriter(str(source / "trainer")),
                    process_group=saver._safety_group,
                )
                reader = dcp.FileSystemReader(str(source / "trainer"))
            elif rank == 1:
                original_metadata = reader.read_metadata

                def metadata():
                    value = copy.deepcopy(original_metadata())
                    value.state_dict_metadata.pop("rng_state_1.random_packed_len")
                    return value

                reader.read_metadata = metadata
            before = module.get_rand_state_dict()
            try:
                saver._load_strict_trainer(reader, objects[3], source)
                status = {"success": True, "rank": rank}
            except RuntimeError as exc:
                after = module.get_rand_state_dict()
                status = {
                    "success": False,
                    "rank": rank,
                    "error": str(exc),
                    "latest_unchanged": True,
                    "rng_unchanged": all(torch.equal(before[key], after[key]) for key in before),
                    "rng_restored": saver.rng_restored_from_same_job,
                    "traceback": traceback.format_exc(),
                }
            (Path(root) / f"rank_{rank}.json").write_text(json.dumps(status))
            return
        original = saver._save_as_pkl
        if rank == 1 and scenario == "dcp_failure":
            writer_factory = module.FileSystemWriter

            def storage_writer(path, **kw):
                writer = writer_factory(path, **kw)
                if Path(path).name == "optim":

                    def fail(*a, **kw):
                        raise OSError("injected rank1 DCP optimizer write failure")

                    writer.write_data = fail
                return writer

            module.FileSystemWriter = storage_writer
        if rank == 0 and scenario == "delay":

            def rank0_sidecar(value, directory):
                result = original(value, directory)
                (Path(root) / "rank0_sidecar_done").write_text("durable")
                return result

            saver._save_as_pkl = rank0_sidecar
        if rank == 1 and scenario in {
            "delay",
            "stall",
            "sidecar_failure",
            "first_failure",
            "partial_sidecar",
            "hard_exit",
        }:

            def sidecar(value, directory):
                event.set()
                if scenario in {"delay", "stall"}:
                    if not release.wait(20):
                        raise RuntimeError("test release timed out")
                    return original(value, directory)
                if scenario in {"partial_sidecar", "hard_exit"}:
                    Path(directory).mkdir(exist_ok=True, parents=True)
                    (Path(directory) / "rank_1.pkl.partial.tmp").write_bytes(b"partial never published")
                    if scenario == "hard_exit":
                        os._exit(17)
                raise OSError("injected rank1 disk/permission failure")

            saver._save_as_pkl = sidecar
        if rank == 0 and scenario in {"pointer_failure", "marker_failure"}:
            original_atomic = saver._atomic_write

            def atomic(path, writer):
                if Path(path).name == (
                    "latest_checkpoint.txt" if scenario == "pointer_failure" else saver.COMPLETE_FILE
                ):
                    raise OSError(f"injected {scenario} temp write failure")
                return original_atomic(path, writer)

            saver._atomic_write = atomic
        try:
            saver.save(*objects, iteration=2 + rank if scenario == "iteration_disagreement" else 2)
            status = {"success": True, "rank": rank}
        except BaseException as exc:
            status = {
                "success": False,
                "rank": rank,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "latest_unchanged": (
                    (Path(saver.save_dirname) / "latest_checkpoint.txt").read_bytes()
                    if (Path(saver.save_dirname) / "latest_checkpoint.txt").exists()
                    else None
                )
                == previous,
            }
        (Path(root) / f"rank_{rank}.json").write_text(json.dumps(status))
    finally:
        dist.destroy_process_group()


class DistributedSaveTests(unittest.TestCase):
    def _scenario(self, scenario):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as cleanup:
            root = Path(directory)
            ctx = mp.get_context("spawn")
            event, release = ctx.Event(), ctx.Event()
            jobs = [
                ctx.Process(target=_worker, args=(rank, str(root), str(root / "rendezvous"), scenario, event, release))
                for rank in range(2)
            ]
            for job in jobs:
                job.start()

                def stop(job=job):
                    if job.is_alive():
                        job.kill()
                    job.join()

                cleanup.callback(stop)
            if scenario == "delay":
                self.assertTrue(event.wait(20), "last rank did not reach sidecar")
                deadline = time.monotonic() + 10
                while not (root / "rank0_sidecar_done").exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertTrue((root / "rank0_sidecar_done").exists())
                self.assertEqual(_reader(root / "checkpoints")["checkpoint"], "iter_000000001")
                self.assertFalse((root / "checkpoints/iter_000000002").exists())
                print("DELAY witness: independent reader sees iter1 until rank1 sidecar acknowledgement")
                release.set()
            if scenario == "stall":
                self.assertTrue(event.wait(20))
                started = time.monotonic()
                deadline = started + 16
                while not (root / "rank_0.json").exists() and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertTrue((root / "rank_0.json").exists(), "bounded Gloo timeout did not fire")
                self.assertEqual(_reader(root / "checkpoints")["checkpoint"], "iter_000000001")
                release.set()
                print("TIMEOUT witness: rank0 failed within " + str(round(time.monotonic() - started, 2)) + " seconds")
            for job in jobs:
                job.join(30)
                if job.is_alive():
                    job.kill()
                    job.join()
                self.assertEqual(
                    job.exitcode, 17 if scenario == "hard_exit" and job is jobs[1] else 0, f"{scenario} worker exit"
                )
            reports = [
                json.loads((root / f"rank_{rank}.json").read_text())
                for rank in range(1 if scenario == "hard_exit" else 2)
            ]
            expected = scenario in {"delay", "resume_success"}
            self.assertTrue(all(report["success"] == expected for report in reports), reports)
            if not expected:
                self.assertTrue(all(report["latest_unchanged"] for report in reports))
                if scenario == "first_failure":
                    self.assertFalse((root / "checkpoints/latest_checkpoint.txt").exists())
                else:
                    self.assertEqual(_reader(root / "checkpoints")["checkpoint"], "iter_000000001")
            else:
                self.assertEqual(
                    _reader(root / "checkpoints")["checkpoint"],
                    "iter_000000001" if scenario == "resume_success" else "iter_000000002",
                )
            if scenario in {"rng_failure", "rng_wrong_rank"}:
                self.assertTrue(all(report["rng_unchanged"] and report["rng_restored"] is False for report in reports))
            if scenario in {"partial_sidecar", "hard_exit"}:
                self.assertTrue(list((root / "checkpoints").rglob("rank_1.pkl.partial.tmp")))
            print("GLOO " + json.dumps({"scenario": scenario, "ranks": reports}, sort_keys=True))

    def test_delayed_last_rank_and_repeated_save_publication(self):
        self._scenario("delay")

    def test_last_rank_sidecar_failure_preserves_previous_latest(self):
        self._scenario("sidecar_failure")

    def test_first_save_failure_never_creates_latest(self):
        self._scenario("first_failure")

    def test_partial_temp_file_never_publishes(self):
        self._scenario("partial_sidecar")

    def test_pointer_write_failure_preserves_previous_latest(self):
        self._scenario("pointer_failure")

    def test_dcp_component_write_failure_is_coordinated_without_publication(self):
        self._scenario("dcp_failure")

    def test_complete_marker_write_failure_never_publishes(self):
        self._scenario("marker_failure")

    def test_rank_process_death_never_publishes_partial_checkpoint(self):
        self._scenario("hard_exit")

    def test_one_rank_missing_rng_fails_all_ranks_before_restore(self):
        self._scenario("rng_failure")

    def test_cross_rank_rng_payload_fails_all_ranks_before_restore(self):
        self._scenario("rng_wrong_rank")

    def test_save_iteration_disagreement_never_publishes(self):
        self._scenario("iteration_disagreement")

    def test_resume_source_disagreement_is_fail_closed(self):
        self._scenario("source_disagreement")

    def test_unresponsive_rank_hits_bounded_gloo_timeout_without_publication(self):
        self._scenario("stall")

    def test_both_ranks_restore_model_optimizer_local_and_distinct_rng(self):
        self._scenario("resume_success")


class SingleRankTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.group_dir = tempfile.TemporaryDirectory()
        dist.init_process_group(
            "gloo",
            init_method="file://" + cls.group_dir.name + "/rendezvous",
            rank=0,
            world_size=1,
            timeout=timedelta(seconds=10),
        )
        cls.module = _modules()

    @classmethod
    def tearDownClass(cls):
        dist.destroy_process_group()
        cls.group_dir.cleanup()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        config, job = _configuration(self.root)
        self.callbacks = Callbacks()
        self.saver = self.module.SafeLocalDistributedCheckpointer(config, job, self.callbacks)
        self.objects = _objects()

    def test_actual_save_load_all_components_and_rng_next_draw(self):
        random.seed(29)
        np.random.seed(29)
        torch.manual_seed(29)
        self.saver.save(*self.objects, iteration=1)
        expected_model = {key: value.clone() for key, value in self.objects[0].state_dict().items()}
        expected = _sample()
        with torch.no_grad():
            for parameter in self.objects[0].parameters():
                parameter.add_(100)
        self.saver.load(*self.objects)
        for key, value in self.objects[0].state_dict().items():
            self.assertTrue(torch.equal(value, expected_model[key]), key)
        self.assertEqual(_sample(), expected)
        self.assertTrue(self.saver.rng_restored_from_same_job)
        self.assertEqual(self.callbacks._callbacks[0].restored["rank"], 0)
        self.assertTrue(_reader(Path(self.saver.save_dirname))["verified"])

    def test_missing_complete_marker_and_torn_pointer_rejected(self):
        self.saver.save(*self.objects, iteration=1)
        path = Path(self.saver.save_dirname)
        marker = path / "iter_000000001" / self.saver.COMPLETE_FILE
        data = marker.read_bytes()
        marker.unlink()
        with self.assertRaises((FileNotFoundError, ValueError)):
            self.saver._read_latest_checkpoint_file()
        marker.write_bytes(data)
        (path / "latest_checkpoint.txt").write_text("iter_000000")
        with self.assertRaisesRegex(ValueError, "torn"):
            self.saver._read_latest_checkpoint_file()

    def test_truncated_shard_and_modified_sidecar_rejected(self):
        self.saver.save(*self.objects, iteration=1)
        path = Path(self.saver.save_dirname) / "iter_000000001"
        sidecar = path / "dataloader/rank_0.pkl"
        original = sidecar.read_bytes()
        sidecar.write_bytes(b"truncated")
        with self.assertRaisesRegex(ValueError, "sidecar changed"):
            self.saver.validate_checkpoint(path, 1)
        sidecar.write_bytes(original)
        shard = next((path / "model").glob("*.distcp"))
        shard.write_bytes(b"")
        with self.assertRaisesRegex(ValueError, "truncated"):
            self.saver.validate_checkpoint(path, 1)

    def test_atomic_sidecar_partial_write_keeps_old_bytes_and_removes_temporary(self):
        path = self.root / "rank_0.pkl"
        path.write_bytes(b"previous complete sidecar")
        before = (path.read_bytes(), path.stat().st_mtime_ns)

        def fail(stream):
            stream.write(b"partial")
            raise OSError("injected write exhaustion")

        with self.assertRaisesRegex(OSError, "exhaustion"):
            self.saver._atomic_write(path, fail)
        self.assertEqual((path.read_bytes(), path.stat().st_mtime_ns), before)
        self.assertEqual(list(self.root.glob("*.tmp")), [])

    def test_post_replace_fsync_failure_rolls_latest_back(self):
        self.saver.save(*self.objects, iteration=1)
        parent = Path(self.saver.save_dirname)
        original = self.saver._fsync_directory
        failed = False

        def fsync(directory):
            nonlocal failed
            if (
                Path(directory) == parent
                and (parent / "latest_checkpoint.txt").read_text() == "iter_000000002\n"
                and not failed
            ):
                failed = True
                raise OSError("injected directory fsync failure after replacement")
            return original(directory)

        with patch.object(type(self.saver), "_fsync_directory", staticmethod(fsync)):
            with self.assertRaisesRegex(RuntimeError, "fsync failure"):
                self.saver.save(*self.objects, iteration=2)
        self.assertTrue(failed)
        self.assertEqual(_reader(parent)["checkpoint"], "iter_000000001")

    def test_existing_checkpoint_is_never_overwritten(self):
        self.saver.save(*self.objects, iteration=1)
        pointer = Path(self.saver.save_dirname) / "latest_checkpoint.txt"
        before = (pointer.read_bytes(), pointer.stat().st_mtime_ns)
        with self.assertRaisesRegex(RuntimeError, "never overwrites"):
            self.saver.save(*self.objects, iteration=1)
        self.assertEqual((pointer.read_bytes(), pointer.stat().st_mtime_ns), before)
        with self.assertRaisesRegex(RuntimeError, "advance latest"):
            self.saver.save(*self.objects, iteration=0)
        self.assertEqual((pointer.read_bytes(), pointer.stat().st_mtime_ns), before)

    def test_profile_rejects_async_and_object_store_without_changing_legacy_constructor(self):
        for overrides in (
            {"dcp_async_mode_enabled": True},
            {"strict_resume": False},
            {"save_to_object_store": NS(enabled=True)},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.module.SafeLocalDistributedCheckpointer(*_configuration(self.root, **overrides))
        legacy = self.module.DistributedCheckpointer(*_configuration(self.root), callbacks=None)
        self.assertIs(legacy.async_mode, self.module.AsyncMode.DISABLED)
        self.assertFalse(getattr(legacy, "strict_rank_rng", False))
        self.assertFalse(hasattr(legacy, "_safety_group"))

    def test_legacy_async_dispatch_and_sidecar_format_are_unchanged(self):
        legacy = self.module.DistributedCheckpointer(*_configuration(self.root), callbacks=self.callbacks)
        legacy.async_mode = self.module.AsyncMode.ASYNC_WITH_PINNED_MEM
        legacy._wait_for_previous_async_checkpoint = MagicMock()
        legacy._checkpoint_async_with_pinned_memory = MagicMock()
        legacy.save(*self.objects, iteration=1)
        legacy._wait_for_previous_async_checkpoint.assert_called_once()
        pending = legacy._checkpoint_async_with_pinned_memory.call_args.args[1]
        self.assertEqual(set(pending), {"model", "optim", "scheduler", "trainer"})
        self.assertNotIn("rank", pending["trainer"][0]["rng_state_0"])
        self.assertTrue((Path(legacy.save_dirname) / "iter_000000001/dataloader/rank_0.pkl").is_file())
        self.assertFalse((Path(legacy.save_dirname) / "latest_checkpoint.txt").exists())

    def test_fresh_model_only_warm_start_needs_no_rng_and_training_warm_start_is_rejected(self):
        source = self.root / "official_like_tiny_synthetic"
        dcp.save(
            self.module.ModelWrapper(self.objects[0]).state_dict(),
            storage_writer=dcp.FileSystemWriter(str(source / "model")),
        )
        config, job = _configuration(self.root, load_path=str(source), load_training_state=False)
        saver = self.module.SafeLocalDistributedCheckpointer(config, job, self.callbacks)
        self.assertEqual(saver.load(*self.objects), 0)
        self.assertIs(saver.rng_restored_from_same_job, False)
        saver.load_training_state = True
        with self.assertRaisesRegex(RuntimeError, "model-only"):
            saver.load(*self.objects)

    def _trainer_checkpoint(self, state, name="iter_000000003"):
        path = self.root / name / "trainer"
        dcp.save(state, storage_writer=dcp.FileSystemWriter(str(path)))
        return dcp.FileSystemReader(str(path)), path.parent

    def test_strict_rng_missing_rank_or_each_leaf_is_rejected_before_restore(self):
        valid = self.saver._rank_rng_state()
        for missing in ("whole_rank", *self.saver.RNG_FIELDS):
            state = {"grad_scaler": {}, "iteration": 3}
            if missing != "whole_rank":
                state["rng_state_0"] = {k: v for k, v in valid.items() if k != missing}
            reader, path = self._trainer_checkpoint(state, "iter_" + f"{10 + len(list(self.root.iterdir())):09}")
            with self.subTest(missing=missing), self.assertRaisesRegex(RuntimeError, "RNG metadata"):
                self.saver._load_strict_trainer(reader, self.objects[3], path)

    def test_strict_rng_wrong_rank_lengths_and_fake_tensors_rejected(self):
        for field, value in (
            ("rank", torch.tensor(1)),
            ("world_size", torch.tensor(2)),
            ("schema_version", torch.tensor(0)),
            ("numpy_packed_len", torch.tensor(0)),
            ("random_packed_bytes", torch.zeros(4096, dtype=torch.uint8)),
            ("torch", torch.zeros_like(torch.get_rng_state())),
        ):
            state = self.saver._rank_rng_state()
            state[field] = value
            with self.subTest(field=field), self.assertRaises(Exception):
                self.saver._validate_rng_payload(state)

    def test_non_tensor_rng_metadata_is_rejected_by_actual_reader(self):
        state = self.saver._rank_rng_state()
        state["torch"] = "fake RNG field"
        reader, path = self._trainer_checkpoint({"rng_state_0": state, "iteration": 3, "grad_scaler": {}})
        with self.assertRaisesRegex(RuntimeError, "metadata type"):
            self.saver._load_strict_trainer(reader, self.objects[3], path)
        self.assertIs(self.saver.rng_restored_from_same_job, False)

    def test_rng_metadata_without_stored_chunks_cannot_reuse_current_rng_template(self):
        rng = self.saver._rank_rng_state()
        reader, path = self._trainer_checkpoint({"rng_state_0": rng, "iteration": 3, "grad_scaler": {}})
        original = reader.read_metadata

        def metadata():
            value = copy.deepcopy(original())
            value.state_dict_metadata["rng_state_0.torch"].chunks = []
            return value

        reader.read_metadata = metadata
        with self.assertRaisesRegex(RuntimeError, "no complete stored tensor"):
            self.saver._load_strict_trainer(reader, self.objects[3], path)
        self.assertIs(self.saver.rng_restored_from_same_job, False)

    def test_deterministic_actual_trainer_resumes_without_reseed_and_fresh_still_seeds(self):
        f0 = _load("f1_f0_boundary", CHILD / "examples/psm_wma_robocasa_phase5_f0_test.py")
        rng = self.saver._rank_rng_state()
        reader, checkpoint = self._trainer_checkpoint({"rng_state_0": rng, "iteration": 3, "grad_scaler": {}})
        expected = _sample()
        for resumed in (True, False):
            case = f0.TrainerTests()
            case.setUp()
            trainer, module = case.trainer, case.module
            module.torch.are_deterministic_algorithms_enabled.return_value = True
            module.torch.amp.GradScaler.return_value.state_dict.return_value = {}

            def seed(*a, **kw):
                random.seed(17)
                np.random.seed(17)
                torch.manual_seed(17)

            module.misc.set_random_seed.side_effect = seed
            adapter = NS(save=MagicMock(), finalize=MagicMock(), rng_restored_from_same_job=False)

            def load(model, optim, sched, scaler):
                if resumed:
                    result = self.saver._load_strict_trainer(reader, scaler, checkpoint)
                    adapter.rng_restored_from_same_job = self.saver.rng_restored_from_same_job
                    return result
                return 0

            adapter.load = load
            trainer.checkpointer = adapter
            trainer.config.trainer.seed = 17
            trainer._grouped_completed_iteration = 3 if resumed else 0
            trainer._execution_max_iter = 4 if resumed else 1
            observed = []
            original_step = trainer.training_step

            def step(*a, **kw):
                if not observed:
                    observed.append(_sample())
                return original_step(*a, **kw)

            trainer.training_step = step
            with self.subTest(resumed=resumed), patch.dict(os.environ, {"GROUPED_MM_SM_CARVEOUT": "0"}):
                trainer.train(case.model, f0.phase5.GroupedTriggerLoader(30000, 2), None)
                if resumed:
                    module.misc.set_random_seed.assert_not_called()
                    self.assertEqual(observed[0], expected)
                    print("NEXT_DRAW " + json.dumps({"expected": expected, "observed": observed[0], "reseed_calls": 0}))
                else:
                    module.misc.set_random_seed.assert_called_once_with(seed=17, by_rank=True)

    def test_corrected_phase5_selects_safe_class_without_changing_formal_schedule(self):
        phase = _load("f1_phase5_wiring", CHILD / "examples/psm_wma_robocasa_corrected_phase5.py")
        phase.SafeLocalDistributedCheckpointer = self.module.SafeLocalDistributedCheckpointer
        phase.GroupedLocalMemoryTrainer = type("SyntheticTrainerTarget", (), {})
        phase.DictConfig = type("SyntheticDictConfig", (), {})
        config = NS(
            model=NS(
                config=NS(
                    parallelism=NS(), tokenizer={}, vlm_config=NS(), max_action_dim=64, local_memory_action_dim=15
                )
            ),
            trainer=NS(),
            optimizer=NS(),
            scheduler=NS(),
            checkpoint=NS(),
            job=NS(),
        )
        contract = NS(resolve_tokenizer_config=lambda value: value, validate_tokenizer_config=lambda value: None)
        args = NS(
            t=16,
            k=4,
            world_size=8,
            ga=2,
            max_iter=30000,
            warmup=500,
            save_iter=100,
            job_name="synthetic_formal",
            base_checkpoint=self.root,
        )
        overlay = ModuleType("examples.psm_wma_robocasa_local_s1")
        overlay.overlay_local_config = lambda config: None
        with patch.dict(sys.modules, {"examples.psm_wma_robocasa_local_s1": overlay}):
            phase.overlay_config(config, args, contract)
        self.assertIs(config.checkpoint.type["_target_"], self.module.SafeLocalDistributedCheckpointer)
        self.assertIs(config.checkpoint.dcp_async_mode_enabled, False)
        self.assertEqual(
            (config.trainer.max_iter, config.scheduler.warm_up_steps, config.checkpoint.save_iter), (30000, [500], 100)
        )
        self.assertEqual(phase.NATIVE_BATCH_LAYOUT, "v2_a2_member_packed_slot_major_v1")


if __name__ == "__main__":
    unittest.main(verbosity=2)
