"""F0 CPU contracts. Real CLI and Trainer methods, synthetic external dependencies only.

Run directly with unittest; no repository-wide GPU/inference conftest is needed.
This is not DCP, real-model, distributed-NCCL or independent DS acceptance.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import multiprocessing
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace as NS
from unittest.mock import MagicMock, patch

HERE = Path(__file__).resolve().parent
CHILD = HERE.parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


phase5 = _load("f0_phase5_cpu", HERE / "psm_wma_robocasa_corrected_phase5.py")


def _args(root, *extra):
    argv = [
        "--phase",
        "fresh",
        "--job-name",
        "formal",
        "--expected-root",
        "a" * 40,
        "--expected-child",
        "b" * 40,
        "--formal-execution-until",
        "100",
    ]
    for name in (
        "output-root",
        "source-root",
        "cache-root",
        "edge",
        "vae",
        "base-checkpoint",
        "root-worktree",
        "dataset-index-root",
    ):
        argv += ["--" + name, str(root if name == "output-root" else root / name)]
    return phase5.parser().parse_args(argv + list(extra))


class Serializer:
    @staticmethod
    def save_pkl(config, path):
        Path(path).write_bytes(b"synthetic resolved config pickle")

    @staticmethod
    def save_yaml(config, path):
        Path(path).write_text(json.dumps({"phase": config.phase, "schedule": [30000, 500, 100]}))


def _report(args, semantic=None):
    return {
        "training_identity": semantic or {"seed": 0, "layout": phase5.NATIVE_BATCH_LAYOUT},
        "config_digest": "d" * 64,
        "execution_stop_after_iter": phase5._execution_stop_iteration(args),
        "checkpoint_load_path": str(phase5.job_namespace(args) / "checkpoints/iter_000000100"),
    }


def _config(args):
    return NS(phase=args.phase, job=NS(path_local=str(phase5.job_namespace(args))))


def _write_job(args, report=None, serializer=Serializer):
    lifecycle = phase5.JobLifecycle(args, _report(args) if report is None else report)
    try:
        lifecycle.write_config(_config(args), serializer)
    finally:
        lifecycle.close()


def _claim_worker(args, barrier, results):
    barrier.wait(timeout=15)
    try:
        _write_job(args)
        results.put("claimed")
    except (FileExistsError, ValueError):
        results.put("rejected")


def _witness(path):
    return {str(p.relative_to(path)): (p.read_bytes(), p.stat().st_mtime_ns) for p in path.rglob("*") if p.is_file()}


def _witness_json(witness):
    return {
        name: {"sha256": hashlib.sha256(data).hexdigest(), "mtime_ns": mtime} for name, (data, mtime) in witness.items()
    }


def _trainer_module():
    """Import the unmodified production module using explicit external-boundary stand-ins."""
    modules = {}

    def mod(name, **members):
        item = ModuleType(name)
        item.__dict__.update(members)
        modules[name] = item
        return item

    dist = MagicMock()
    torch = MagicMock()
    torch.distributed = dist
    torch.are_deterministic_algorithms_enabled.return_value = False
    mod(
        "torch",
        **{
            "distributed": dist,
            "utils": torch.utils,
            "amp": torch.amp,
            "Tensor": torch.Tensor,
            "no_grad": torch.no_grad,
            "nn": torch.nn,
            "optim": torch.optim,
            "are_deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled,
        },
    )
    modules["torch.distributed"] = dist
    modules["torch.utils"] = torch.utils
    modules["torch.utils.data"] = torch.utils.data
    mod("cosmos_framework", __path__=[])
    mod(
        "cosmos_framework.utils",
        __path__=[],
        **{name: MagicMock() for name in ("callback", "distributed", "ema", "log", "misc")},
    )
    mod("cosmos_framework.model", __path__=[])
    mod("cosmos_framework.model._base", ImaginaireModel=MagicMock())
    mod("cosmos_framework.utils.flags", INTERNAL=False)
    mod("cosmos_framework.utils.context_managers", distributed_init=nullcontext)
    mod(
        "cosmos_framework.utils.profiling",
        **{
            name: lambda *a, **kw: nullcontext()
            for name in ("maybe_enable_memory_snapshot", "maybe_enable_nsys_profiling", "maybe_enable_profiling")
        },
    )
    mod("cosmos_framework.utils.lazy_config", LazyConfig=Serializer, instantiate=MagicMock())
    mod("cosmos_framework.utils.checkpointer", Checkpointer=MagicMock())
    mod("cosmos_framework.utils.misc", StragglerDetectorV2=MagicMock())
    with patch.dict(sys.modules, modules):
        module = _load("f0_trainer_cpu", CHILD / "cosmos_framework/trainer/__init__.py")
    module.misc.to.side_effect = lambda data, **kw: data
    return module


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.args = _args(self.root)

    def fresh(self):
        _write_job(self.args)
        path = phase5.job_namespace(self.args)
        checkpoint = path / "checkpoints/iter_000000100"
        for component in ("model", "optim", "scheduler", "trainer"):
            (checkpoint / component).mkdir(parents=True)
            (checkpoint / component / ".metadata").write_text("synthetic DCP metadata, never loaded")
        (checkpoint / "dataloader").mkdir()
        for rank in range(self.args.world_size):
            (checkpoint / "dataloader" / f"rank_{rank}.pkl").write_bytes(b"synthetic local sidecar, never loaded")
        (path / "checkpoints/latest_checkpoint.txt").write_text(checkpoint.name)
        return path

    def test_concurrent_resume_is_exclusive_and_latest_drift_is_rejected(self):
        path = self.fresh()
        self.args.phase = "resume"
        self.args.formal_execution_until = 500
        first = phase5.JobLifecycle(self.args, _report(self.args))
        first.write_config(_config(self.args), Serializer)
        self.addCleanup(first.close)
        second = phase5.JobLifecycle(self.args, _report(self.args))
        self.addCleanup(second.close)
        before = _witness(path)
        with self.assertRaises(BlockingIOError):
            second.write_config(_config(self.args), Serializer)
        self.assertEqual(before, _witness(path))
        first.close()
        second.close()
        stale_report = _report(self.args)
        stale_report["checkpoint_load_path"] = str(path / "checkpoints/iter_000000099")
        with self.assertRaisesRegex(ValueError, "changed between preflight"):
            _write_job(self.args, stale_report)
        self.assertEqual(before, _witness(path))

    def cli(self, args=None):
        args = args or self.args
        argv = []
        for key, value in vars(args).items():
            if isinstance(value, bool):
                if value:
                    argv += ["--" + key.replace("_", "-")]
            elif value is not None:
                argv += ["--" + key.replace("_", "-"), str(value)]
        if "--preflight" not in argv:
            argv += ["--preflight"]
        proc = subprocess.run(
            [sys.executable, str(HERE / "psm_wma_robocasa_corrected_phase5.py"), *argv],
            text=True,
            capture_output=True,
            timeout=15,
        )
        self.assertNotEqual(proc.returncode, 0, proc.stdout)
        self.assertNotIn("No module named", proc.stderr, proc.stderr)
        print(f"CLI negative: {self.id().split('.')[-1]} rc={proc.returncode}: {proc.stderr.splitlines()[-1]}")
        return proc

    def test_existing_namespaces_rejected_by_real_cli_without_any_old_file_change(self):
        path = phase5.job_namespace(self.args)
        for relative in (None, "checkpoints/latest_checkpoint.txt", "config.yaml", "evidence/old.json", "partial/tmp"):
            with self.subTest(existing=relative):
                path.mkdir(parents=True, exist_ok=True)
                if relative:
                    old = path / relative
                    old.parent.mkdir(parents=True, exist_ok=True)
                    old.write_text("original evidence")
                before = _witness(path)
                directory_mtime = path.stat().st_mtime_ns
                self.assertIn("previously nonexistent", self.cli().stderr)
                self.assertEqual(before, _witness(path))
                self.assertEqual(directory_mtime, path.stat().st_mtime_ns)
                print(
                    "IMMUTABILITY "
                    + json.dumps(
                        {
                            "case": relative or "existing_empty",
                            "before": _witness_json(before),
                            "after": _witness_json(_witness(path)),
                        },
                        sort_keys=True,
                    )
                )

    def test_real_cli_rejects_symlink_output_alias(self):
        alias = self.root / "alias"
        alias.symlink_to(self.root, target_is_directory=True)
        self.args.output_root = alias
        self.assertIn("symlink", self.cli().stderr)

    def test_real_cli_rejects_namespace_symlink(self):
        path = phase5.job_namespace(self.args)
        path.parent.mkdir(parents=True)
        path.symlink_to(self.root, target_is_directory=True)
        self.assertIn("symlink", self.cli().stderr)

    def test_real_cli_rejects_path_traversal(self):
        self.args.output_root = self.root / "x/../other"
        self.assertIn("traversal", self.cli().stderr)
        self.args.output_root = self.root
        self.args.job_name = "../old_job"
        self.assertIn("one ordinary path", self.cli().stderr)

    def test_two_concurrent_fresh_have_exactly_one_atomic_claim(self):
        ctx = multiprocessing.get_context("spawn")
        barrier, queue = ctx.Barrier(2), ctx.Queue()
        jobs = [ctx.Process(target=_claim_worker, args=(self.args, barrier, queue)) for _ in range(2)]
        for job in jobs:
            job.start()
        for job in jobs:
            job.join(20)
            if job.is_alive():
                job.kill()
                job.join()
            self.assertEqual(job.exitcode, 0)
        self.assertCountEqual([queue.get(timeout=2), queue.get(timeout=2)], ["claimed", "rejected"])
        path = phase5.job_namespace(self.args)
        self.assertEqual(len(list((path / "launches").iterdir())), 1)
        self.assertTrue((path / phase5.JOB_IDENTITY).is_file())

    def test_resume_keeps_canonical_bytes_mtime_and_adds_new_snapshot(self):
        path = self.fresh()
        before = _witness(path)
        self.args.phase = "resume"
        self.args.formal_execution_until = 500
        _write_job(self.args)
        after = _witness(path)
        for name, witness in before.items():
            self.assertEqual(witness, after[name], name)
        print(
            "RESUME_IMMUTABILITY "
            + json.dumps(
                {
                    "before": _witness_json(before),
                    "after_existing": _witness_json({name: after[name] for name in before}),
                },
                sort_keys=True,
            )
        )
        self.assertEqual(len(list((path / "launches").iterdir())), 2)
        print("Canonical bytes+mtime and prior launch witnesses unchanged across resume")

    def test_real_cli_resume_requires_identity_and_canonical_config(self):
        self.args.phase = "resume"
        self.assertIn("v3_job_identity.json", self.cli().stderr)
        self.args.phase = "fresh"
        path = self.fresh()
        self.args.phase = "resume"
        (path / "config.pkl").write_text("modified config")
        before = _witness(path)
        self.assertIn("canonical config changed", self.cli().stderr)
        self.assertEqual(before, _witness(path))

    def test_real_cli_resume_rejects_other_root_child_and_layout(self):
        path = self.fresh()
        self.args.phase = "resume"
        before = _witness(path)
        for identity_key in ("expected_root", "expected_child"):
            with self.subTest(field=identity_key):
                old = getattr(self.args, identity_key)
                setattr(self.args, identity_key, "c" * 40)
                self.assertIn("identity", self.cli().stderr)
                setattr(self.args, identity_key, old)
                self.assertEqual(before, _witness(path))
        identity = path / phase5.JOB_IDENTITY
        record = json.loads(identity.read_text())
        record["native_batch_layout"] = "legacy_index8"
        identity.write_text(json.dumps(record))
        before = _witness(path)
        self.assertIn("identity", self.cli().stderr)
        self.assertEqual(before, _witness(path))

    def test_semantic_mismatch_is_rejected_before_write(self):
        path = self.fresh()
        self.args.phase = "resume"
        before = _witness(path)
        with self.assertRaisesRegex(ValueError, "semantic"):
            _write_job(self.args, _report(self.args, {"seed": 1}))
        self.assertEqual(before, _witness(path))

    def test_resume_rejects_checkpoint_symlinks_and_missing_rank_sidecars(self):
        path = self.fresh()
        self.args.phase = "resume"
        latest = path / "checkpoints/latest_checkpoint.txt"
        old = latest.read_bytes()
        latest.unlink()
        target = self.root / "outside_latest.txt"
        target.write_bytes(old)
        latest.symlink_to(target)
        with self.assertRaisesRegex(ValueError, "symlink"):
            phase5._resume_checkpoint(self.args)
        latest.unlink()
        latest.write_bytes(old)
        sidecar = latest.parent / old.decode() / "dataloader/rank_7.pkl"
        sidecar.unlink()
        before = _witness(path)
        with self.assertRaisesRegex(FileNotFoundError, "rank_7"):
            phase5._resume_checkpoint(self.args)
        self.assertEqual(before, _witness(path))

    def test_incomplete_snapshot_cannot_be_resumed(self):
        serializer = NS(save_pkl=lambda *a: None, save_yaml=lambda *a: None)
        with self.assertRaisesRegex(RuntimeError, "snapshot missing"):
            _write_job(self.args, serializer=serializer)
        self.args.phase = "resume"
        self.assertIn("v3_job_identity.json", self.cli().stderr)

    def test_bounded_diagnostic_remains_nonresumable(self):
        self.args.job_name = "bounded_probe"
        self.args.formal_execution_until = None
        self.args.stop_after_iter = 3
        self.assertEqual(phase5._execution_stop_iteration(self.args), 3)
        self.fresh()
        self.args.phase = "resume"
        self.assertIn("must not be resumed", self.cli().stderr)

    def test_real_cli_invalid_budgets_rejected_before_runtime(self):
        for budget in (-1, 0, 30001):
            with self.subTest(budget=budget):
                self.args.formal_execution_until = budget
                self.assertIn("formal execution budget", self.cli().stderr)

    def test_formal_and_diagnostic_options_cannot_mix(self):
        for key, value in (
            ("stop_after_iter", 3),
            ("audit_missing_grads", True),
            ("job_name", "bounded_probe"),
            ("max_iter", 100),
            ("warmup", 0),
            ("save_iter", 1),
        ):
            args = _args(self.root)
            setattr(args, key, value)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "formal execution"):
                phase5._execution_stop_iteration(args)


@dataclass
class SemanticConfig:
    model: dict = field(default_factory=lambda: {"raw_action_dim": 15, "mask": "raw15", "precision": "bf16"})
    optimizer: dict = field(
        default_factory=lambda: {"keys_to_select": ["local"], "weight_decay": 0.05, "weight_decay_skip": ["bias"]}
    )
    scheduler: dict = field(default_factory=lambda: {"cycle_lengths": [30000], "warm_up_steps": [500]})
    trainer: dict = field(default_factory=lambda: {"seed": 0, "max_iter": 30000})
    model_parallel: dict = field(default_factory=lambda: {"world_size": 8})
    data_setting: dict = field(default_factory=dict)
    checkpoint: NS = field(default_factory=lambda: NS(save_iter=100, strict_resume=True))


class IdentityTests(unittest.TestCase):
    def test_all_rank_preflight_errors_and_identity_disagreement_block_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            args = _args(Path(directory))
            config = NS(validate=lambda: None, freeze=lambda: None)
            report = _report(args)
            for injected in ("peer-error", "different-digest"):
                calls = []

                def gather(output, value):
                    calls.append(value)
                    output[:] = [value] * args.world_size
                    if injected == "peer-error" or len(calls) == 2:
                        output[-1] = injected

                with (
                    self.subTest(injected=injected),
                    patch.object(phase5, "preflight", return_value=(report, config, None, None)),
                    patch.object(phase5, "torch", NS(distributed=NS(all_gather_object=gather)), create=True),
                    self.assertRaises((RuntimeError, ValueError)),
                ):
                    phase5._agree_preflight(args)
                self.assertFalse(phase5.job_namespace(args).exists())

    def test_budget_phase_load_path_do_not_change_semantic_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            args = _args(Path(directory))
            args.dataset_index_root.mkdir()
            (args.dataset_index_root / "receipt.json").write_text("accepted receipt")
            config = SemanticConfig()
            report = dict.fromkeys(
                (
                    "cache_manifest_sha256",
                    "cache_corpus_digest",
                    "source_binding_digest",
                    "model_witnesses",
                    "transform_rng_mode",
                ),
                "fixture",
            )
            with patch.object(phase5, "torch", NS(are_deterministic_algorithms_enabled=lambda: False), create=True):
                initial = phase5.training_identity(args, config, report)
                args.phase, args.formal_execution_until = "resume", 500
                config.checkpoint.load_path, config.checkpoint.load_training_state = "iter_000000100", True
                self.assertEqual(initial, phase5.training_identity(args, config, report))
                print(
                    "DIGEST_INVARIANT "
                    + json.dumps(
                        {
                            "fresh100": phase5._json_digest(initial),
                            "resume500": phase5._json_digest(phase5.training_identity(args, config, report)),
                        }
                    )
                )
                self.assertEqual(config.scheduler, {"cycle_lengths": [30000], "warm_up_steps": [500]})
                for section, key, changed in (
                    ("model", "mask", "wrong"),
                    ("model", "precision", "fp16"),
                    ("optimizer", "weight_decay_skip", []),
                    ("trainer", "seed", 1),
                ):
                    with self.subTest(section=section, key=key):
                        old = getattr(config, section)[key]
                        getattr(config, section)[key] = changed
                        self.assertNotEqual(initial, phase5.training_identity(args, config, report))
                        getattr(config, section)[key] = old
            print("Semantic digest stable: fresh100 / resume500; mask, precision, WD skip and seed changes rejected")


class TrainerTests(unittest.TestCase):
    def setUp(self):
        self.module = _trainer_module()
        self.trainer = object.__new__(self.module.ImaginaireTrainer)
        self.trainer.config = NS(
            trainer=NS(
                memory_format=None,
                grad_scaler_args={},
                max_iter=30000,
                distributed_parallelism="fsdp",
                run_validation=False,
                save_zero_checkpoint=False,
                grad_accum_iter=2,
                timeout_period=0,
                straggler_detection=NS(analyze_dataloading=False),
            ),
            optimizer={},
            scheduler={"cycle_lengths": [30000], "warm_up_steps": [500]},
            checkpoint=NS(save_iter=100),
        )
        self.trainer.callbacks = MagicMock()
        self.trainer.checkpointer = MagicMock()
        self.trainer.checkpointer.load.return_value = 0
        self.trainer.training_timer = lambda *a: nullcontext()
        self.trainer.straggler_detector = MagicMock()
        self.trainer.straggler_detector.profile_section.side_effect = lambda *a, **kw: nullcontext()
        self.trainer._cp_data_window = MagicMock()
        self.trainer._validation_iterator = None
        self.trainer._require_committed_execution_boundary = True
        self.trainer._grouped_completed_iteration = 0
        self.trainer._context_parallel_size = lambda model: 1
        self.trainer._fetch_data_batch = lambda model, iterator: (next(iterator), False)
        self.model = MagicMock(training=True)
        self.model.to.return_value = self.model
        self.model.init_optimizer_scheduler.return_value = (MagicMock(), MagicMock())

        def step(*a, iteration, grad_accum_iter):
            next_ga = (grad_accum_iter + 1) % 2
            if next_ga == 0:
                self.trainer._grouped_completed_iteration = iteration + 1
            return {}, 1.0, next_ga

        self.trainer.training_step = step

    def run_train(self, start, until, loader=None):
        self.trainer.checkpointer.load.return_value = start
        self.trainer._grouped_completed_iteration = start
        self.trainer._execution_max_iter = until
        loader = phase5.GroupedTriggerLoader(30000, 2) if loader is None else loader
        with patch.dict(os.environ, {"GROUPED_MM_SM_CARVEOUT": "0"}):
            self.trainer.train(self.model, loader, None)

    def test_real_trainer_fresh100_then_resume500_schedule_unchanged(self):
        for start, until in ((0, 100), (100, 500)):
            with self.subTest(start=start, until=until):
                self.trainer.checkpointer.save.reset_mock()
                self.run_train(start, until)
                self.assertEqual(self.trainer._grouped_completed_iteration, until)
                self.assertEqual(self.trainer.config.trainer.max_iter, 30000)
                self.assertEqual(self.trainer.config.scheduler, {"cycle_lengths": [30000], "warm_up_steps": [500]})
                self.assertEqual(
                    [c.kwargs["iteration"] for c in self.trainer.checkpointer.save.call_args_list],
                    list(range(start + 100, until + 1, 100)),
                )
        self.assertEqual(self.module.log.success.call_count, 2)

    def test_real_trainer_resume101_makes_off_cadence_checkpoint(self):
        self.run_train(100, 101)
        self.assertEqual([c.kwargs["iteration"] for c in self.trainer.checkpointer.save.call_args_list], [101])
        self.trainer.checkpointer.finalize.assert_called_once()
        self.module.distributed.barrier.assert_called_once()

    def test_phase5_main_wires_lifecycle_budget_and_actual_trainer_loop(self):
        with tempfile.TemporaryDirectory() as directory:
            args = _args(Path(directory))
            report = _report(args)
            report.update(
                dataset_init={},
                transform_rng_mode="legacy_global",
                full_segment_prefetch=False,
                checkpoint_load_path=str(args.base_checkpoint),
            )
            config = self.trainer.config
            config.job, config.phase = _config(args).job, args.phase
            config.model = {}
            self.trainer.bind_grouped_stream = MagicMock()

            def construct(actual_config, *, config_writer):
                self.assertIs(actual_config, config)
                config_writer(config, Serializer)
                return self.trainer

            observer = MagicMock(completed=100)
            with (
                patch.multiple(
                    phase5,
                    create=True,
                    _load_runtime=MagicMock(),
                    _agree_preflight=MagicMock(return_value=(report, config, None, None)),
                    torch=NS(cuda=NS(is_available=lambda: True)),
                    distributed=MagicMock(),
                    GroupedLocalMemoryTrainer=construct,
                    model_init=nullcontext,
                    instantiate=MagicMock(return_value=self.model),
                    install_optimizer_inventory_check=MagicMock(),
                    ExactWindowRankPlanner=MagicMock(),
                    ExactWindowSegmentProducer=MagicMock(),
                    TelemetryJournal=MagicMock(),
                    GroupedPlanObserver=MagicMock(return_value=observer),
                ),
                patch.object(phase5, "parser", return_value=NS(parse_args=lambda argv: args)),
                patch.dict(os.environ, {"RANK": "0", "WORLD_SIZE": "8", "GROUPED_MM_SM_CARVEOUT": "0"}),
            ):
                phase5.main([])
            self.assertEqual(self.trainer._execution_max_iter, 100)
            self.assertEqual(self.trainer._grouped_completed_iteration, 100)
            self.assertTrue((phase5.job_namespace(args) / phase5.JOB_IDENTITY).is_file())
            self.trainer.checkpointer.finalize.assert_called_once()

    def test_launcher_has_sealed_formal_profile_and_no_legacy_entry(self):
        launcher = CHILD.parent / "scripts/train_v3_packed128_staged.sh"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake_python = root / "capture-python"
            capture = root / "argv.json"
            fake_python.write_text(
                "#!"
                + sys.executable
                + "\nimport json,sys\nfrom pathlib import Path\n"
                + "Path("
                + repr(str(capture))
                + ").write_text(json.dumps(sys.argv[1:]))\n"
            )
            fake_python.chmod(0o700)
            env = dict(os.environ, PYTHON=str(fake_python))
            for key in (
                "OUTPUT_ROOT",
                "SOURCE_ROOT",
                "CACHE_ROOT",
                "DATASET_INDEX_ROOT",
                "EDGE_POLICY_CHECKPOINT",
                "WAN_VAE_PATH",
                "BASE_CHECKPOINT_PATH",
                "JOB_NAME",
                "EXPECTED_ROOT",
                "EXPECTED_CHILD",
            ):
                env[key] = "synthetic_" + key
            proc = subprocess.run(["bash", str(launcher), "fresh", "100"], env=env, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            argv = json.loads(capture.read_text())
            self.assertEqual(argv[:4], ["-m", "torch.distributed.run", "--standalone", "--nproc_per_node=8"])
            self.assertIn("examples/psm_wma_robocasa_corrected_phase5.py", argv)
            for key, value in (
                ("--world-size", "8"),
                ("--num-workers", "0"),
                ("--max-iter", "30000"),
                ("--warmup", "500"),
                ("--save-iter", "100"),
                ("--ga", "2"),
            ):
                self.assertEqual(argv[argv.index(key) + 1], value)
            self.assertNotIn("--full-segment-prefetch", argv)
            proc = subprocess.run(
                ["bash", str(launcher), "fresh", "100", "--max-iter", "100"], env=env, capture_output=True, text=True
            )
            self.assertEqual(proc.returncode, 2)

    def test_real_trainer_invalid_resume_budget_rejected_before_consumption(self):
        for start, until in ((100, 100), (101, 100), (0, -1), (0, 30001)):
            with self.subTest(start=start, until=until), self.assertRaisesRegex(ValueError, "Execution iteration"):
                self.run_train(start, until, loader=iter(()))
        self.trainer.checkpointer.save.assert_not_called()
        self.module.log.success.assert_not_called()

    def test_real_trainer_half_ga_cannot_finish_or_save(self):
        with self.assertRaisesRegex(RuntimeError, "Dataloader ended"):
            self.run_train(0, 100, loader=iter(({},)))
        self.trainer.checkpointer.save.assert_not_called()
        self.module.log.success.assert_not_called()

    def test_real_trainer_missing_local_commit_cannot_finish(self):
        self.trainer.training_step = lambda *a, **kw: ({}, 1.0, 0)
        with self.assertRaisesRegex(RuntimeError, "committed optimizer/Local boundary"):
            self.run_train(100, 101)
        self.trainer.checkpointer.save.assert_not_called()
        self.module.log.success.assert_not_called()

    def test_real_trainer_save_or_finalize_failure_never_reports_success(self):
        for operation in ("save", "finalize"):
            with self.subTest(operation=operation):
                getattr(self.trainer.checkpointer, operation).side_effect = OSError("synthetic storage failure")
                with self.assertRaisesRegex(OSError, "storage failure"):
                    self.run_train(100, 101)
                self.module.log.success.assert_not_called()
                getattr(self.trainer.checkpointer, operation).side_effect = None

    def test_real_constructor_propagates_writer_failure_before_legacy_config_overwrite(self):
        config = NS(model=NS(), trainer=NS(), job=NS(path_local="MUST_NOT_BE_CREATED"))
        with self.assertRaisesRegex(RuntimeError, "Job lifecycle config write failed"):
            self.module.ImaginaireTrainer(config, config_writer=MagicMock(side_effect=FileExistsError("old job")))
        self.module.dist.broadcast_object_list.assert_called_once()
        self.assertFalse(Path(config.job.path_local).exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
