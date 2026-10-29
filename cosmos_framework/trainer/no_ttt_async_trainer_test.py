"""Production No-TTT trainer/async-loader integration with synthetic CPU inputs.

The native collator, source planner, ActionTransformPipeline, loader, callback
and optimizer lifecycle are real. Physics, tokenizer vocabulary and model are
small CPU doubles. These tests neither load a real DCP nor exercise CUDA.
"""

from __future__ import annotations

import hashlib
import pickle
import random
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import fields, is_dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from cosmos_framework.model.generator.mot.robocasa_exact_window_local import ExactWindowRankPlanner
from cosmos_framework.trainer import matched_no_ttt as module
from cosmos_framework.trainer.local_memory_grouped_test import _Hooks
from cosmos_framework.trainer.no_ttt_async_loader import NoTTTAsyncLoader
from cosmos_framework.trainer.no_ttt_async_test_support import SyntheticAsyncRecipe, build_synthetic_producer
from cosmos_framework.utils.no_ttt_loader_profile import LoaderProfile


def _fingerprint(value):
    """Compare every native input value without expanding one-byte zero videos."""
    if isinstance(value, torch.Tensor):
        assert value.device.type == "cpu"
        header = (str(value.dtype), tuple(value.shape))
        if value.ndim == 4 and value.stride() == (0, 0, 0, 0):
            assert value.dtype == torch.uint8 and value.untyped_storage().nbytes() == 1
            assert int(value[0, 0, 0, 0]) == 0
            return (*header, "all_zero")
        encoded = value.detach().contiguous().numpy().tobytes()
        return (*header, hashlib.sha256(encoded).hexdigest())
    if isinstance(value, dict):
        return {key: _fingerprint(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, tuple(_fingerprint(item) for item in value))
    if is_dataclass(value):
        return (
            type(value).__qualname__,
            {field.name: _fingerprint(getattr(value, field.name)) for field in fields(value)},
        )
    return value


class NoisyNativeModel(nn.Module):
    """No Local parameters; no prefix kwarg; loss observes the selected CFG branch."""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(local_memory_enabled=False)
        self.net = nn.Module()
        self.net.moe_gen = nn.Parameter(torch.tensor(2.0))
        self.seen = []
        self.batch_sizes = []
        self.before_zero = None

    def training_step(self, batch, iteration):
        self.seen.append(_fingerprint(batch))
        size = len(batch["sequence_plan"])
        self.batch_sizes.append(size)
        assert len(batch["action"]) == len(batch["text_token_ids"]) == size
        action = torch.stack([row[0][1].mean() for row in batch["action_raw"]]).mean()
        tokens = sum(float(row[0].sum()) for row in batch["text_token_ids"]) / size
        # These draws must stay BETWEEN members' original CFG decisions, even
        # while workers prepare later members and the next Window in advance.
        noise = random.random() + float(np.random.rand()) + torch.rand(())
        loss = self.net.moe_gen * (1 + action * 0.001 + tokens * 0.00001 + noise * 0.001)
        return {"flow_matching_loss_action": loss, "aux_loss_gen": loss * 0.1}, loss

    def on_after_backward(self):
        pass

    def on_before_optimizer_step(self, *args, **kwargs):
        pass

    def on_before_zero_grad(self, *args, **kwargs):
        self.before_zero = self.net.moe_gen.grad.detach().clone()


def _rng():
    return random.getstate(), deepcopy(np.random.get_state()), torch.get_rng_state().clone()


def _set_rng(state):
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])


def _equal_rng(left, right):
    assert left[0] == right[0]
    assert left[1][0] == right[1][0] and left[1][2:] == right[1][2:]
    np.testing.assert_array_equal(left[1][1], right[1][1])
    torch.testing.assert_close(left[2], right[2], rtol=0, atol=0)


@pytest.fixture(autouse=True)
def cpu_integration_boundary(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("PSM_V3_COMPACT_CACHED_VIDEO", "1")

    def loader(producer, profile, planner):
        raw = producer.catalog.raw
        recipe = SyntheticAsyncRecipe(
            count=raw.records[0].window_count,
            t=producer.catalog.ttt_tbptt_steps,
            world=planner.world_size,
            rate=producer.catalog.wrapped_sft._transform.text_tokenizer.cfg_dropout_rate,
        )
        return NoTTTAsyncLoader(producer, profile, planner, recipe=recipe)

    # Only the worker's data recipe is doubled. The real async backend, native
    # Packed128 collator, CPU transfer and Trainer implementation still execute.
    monkeypatch.setattr(module, "NoTTTAsyncLoader", loader)
    with torch.enable_grad():
        yield


def _make(*, rank=0, workers=0, count=9, t=2):
    producer = build_synthetic_producer(count=count, t=t, world=4, rate=0.25)
    planner = ExactWindowRankPlanner(producer.catalog, rank=rank, world_size=4, b_stream=8, active_ga=4, seed=83)
    trainer = object.__new__(module.MatchedNoTTTTrainer)
    trainer.config = SimpleNamespace(trainer=SimpleNamespace(grad_accum_iter=4, distributed_parallelism="fsdp"))
    trainer.callbacks = _Hooks()
    trainer.training_timer = lambda name: nullcontext()
    options = {"loader_profile": LoaderProfile(workers, 1, True, 60)} if workers else {}
    trainer.bind_grouped_stream(planner, producer, config_digest="config", **options)
    trainer.cpu_loader_observer = lambda _: None
    model = NoisyNativeModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min((step + 1) / 500, 1))
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    return trainer, model, optimizer, scheduler, scaler


def _step(state, iteration):
    trainer, model, *optim = state
    planner = trainer._grouped_planner
    before = (
        trainer._grouped_window.live.frontier if hasattr(trainer, "_grouped_window") else planner.initial_frontier()
    )
    expected = planner.plan_window(before).candidate_frontier
    for member in range(4):
        output, loss, next_member = trainer.training_step(model, *optim, {}, iteration, member)
        assert next_member == (member + 1) % 4 and bool(torch.isfinite(loss))
        assert set(output) == {"flow_matching_loss_action", "aux_loss_gen"}
        assert trainer._grouped_window.live.frontier == (expected if member == 3 else before)
        assert trainer._grouped_completed_iteration == iteration + (member == 3)


def _close(state):
    if state is not None and state[0]._no_ttt_loader is not None:
        loader = state[0]._no_ttt_loader
        loader.close()
        assert loader.pending is None and loader.preview is None
        assert all(
            row["exitcode"] is not None and not Path(f"/proc/{row['pid']}").exists() for row in loader.last_cleanup
        )


def _checkpoint(state):
    trainer, model, optimizer, scheduler, scaler = state
    before = _rng()
    saved = {
        "model": model.state_dict(),
        "optim": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "trainer": {"iteration": trainer._grouped_completed_iteration, "rng": before},
        "dataloader": trainer.callbacks._callbacks[0].state_dict(),
    }
    _equal_rng(before, _rng())
    assert saved["dataloader"]["format"] == module.FORMAT
    assert set(saved["dataloader"]) == {
        "format",
        "iteration",
        "frontier",
        "config_digest",
        "profile",
        "cache_manifest_sha256",
        "cache_corpus_digest",
        "source_binding_digest",
    }
    return pickle.loads(pickle.dumps(saved))


def _restore(state, saved):
    trainer, model, optimizer, scheduler, scaler = state
    model.load_state_dict(saved["model"])
    optimizer.load_state_dict(saved["optim"])
    scheduler.load_state_dict(saved["scheduler"])
    scaler.load_state_dict(saved["scaler"])
    trainer.callbacks._callbacks[0].load_state_dict(saved["dataloader"])
    trainer._resume_required = True
    _set_rng(saved["trainer"]["rng"])


def _equal_training(left, right):
    a, b = left[0], right[0]
    assert a._grouped_completed_iteration == b._grouped_completed_iteration
    assert a._grouped_window.live == b._grouped_window.live
    torch.testing.assert_close(left[1].net.moe_gen, right[1].net.moe_gen, rtol=0, atol=0)
    torch.testing.assert_close(left[1].before_zero, right[1].before_zero, rtol=0, atol=0)
    assert left[3].state_dict() == right[3].state_dict()
    assert left[4].state_dict() == right[4].state_dict()
    torch.testing.assert_close(
        left[2].state[left[1].net.moe_gen]["momentum_buffer"],
        right[2].state[right[1].net.moe_gen]["momentum_buffer"],
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("rank", range(4))
@pytest.mark.parametrize("count,counts", [(64, [128, 128, 128, 128]), (17, [128, 8, 128, 8])])
def test_full_packed128_and_terminal_partial_four_ranks_match_sync_values_and_rng(rank, count, counts):
    random.seed(125 + rank)
    np.random.seed(125 + rank)
    torch.manual_seed(125 + rank)
    initial = _rng()
    sync = _make(rank=rank, count=count, t=16)
    async_state = None
    try:
        _step(sync, 0)
        expected_rng = _rng()
        _set_rng(initial)
        async_state = _make(rank=rank, workers=2, count=count, t=16)
        _step(async_state, 0)
        assert sync[1].batch_sizes == async_state[1].batch_sizes == counts
        assert sync[1].seen == async_state[1].seen
        _equal_training(sync, async_state)
        _equal_rng(expected_rng, _rng())
        assert async_state[0]._no_ttt_loader.preview is not None
        assert async_state[0]._no_ttt_loader.sampler.jobs, (
            "checkpoint must occur before all preview requests are dispatched"
        )
        assert _checkpoint(sync)["dataloader"] == _checkpoint(async_state)["dataloader"]
    finally:
        _close(sync)
        _close(async_state)


@pytest.mark.parametrize("rank", range(4))
@pytest.mark.parametrize("from_workers,to_workers", [(0, 8), (8, 0), (8, 8)])
def test_same_job_checkpoint_resume_worker_changes_preserve_entire_next_steps(rank, from_workers, to_workers):
    random.seed(270 + rank)
    np.random.seed(270 + rank)
    torch.manual_seed(270 + rank)
    source = _make(rank=rank, workers=from_workers)
    resumed = None
    try:
        _step(source, 0)
        if from_workers:
            assert source[0]._no_ttt_loader.pending is None and source[0]._no_ttt_loader.preview is not None
            assert source[0]._no_ttt_loader.sampler.jobs
        saved = _checkpoint(source)
        for iteration in (1, 2):
            _step(source, iteration)
        expected_rng = _rng()
        expected_batches = source[1].seen[4:]
        expected_frontier = _checkpoint(source)["dataloader"]
        _close(source)  # Never overlap two separate rank-owned pools.
        resumed = _make(rank=rank, workers=to_workers)
        _restore(resumed, saved)
        for iteration in (1, 2):
            _step(resumed, iteration)
        assert resumed[1].seen == expected_batches
        _equal_training(source, resumed)
        _equal_rng(expected_rng, _rng())
        assert expected_frontier == _checkpoint(resumed)["dataloader"]
    finally:
        _close(source)
        _close(resumed)


@pytest.mark.parametrize("failure", ["nan_loss", "nan_gradient", "step_skip", "native_error"])
def test_failed_active_window_keeps_checkpoint_frontier_and_reaps_speculation(monkeypatch, failure):
    state = _make(workers=2)
    trainer, model = state[:2]
    try:
        _step(state, 0)
        saved = _checkpoint(state)
        before_failed_window_rng = _rng()
        original = model.training_step
        if failure == "nan_loss":
            monkeypatch.setattr(model, "training_step", lambda *args: ({}, model.net.moe_gen * float("nan")))
        elif failure == "nan_gradient":
            model.net.moe_gen.register_hook(lambda grad: grad * float("nan"))
        elif failure == "step_skip":
            monkeypatch.setattr(trainer, "_optimizer_step_success", lambda *args: False)
        else:

            def explode(*args):
                raise OSError("injected async native compute failure")

            monkeypatch.setattr(model, "training_step", explode)
        with pytest.raises((ValueError, FloatingPointError, RuntimeError, OSError)):
            _step(state, 1)
        assert trainer._grouped_window.live.frontier == saved["dataloader"]["frontier"]
        assert trainer._grouped_completed_iteration == 1 and state[3].last_epoch == 1
        assert trainer._no_ttt_failed and trainer._no_ttt_loader.closed
        assert trainer._no_ttt_loader.pending is None and trainer._no_ttt_loader.preview is None
        _equal_rng(before_failed_window_rng, _rng())
        with pytest.raises(RuntimeError, match="committed"):
            trainer.callbacks._callbacks[0].state_dict()
        with pytest.raises(RuntimeError, match="failed instance"):
            trainer.training_step(model, *state[2:], {}, 1, 0)
        # The persisted pre-failure state remains immutable and reloadable; a
        # failed Trainer is never reused or mistaken for an in-memory rollback.
        assert saved["trainer"]["iteration"] == saved["dataloader"]["iteration"] == 1
        monkeypatch.setattr(model, "training_step", original)
    finally:
        _close(state)
