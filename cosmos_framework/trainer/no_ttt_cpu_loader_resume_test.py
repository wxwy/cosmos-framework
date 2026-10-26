"""Synthetic full CPU state roundtrip100→101/500; no real DCP/CUDA test."""

import pickle
import random
from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cosmos_framework.trainer import matched_no_ttt as trainer_module
from cosmos_framework.trainer import no_ttt_cpu_loader as loader_module
from cosmos_framework.trainer.local_memory_grouped_test import _Hooks
from cosmos_framework.trainer.matched_no_ttt_test import Model
from cosmos_framework.trainer.no_ttt_cpu_loader_test import source_setup
from cosmos_framework.trainer.no_ttt_loader_test_support import SyntheticRecipe
from cosmos_framework.utils.no_ttt_loader_profile import LoaderProfile


class NoisyModel(Model):
    def training_step(self, batch, iteration):
        out, loss = super().training_step(batch, iteration)
        loss = loss * (1 + 0.001 * (random.random() + np.random.rand() + torch.rand(())))
        return out, loss


def make(rank=0, workers=1):
    # Scale-reduced long-run geometry. Full B8/T16/GA4 rank parity is tested separately.
    _, _, planner, producer = source_setup(rank=rank, world=4, count=3000, t=1, b=1, ga=4)
    result = object.__new__(trainer_module.MatchedNoTTTTrainer)
    result.config = SimpleNamespace(trainer=SimpleNamespace(grad_accum_iter=4, distributed_parallelism="fsdp"))
    result.callbacks = _Hooks()
    result.training_timer = lambda name: nullcontext()
    result.bind_grouped_stream(
        planner, producer, config_digest="config", loader_profile=LoaderProfile(workers, 1, True, 60)
    )
    result.cpu_loader_observer = lambda _: None
    model = NoisyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.00001, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1 / (1 + step / 500))
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    return result, model, optimizer, scheduler, scaler


def step(state, iteration):
    trainer, model, optimizer, scheduler, scaler = state
    for member in range(4):
        trainer.training_step(model, optimizer, scheduler, scaler, {}, iteration, member)


@pytest.fixture(autouse=True)
def cpu_boundaries(monkeypatch):
    monkeypatch.setattr(loader_module.RawRecipe, "from_producer", classmethod(lambda cls, p, c: SyntheticRecipe()))
    monkeypatch.setattr(trainer_module, "collate_grouped_native_batch", lambda payloads: {"payloads": payloads})
    monkeypatch.setattr(trainer_module.misc, "to", lambda value, **kwargs: value)
    # Native config fixtures may leave inference grad mode disabled globally.
    # Scope this toy training oracle and restore the incoming mode afterwards.
    with torch.enable_grad():
        yield


def test_four_logical_ranks_full_cpu_state_100_resume101_500():
    for rank in range(4):
        random.seed(19 + rank)
        np.random.seed(19 + rank)
        torch.manual_seed(19 + rank)
        source = make(rank)
        other = None
        try:
            for i in range(100):
                step(source, i)
            checkpoint = pickle.loads(
                pickle.dumps(
                    {
                        "model": source[1].state_dict(),
                        "optim": source[2].state_dict(),
                        "scheduler": source[3].state_dict(),
                        "scaler": source[4].state_dict(),
                        "dataloader": source[0].callbacks._callbacks[0].state_dict(),
                        "trainer": {
                            "iteration": 100,
                            "python_rng": random.getstate(),
                            "numpy_rng": np.random.get_state(),
                            "torch_rng": torch.get_rng_state(),
                            "cuda_rng": None,
                        },
                    }
                )
            )
            assert checkpoint["dataloader"]["format"] == trainer_module.LOADER_FORMAT
            assert set(checkpoint) == {"model", "optim", "scheduler", "scaler", "dataloader", "trainer"}
            for i in range(100, 500):
                step(source, i)
                if i == 100:
                    at101 = source[1].net.moe_gen.detach().clone()
            expected = source[1].net.moe_gen.detach().clone()
            expected_plan = source[0]._grouped_window.live
            expected_trace = source[1].seen[400:]
            source[0]._no_ttt_loader.close()
            other = make(rank)
            other[1].load_state_dict(checkpoint["model"])
            other[2].load_state_dict(checkpoint["optim"])
            other[3].load_state_dict(checkpoint["scheduler"])
            other[4].load_state_dict(checkpoint["scaler"])
            other[0].callbacks._callbacks[0].load_state_dict(checkpoint["dataloader"])
            other[0]._resume_required = True
            random.setstate(checkpoint["trainer"]["python_rng"])
            np.random.set_state(checkpoint["trainer"]["numpy_rng"])
            torch.set_rng_state(checkpoint["trainer"]["torch_rng"])
            for i in range(100, 500):
                step(other, i)
                if i == 100:
                    torch.testing.assert_close(other[1].net.moe_gen, at101, rtol=0, atol=0)
            torch.testing.assert_close(other[1].net.moe_gen, expected, rtol=0, atol=0)
            assert other[1].seen == expected_trace and other[0]._grouped_window.live == expected_plan
            assert source[3].state_dict() == other[3].state_dict()
            torch.testing.assert_close(
                source[2].state[source[1].net.moe_gen]["momentum_buffer"],
                other[2].state[other[1].net.moe_gen]["momentum_buffer"],
                rtol=0,
                atol=0,
            )
            assert other[0]._grouped_completed_iteration == 500
        finally:
            source[0]._no_ttt_loader.close()
            if other:
                other[0]._no_ttt_loader.close()


@pytest.mark.parametrize(
    "damage", ["workers", "prefetch", "rank", "world", "seed", "config", "corpus", "source", "format", "sidecar"]
)
def test_wrong_checkpoint_profile_never_starts_workers_or_native_model(damage):
    source = make()
    try:
        step(source, 0)
        saved = deepcopy(source[0].callbacks._callbacks[0].state_dict())
    finally:
        source[0]._no_ttt_loader.close()
    if damage in ("workers", "prefetch"):
        saved["loader_profile"][{"workers": "workers", "prefetch": "prefetch_factor"}[damage]] = 2
    elif damage in ("rank", "world", "seed"):
        index = {"rank": 0, "world": 1, "seed": 2}[damage]
        values = list(saved["profile"])
        values[index] += 1
        saved["profile"] = tuple(values)
    elif damage in ("config", "corpus", "source"):
        saved[
            {"config": "config_digest", "corpus": "cache_corpus_digest", "source": "source_binding_digest"}[damage]
        ] = "wrong"
    else:
        saved[damage] = "wrong"
    target = make()
    try:
        target[0].callbacks._callbacks[0].load_state_dict(saved)
        target[0]._resume_required = True
        with pytest.raises(ValueError, match="No-TTT"):
            step(target, 1)
        assert target[1].seen == [] and target[0]._no_ttt_loader.context.processes == []
    finally:
        target[0]._no_ttt_loader.close()
