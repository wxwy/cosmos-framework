from __future__ import annotations

import torch

from cosmos_framework.model.generator.mot.parallelize_vfm_network import (
    _local_memory_fsdp_ignored_parameters,
)


def test_local_memory_fsdp_ignored_parameters_cover_only_local_slow_groups() -> None:
    model = torch.nn.Module()
    model.baseline = torch.nn.Linear(4, 4)
    model.local_memory_runtime = torch.nn.Sequential(torch.nn.Linear(3, 5), torch.nn.LayerNorm(5))
    model.local_memory2llm = torch.nn.Linear(5, 7)
    model.local_memory_modality_embed = torch.nn.Parameter(torch.zeros(7))

    ignored = _local_memory_fsdp_ignored_parameters(model)
    expected = {id(parameter) for parameter in model.local_memory_runtime.parameters()}
    expected |= {id(parameter) for parameter in model.local_memory2llm.parameters()}
    expected.add(id(model.local_memory_modality_embed))

    assert {id(parameter) for parameter in ignored} == expected
    assert all(id(parameter) not in expected for parameter in model.baseline.parameters())
