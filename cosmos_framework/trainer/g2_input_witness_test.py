# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""CPU-only native G2 input proof; no real model weights or private production assets."""
import json

import pytest
import torch

from cosmos_framework.model.generator.mot.g2_contracts import G2ConditionBatch, G2SampleIdentity
from cosmos_framework.model.generator.mot.g2_contract_fixtures import synthetic_text, synthetic_view
from cosmos_framework.trainer.g2_input_witness import (
    append_g2_input_witness,
    inspect_native_g2_inputs,
)


def _case(profile: str, *, dropped: bool = False):
    condition_text = synthetic_text(lengths=(3,), layers=1).text
    from dataclasses import replace

    condition_text = replace(condition_text, dropped=(dropped,))
    identity = G2SampleIdentity(
        episode_id="Task/episode_000001", scene_id="fixture-scene", subtask_id="fixture-subtask",
        decision_frame=5, logical_state_frame=4, data_version="certified-id",
        alignment_version="fixture-alignment", state15_recipe_id="fixture-state15",
        task_binding_id="task-raw-source",
    )
    action = torch.zeros(17, 15, dtype=torch.float32)
    action[0, :5] = torch.tensor([1.0, 2.0, 0.5, 0.0, 1.0])
    text = "" if dropped else json.dumps(
        {"actions": [{"description": "Task: pick. Initial spatial context (world frame): object at x=2m."}]}
    )
    item = {
        "g2_sample_identity": identity,
        "g2_provider_scope": "verified_data_c",
        "start_frame": 5,
        "text_task_identity": "source-text-id",
        "action_raw": action,
        "action": torch.nn.functional.pad(action, (0,49)),
        "text_token_ids": torch.tensor([1,2],dtype=torch.long),
        "ai_caption": text,
        "text_dropout_applied": dropped,
        "text_recipe_identity": condition_text.formatter_id,
    }
    view = synthetic_view()
    if profile == "R2":
        from dataclasses import replace as repl
        view = repl(view, identity=repl(view.identity, image_owner_frame=4))
    condition = G2ConditionBatch(text=condition_text, views=((view,) if profile == "R2" else None,))
    return [item], condition


@pytest.mark.parametrize("profile", ("R1","R2"))
@pytest.mark.parametrize("dropped", (False,True))
def test_native_inputs_capture_true_text_state_and_source_views(tmp_path, profile, dropped):
    items,condition=_case(profile, dropped=dropped)
    report=inspect_native_g2_inputs(items, condition, profile=profile, iteration=1, rank=0)
    assert report["profile"]==profile and report["optimizer_iteration"]==1
    assert report["samples"]==1 and report["enhanced_active_samples"]==int(not dropped)
    assert report["cfg_dropout_samples"]==int(dropped)
    assert report["selected_view_latents"]==(1 if profile=="R2" else 0)
    row=report["rows"][0]
    assert row["pose5_actual"] == [1.0,2.0,0.5,0.0,1.0]
    assert row["has_initial_spatial_context"] is not dropped
    if profile=="R2":
        assert row["selected_views"][0]["image_owner_frame"]==4
        assert row["selected_views"][0]["latent_shape"]==[4,2,3]
        assert row["selected_views"][0]["valid_pixels"]==6
    p=append_g2_input_witness(tmp_path,report,rank=0)
    assert json.loads(p.read_text().splitlines()[0])==report


def test_native_witness_rejects_original_instruction_without_cap2():
    items,condition=_case("R1")
    items[0]["ai_caption"]=json.dumps({"actions":[{"description":"pick lid"}]})
    with pytest.raises(ValueError,match="CAP2"):
        inspect_native_g2_inputs(items,condition,profile="R1",iteration=1,rank=0)


def test_native_witness_rejects_dropped_nonempty_and_bad_state15():
    items,condition=_case("R1",dropped=True)
    items[0]["ai_caption"]="not empty"
    with pytest.raises(ValueError,match="CFG-dropped"):
        inspect_native_g2_inputs(items,condition,profile="R1",iteration=1,rank=0)
    items,condition=_case("R1")
    items[0]["action_raw"][0,3:5]=0
    with pytest.raises(ValueError,match="Pose5"):
        inspect_native_g2_inputs(items,condition,profile="R1",iteration=1,rank=0)


def test_native_witness_rejects_missing_r2_latents_and_r1_spatial_leak():
    items,condition=_case("R2")
    from dataclasses import replace
    missing=replace(condition,views=(None,))
    with pytest.raises(ValueError,match="R2 training"):
        inspect_native_g2_inputs(items,missing,profile="R2",iteration=1,rank=0)
    with pytest.raises(ValueError,match="R1 must have no"):
        inspect_native_g2_inputs(items,condition,profile="R1",iteration=1,rank=0)


def test_witness_rejects_future_view():
    from dataclasses import replace
    items,condition=_case("R2")
    view=condition.views[0][0]
    condition=replace(condition,views=((replace(view,identity=replace(view.identity,image_owner_frame=5)),),))
    with pytest.raises(ValueError,match="view latent"):
        inspect_native_g2_inputs(items,condition,profile="R2",iteration=1,rank=0)
