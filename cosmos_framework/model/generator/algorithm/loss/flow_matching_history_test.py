from __future__ import annotations

import torch

from cosmos_framework.model.generator.algorithm.loss.flow_matching import compute_flow_matching_loss


class _UnitWeightFlow:
    def train_time_weight(self, timesteps, tensor_kwargs):
        return torch.ones_like(torch.as_tensor(timesteps), **{
            key: value for key, value in tensor_kwargs.items() if key in {"device", "dtype"}
        })


def test_exclude_fully_conditioned_items_preserves_target_loss_scale():
    # Item 0 is a fully-clean native history control. Item 1 mirrors the WAM
    # target item: latent frame 0 clean, four future latent frames noisy.
    pred = [
        torch.ones(1, 1, 1, 1),
        torch.ones(1, 5, 1, 1),
    ]
    target = [torch.zeros_like(item) for item in pred]
    condition = [
        torch.ones(1, 1, 1),
        torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0]).reshape(5, 1, 1),
    ]
    timesteps = torch.ones(2, 1)

    diluted, per_item = compute_flow_matching_loss(
        pred=pred,
        target=target,
        condition_mask=condition,
        timesteps=timesteps,
        has_valid_tokens=True,
        rectified_flow=_UnitWeightFlow(),
        tensor_kwargs_fp32={"dtype": torch.float32},
        exclude_fully_conditioned_items=False,
    )
    matched, per_item_matched = compute_flow_matching_loss(
        pred=pred,
        target=target,
        condition_mask=condition,
        timesteps=timesteps,
        has_valid_tokens=True,
        rectified_flow=_UnitWeightFlow(),
        tensor_kwargs_fp32={"dtype": torch.float32},
        exclude_fully_conditioned_items=True,
    )

    torch.testing.assert_close(per_item, torch.tensor([0.0, 0.8]))
    torch.testing.assert_close(per_item_matched, per_item)
    torch.testing.assert_close(diluted, torch.tensor(0.4))
    torch.testing.assert_close(matched, torch.tensor(0.8))
