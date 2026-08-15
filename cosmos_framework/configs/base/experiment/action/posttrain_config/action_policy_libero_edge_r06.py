"""Edge-Policy-DROID LIBERO task-0 latent-cache SFT baseline."""

import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_libero_edge_warmstart import (
    action_policy_libero_edge_warmstart,
)


action_policy_libero_edge_r06 = copy.deepcopy(action_policy_libero_edge_warmstart)
action_policy_libero_edge_r06["job"].update(name="action_policy_libero_edge_r06")
dataset_cfg = action_policy_libero_edge_r06["trainer"]["dataloader"]["datasets"]["libero"]["dataset"]
dataset_cfg.update(
    task_index=0,
    use_latent_cache=True,
    latent_cache_root="${oc.env:LIBERO_LATENT_CACHE_ROOT,}",
    latent_cache_parity_path="${oc.env:LIBERO_LATENT_CACHE_PARITY,}",
)
action_policy_libero_edge_r06["trainer"].update(max_iter=500, logging_iter=1, grad_accum_iter=1)

ConfigStore.instance().store(
    group="experiment",
    package="_global_",
    name="action_policy_libero_edge_r06",
    node=action_policy_libero_edge_r06,
)
