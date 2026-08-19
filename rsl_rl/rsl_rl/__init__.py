# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Main module for the rsl_rl package."""

import os

PROJ_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

from .vec_env import VecEnv
from .rollout_storage import RolloutStorage
from .mlp import MLP
from .actor_critic_encoder import ActorCriticEncoder
from .ppo import PPO
from .on_policy_runner import OnPolicyRunner
from .utils import *

import sys
import types

# Compatibility aliases for external packages (such as Isaac Lab) expecting legacy submodule paths
def _register_submodule_alias(submodule_name: str, **attributes):
    full_module_name = f"rsl_rl.{submodule_name}"
    module_obj = types.ModuleType(full_module_name)
    for attr_name, attr_val in attributes.items():
        setattr(module_obj, attr_name, attr_val)
    sys.modules[full_module_name] = module_obj
    return module_obj

env = _register_submodule_alias("env", VecEnv=VecEnv)
runners = _register_submodule_alias("runners", OnPolicyRunner=OnPolicyRunner)
algorithms = _register_submodule_alias("algorithms", PPO=PPO)
modules = _register_submodule_alias("modules", ActorCriticEncoder=ActorCriticEncoder)
networks = _register_submodule_alias("networks", MLP=MLP)
storage = _register_submodule_alias("storage", RolloutStorage=RolloutStorage)
utils_submod = _register_submodule_alias(
    "utils",
    PROJ_ROOT_DIR=PROJ_ROOT_DIR,
    resolve_obs_groups=resolve_obs_groups,
    resolve_nn_activation=resolve_nn_activation,
    store_code_state=store_code_state,
    split_and_pad_trajectories=split_and_pad_trajectories,
)

__all__ = [
    "VecEnv",
    "RolloutStorage",
    "MLP",
    "ActorCriticEncoder",
    "PPO",
    "OnPolicyRunner",
    "PROJ_ROOT_DIR",
]


