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

__all__ = [
    "VecEnv",
    "RolloutStorage",
    "MLP",
    "ActorCriticEncoder",
    "PPO",
    "OnPolicyRunner",
    "PROJ_ROOT_DIR",
]

