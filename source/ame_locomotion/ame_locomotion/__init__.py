# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Python module serving as a project/extension template.
"""

import gymnasium as gym

gym.register(
    id="AME-Go2-Custom-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_env_cfg:UnitreeGo2CustomEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:Go2AMEPPORunnerCfg",
    },
)

gym.register(
    id="AME-Go2-Custom-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_env_cfg:UnitreeGo2CustomEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:Go2AMEPPORunnerCfg",
    },
)

gym.register(
    id="AME-G1-29DOF-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_29dof_env_cfg:G1RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:G1AMEPPORunnerCfg",
    },
)

gym.register(
    id="AME-G1-29DOF-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_29dof_env_cfg:G1RoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:G1AMEPPORunnerCfg",
    },
)


