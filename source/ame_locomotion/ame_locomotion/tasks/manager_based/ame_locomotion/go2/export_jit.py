# /home/brave/isaaclab_pj/AME_Locomotion/source/ame_locomotion/ame_locomotion/tasks/manager_based/ame_locomotion/go2/export_jit.py

import sys

# 将项目根目录和 rsl_rl 脚本目录加入 sys.path
# 脚本从项目根目录运行，所以直接用相对路径即可
sys.path.insert(0, 'scripts/rsl_rl')
sys.path.insert(0, 'rsl_rl')  # rsl_rl 包本身

from exporter import export_policy_as_jit
import torch

# 加载训练 checkpoint
ckpt = torch.load('logs/rsl_rl/go2_ame/2026-08-12_23-52-36/model_3999.pt', map_location='cpu', weights_only=False)

# 重建模型 (不需要 isaaclab_tasks，直接手动构造即可)
from rsl_rl.modules.actor_critic_encoder import ActorCriticEncoder

# 观测空间信息
# 训练时 policy 观测顺序:
#   base_ang_vel(3) + projected_gravity(3) + velocity_commands(3) + joint_pos(12) + joint_vel(12) + actions(12) = 45
#   height_scan = 33*21*3 = 2079
#   total = 2124
# critic 多了 base_lin_vel(3), proprio = 48, total = 2127
obs_groups = {
    'policy': ['base_ang_vel', 'projected_gravity', 'velocity_commands',
               'joint_pos', 'joint_vel', 'actions', 'height_scan'],
    'critic': ['base_lin_vel', 'base_ang_vel', 'projected_gravity',
               'velocity_commands', 'joint_pos', 'joint_vel', 'actions', 'height_scan'],
}

num_envs = 1
obs_dict = {
    'base_ang_vel': torch.zeros(num_envs, 3),
    'projected_gravity': torch.zeros(num_envs, 3),
    'velocity_commands': torch.zeros(num_envs, 3),   # lin_vel_x, lin_vel_y, ang_vel_z (不含 heading)
    'joint_pos': torch.zeros(num_envs, 12),
    'joint_vel': torch.zeros(num_envs, 12),
    'actions': torch.zeros(num_envs, 12),
    'height_scan': torch.zeros(num_envs, 2079),
    'base_lin_vel': torch.zeros(num_envs, 3),
}

policy = ActorCriticEncoder(
    obs=obs_dict,
    obs_groups=obs_groups,
    num_actions=12,
    actor_hidden_dims=[512, 256, 128],
    critic_hidden_dims=[512, 256, 128],
    activation='elu',
    init_noise_std=1.0,
    map_scan_dim=(33, 21, 3),
    mha_dim=64,
    num_heads=16,
    cnn_downsample=True,
    attach_global=False,
)
policy.load_state_dict(ckpt['model_state_dict'], strict=False)
policy.eval()

# 导出 JIT
export_policy_as_jit(policy, normalizer=None, path='logs/rsl_rl/go2_ame/2026-08-12_23-52-36/exported', filename='policy.pt')
print('Export done!')