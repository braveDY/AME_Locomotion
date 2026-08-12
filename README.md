# AME Locomotion

基于 Isaac Lab 与 RSL-RL 的 Attention-Based Map Encoding（AME）腿式运动复现项目，包含 Unitree G1 29DoF 与 Go2 任务。

## 环境

- Isaac Sim / Isaac Lab
- Python 包：`ame_locomotion`、本仓库的 `rsl_rl`
- AME 编码器：[rsl_rl/rsl_rl/modules/actor_critic_encoder.py](rsl_rl/rsl_rl/modules/actor_critic_encoder.py)

安装扩展：

```bash
python -m pip install -e source/ame_locomotion
python -m pip install -e rsl_rl
```

## 训练与评估

### G1 29DoF

```bash
bash run_train.sh
python scripts/rsl_rl/play.py \
  --task AME-G1-29DOF-Play-v0 \
  --checkpoint pretrained/ame1.pt \
  --vis_attention
```

G1 使用两阶段训练。第一阶段结束后，将
[velocity_env_cfg_29dof.py](source/ame_locomotion/ame_locomotion/tasks/manager_based/ame_locomotion/29dof/velocity_env_cfg_29dof.py)
中的 `FINETUNE` 改为 `True`，再从第一阶段 checkpoint 继续训练。

### Go2

```bash
python scripts/rsl_rl/train.py --task AME-Go2-v0 --headless

python scripts/rsl_rl/play.py \
  --task AME-Go2-Play-v0 \
  --checkpoint <checkpoint路径> \
  --vis_attention
```

首次运行可附加 `--num_envs 16 --max_iterations 50` 验证资产、足端接触和高度扫描器。Go2 不能直接加载 G1 checkpoint。

## 当前 Go2 任务

配置：[velocity_env_cfg_go2.py](source/ame_locomotion/ame_locomotion/tasks/manager_based/ame_locomotion/go2/velocity_env_cfg_go2.py)

- 线速度：机身系前向 `0.5–1.5 m/s`，每 10 秒重采样；
- 航向：固定世界系 `+x`，通过 heading controller 生成角速度指令；
- 终点：从起点向世界系 `+x` 偏移 `3 m`；仅在正常超时时按完成进度发放奖励；
- Play：固定金字塔台阶，便于观察下台阶行为。

Go2 复杂地形奖励参考 `wmp_cnn`：限幅速度跟踪、足端腾空时间、足端滑移、绊脚、台阶边缘接触与终止惩罚。具体实现见
[rewards.py](source/ame_locomotion/ame_locomotion/tasks/manager_based/ame_locomotion/mdp/rewards.py)。

## AME 实现

- 高程图输入：`33 × 21 × 3`；
- 编码流程：CNN 下采样 → 局部地形特征与本体状态的多头注意力融合；
- 可选全局上下文：MLP + max-pool（AME2 风格）；
- 预训练模型：`pretrained/ame1.pt`、`pretrained/ame2.pt`。

## 常用文件

- G1 配置：[29dof/velocity_env_cfg_29dof.py](source/ame_locomotion/ame_locomotion/tasks/manager_based/ame_locomotion/29dof/velocity_env_cfg_29dof.py)
- Go2 配置：[go2/velocity_env_cfg_go2.py](source/ame_locomotion/ame_locomotion/tasks/manager_based/ame_locomotion/go2/velocity_env_cfg_go2.py)
- 自定义 MDP：[mdp/rewards.py](source/ame_locomotion/ame_locomotion/tasks/manager_based/ame_locomotion/mdp/rewards.py)
- 训练脚本：[scripts/rsl_rl/train.py](scripts/rsl_rl/train.py)
- 评估脚本：[scripts/rsl_rl/play.py](scripts/rsl_rl/play.py)
