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

rsync -avzP --ignore-existing isaaclab:/root/IsaacLab/logs .
