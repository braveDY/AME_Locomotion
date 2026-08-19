# AME Locomotion (Unitree Go2)

基于 Isaac Lab 与 RSL-RL 的 Attention-Based Map Encoding（AME）四足机器人运动控制项目，专注 Unitree Go2 崎岖地形行走训练。

## 安装

```bash
python -m pip install -e source/ame_locomotion
python -m pip install -e rsl_rl
```

## 快速开始

### 1. 训练 Go2 AME

```bash
bash run_train.sh
# 或指定参数
python scripts/rsl_rl/train.py --task AME-Go2-Custom-v0 --headless --num_envs 2048 --max_iterations 10000
```

### 2. 回放与注意力可视化

```bash
bash run_play.sh --checkpoint logs/rsl_rl/go2_ame/<run_dir>/model_<step>.pt
```

### 3. 注意力热图绘制

```bash
python scripts/plot_attention.py
```



python scripts/rsl_rl/play.py \
  --task AME-G1-29DOF-Play-v0 \
  --checkpoint pretrained/ame1.pt \
  --vis_attention
