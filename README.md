# AME Locomotion: Unitree Go2

基于 **Isaac Lab** 与 **RSL-RL** 的 **Attention-Based Map Encoding (AME)** 四足机器人运动控制复现工程，专为 **Unitree Go2** 在复杂崎岖地形上的高动态行走设计。

---

## 🌟 项目亮点

* **精简扁平化架构**：彻底精简无用模块，剔除冗余依赖，专注 Go2 AME 算法训练与验证。
* **严格对齐 AME 架构**：
  * **高程 CNN 提取**：使用保持空间分辨率的 Conv2d 处理单通道地形高度图。
  * **显式位置编码**：将局部点云 $(x, y)$ 坐标与 CNN 局部特征拼接，构建 64 维空间感知向量。
  * **跨模态多头注意力 (MHA)**：以本体感知特征为 Query，在地形空间特征上进行动态注意力加权聚合。
* **丰富地形预设**：集成连续斜坡、正反金字塔台阶、随机方块网格、梅花桩、导轨等多种地形生成器。
* **全流程支持**：涵盖从仿真训练、3D/2D 注意力可视化、到 TorchScript / ONNX 部署策略导出的完整工具链。

---

## 📁 目录结构

```text
AME_Locomotion/
├── source/ame_locomotion/ame_locomotion/   # 任务与环境模块
│   ├── __init__.py                         # Gym 任务注册（Go2 与 G1 29DOF 任务）
│   ├── go2_env_cfg.py                      # Go2 仿真场景、奖励函数、域随机化及地形预设
│   ├── g1_29dof_env_cfg.py                 # G1 29DOF 仿真场景、全身协同奖励与地形配置
│   ├── ppo_cfg.py                          # PPO 算法与 AME 神经网络超参数配置
│   ├── mdp/                                # 自定义 MDP 项（奖励函数、观测组、速度指令）
│   └── terrains/                           # 自定义复杂高度图地形生成器
├── rsl_rl/rsl_rl/                          # 扁平化 RSL-RL 强化学习库
│   ├── __init__.py                         # 顶层包导出与根路径定义
│   ├── actor_critic_encoder.py             # AME 核心网络（CNN + Positional Encoding + MHA）
│   ├── ppo.py                              # 高效 On-Policy PPO 算法核心
│   ├── on_policy_runner.py                 # 采样、训练调度与 TensorBoard 日志记录器
│   ├── rollout_storage.py                  # 经验回放池与 GAE 优势计算
│   ├── mlp.py                              # MLP 骨干网络
│   ├── utils.py                            # 激活函数解析与观测组校验工具
│   └── vec_env.py                          # 向量化环境抽象接口
├── scripts/                                # 脚本工具集
│   ├── list_envs.py                        # 查看已注册的 Gym 任务
│   ├── plot_attention.py                   # 注意力热图离线渲染脚本
│   └── rsl_rl/
│       ├── train.py                        # 强化学习训练主入口
│       ├── play.py                         # 策略评估、录屏与实时 3D 注意力可视化
│       ├── exporter.py                     # 策略导出工具 (JIT / ONNX)
│       └── cli_args.py                     # 命令行参数解析
└── README.md
```

---

## 🛠️ 安装指南

### 1. 前置依赖
* **NVIDIA Isaac Sim** (4.0+ 或容器镜像)
* **Isaac Lab** (1.0+)
* **PyTorch** (带 CUDA 支持)

### 2. 安装本地包

在项目根目录下，以可编辑模式（editable mode）安装：

```bash
python -m pip install -e source/ame_locomotion
python -m pip install -e rsl_rl
```

---

## 🚀 快速上手

### 1. 启动训练

使用单卡启动 Go2 AME 训练：

```bash
# 快速启动（无渲染模式）
python scripts/rsl_rl/train.py \
  --task AME-Go2-Custom-v0 \
  --headless \
  --num_envs 4096 \
  --max_iterations 10000
```

> **提示**：训练日志与模型权重将默认保存在 `logs/rsl_rl/go2_ame/<时间戳>/` 目录下。可通过 `tensorboard --logdir logs/rsl_rl/` 查看实时训练曲线。

### 2. 策略评估与回放

运行训练好的检查点，并开启实时 3D 注意力可视化与视频录制：

```bash
rsync -avzP --ignore-existing lab:/home/ubuntu20/.braveDY/robot_rl/AME_Locomotion/logs/ ./logs/

python scripts/rsl_rl/play.py \
  --task AME-Go2-Custom-Play-v0 \
  --checkpoint logs/rsl_rl/go2_ame/<run_folder>/model_<iteration>.pt \
  --num_envs 1 \
  --vis_attention \
  --save_attention_weights \
  --video \
  --video_length 300

python scripts/rsl_rl/play.py \
  --task AME-Go2-Custom-Play-v0 \
  --checkpoint logs/rsl_rl/go2_ame/2026-08-20_09-45-49/model_9999.pt \
  --vis_attention

python scripts/rsl_rl/play.py \
  --task AME-G1-29DOF-Play-v0 \
  --checkpoint logs/rsl_rl/g1_ame/2026-08-21_03-59-45/model_3000.pt \
  --vis_attention 
```

* `--vis_attention`：在仿真窗口中以不同深浅颜色的 3D Marker 实时绘制机器人足底当前聚焦的地形区域。
* `--save_attention_weights`：将采集到的注意力权重保存至本地 `attention_weights.npy`。

### 3. 注意力热图渲染

渲染注意力随时间变化的 2D 俯视高程热图：

```bash
python scripts/plot_attention.py
```
渲染后的热图序列将保存在 `attn_vis/` 目录下。

### 4. 导出 Policy（TorchScript / ONNX）

将训练好的 `.pt` 权重导出为跨平台部署格式：

```bash
# 导出指定 Checkpoint
python scripts/rsl_rl/play.py \
  --task AME-Go2-Custom-Play-v0 \
  --checkpoint logs/rsl_rl/go2_ame/<run_folder>/model_<iteration>.pt \
  --export
```
导出的 `policy.pt` (TorchScript) 与 `policy.onnx` 将自动存放于对应实验的 `exported/` 文件夹内。

---

## ⚙️ 地形配置自定义

在 `source/ame_locomotion/ame_locomotion/go2_env_cfg.py` 的 `UnitreeGo2CustomEnvCfg_PLAY` 中，提供了分门别类的地形预设，只需通过取消/添加注释即可灵活切换：

```python
self.scene.terrain.terrain_generator.sub_terrains = {
    # 1. 经典粗糙地形
    "random_rough": terrain_gen.HfRandomUniformTerrainCfg(proportion=1.0, noise_range=(0.02, 0.10), noise_step=0.02),
    # "boxes": terrain_gen.MeshRandomGridTerrainCfg(proportion=1.0, grid_width=0.45, grid_height_range=(0.05, 0.2)),
    # "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(proportion=1.0, step_height_range=(0.05, 0.23), step_width=0.3),
    # 2. 几何网格障碍 (Gaps, Pits, Rails, Stepping Stones 等)
    # "mesh_gap": terrain_gen.MeshGapTerrainCfg(proportion=1.0, gap_width_range=(0.3, 0.6)),
}
```

---

## 🧠 AME 地形编码器架构

```text
       [ 高度通道 (z) : 1 x 33 x 21 ] ──> Conv2d (5x5) ──> [ 62 x 33 x 21 ]
                                                                 │
       [ 局部坐标 (x, y) : 2 x 33 x 21 ] ────────────────────────┴──> [ 64 x 33 x 21 ]
                                                                          │ (Flatten)
                                                             Key / Value: [ 693 x 64 ]
                                                                          │
  [ 本体感知观测 (Proprioception) ] ──> Linear ──> Query: [ 1 x 64 ] ────┴──> Multi-Head Attention
                                                                                   │
                                                                           [ 融合特征输出 ] ──> Actor / Critic MLP
```
