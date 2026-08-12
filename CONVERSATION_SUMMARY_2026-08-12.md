# AME Locomotion 对话摘要（2026-08-12）

## 环境与兼容性

- 当前项目路径：`/home/brave/isaaclab_pj/AME_Locomotion`。
- 使用 Conda 环境 `isaaclab`，Isaac Lab 版本为 `0.54.4`，本地 `rsl-rl-lib` 版本为 `3.0.1`。
- 初始运行 `scripts/rsl_rl/play.py` 时出现：

  ```text
  ModuleNotFoundError: No module named 'isaaclab.utils.pretrained_checkpoint'
  ```

- 已对 `scripts/rsl_rl/play.py` 与 `scripts/rsl_rl/train.py` 做兼容调整：
  - 使用 `isaaclab_rl.utils.pretrained_checkpoint`；
  - 调用 `handle_deprecated_rsl_rl_cfg`；
  - 对 RSL-RL `< 4.0` 移除 `share_cnn_encoders`；
  - `--vis_attention` 可直接使用；保存权重需额外传递 `--save_attention_weights`。

## 当前 Go2 训练配置

配置文件：`source/ame_locomotion/ame_locomotion/tasks/manager_based/ame_locomotion/go2/velocity_env_cfg_go2.py`。

第一阶段 Go2 的速度命令为：

```python
resampling_time_range = (10.0, 10.0)
rel_heading_envs = 0.0
heading_command = False
lin_vel_x = (0.5, 1.5)
lin_vel_y = (0.0, 0.0)
ang_vel_z = (0.0, 0.0)
heading = None
```

含义：每 10 秒重采样一次前向线速度；命令在**机器人机身坐标系**中定义，始终要求它沿自身前方运动，不要求维持某个世界坐标系朝向。

当前 Go2 奖励的关键部分：

```python
track_lin_vel_xy_exp.weight = 2.0
track_ang_vel_z_exp.weight = 0.0
stuck.weight = -1.0
```

- `track_lin_vel_xy_exp` 依据机身系线速度跟踪计算。
- `stuck` 在 `command[:, 0] > 0.1` 且 `root_lin_vel_b[:, 0] < 0.1` 时才触发。
- `track_ang_vel_z_exp` 为零，因此持续偏航没有来自该项的约束。

## Play 中绿色箭头的含义

运行示例：

```bash
python scripts/rsl_rl/play.py \
  --task AME-Go2-Play-v0 \
  --checkpoint logs/rsl_rl/go2_ame/2026-08-12_09-43-19/model_1000.pt \
  --vis_attention
```

- 绿色箭头：目标速度 `base_velocity`。
- 蓝色箭头：机器人当前实际线速度。
- 速度命令首先在机器人机身坐标系中生成；Isaac Lab 可视化时，再使用机器人当前世界坐标系四元数将箭头转到世界坐标系。
- 因此，即使目标一直是 `vx > 0, vy = 0, yaw_rate = 0`，机器人自身发生转向时，绿色箭头在画面里的方向也会随机器人旋转。
- 绿色箭头转向不代表每一帧都产生了新的世界方向指令。

相关 Isaac Lab 实现：

`/home/brave/open_src/ssh_env_hub/task/isaaclab/IsaacLab/source/isaaclab/isaaclab/envs/mdp/commands/velocity_command.py`

- `_debug_vis_callback()` 可视化目标与实际速度；
- `_resolve_xy_velocity_to_arrow()` 中用 `root_quat_w` 把机身系箭头变换到世界系。

## 当前 Play 地形

`Go2AMEEnvCfg_PLAY` 当前使用固定的金字塔台阶地形：

```python
MeshPyramidStairsTerrainCfg(
    step_height_range=(0.15, 0.15),
    step_width=0.4,
    platform_width=3.0,
    border_width=1.0,
    holes=False,
)
```

此处最近从相机 + 固定 gap 调试地形改为固定台阶。该改动应视为用户已有的工作树修改，不要在无明确需求时覆盖。

## Go2 在台阶顶端绕圈的根因

现象：机器狗能跟随局部速度命令，却不从台阶顶端向下走；它留在安全的平整区域不断转向和绕圈，以降低跌落风险并继续获得奖励。

根因是任务目标存在奖励漏洞：

1. 线速度奖励只要求机身系前向速度符合 `vx`。
2. 机器人可先转身，再沿自己新的前方运动；在世界坐标系里形成绕圈，但局部速度奖励仍然很高。
3. `stuck` 只惩罚机身系前向速度过低；绕圈时它仍在前进，因此不会触发。
4. `track_ang_vel_z_exp` 权重为零，持续偏航没有直接的跟踪损失。
5. terrain curriculum 仅使用机器人到出生点的直线距离来更新难度，不验证是否沿着某一条下台阶路线移动。
6. 当前 checkpoint `model_1000.pt` 仍属于较早训练阶段；Play 的固定高台阶顶部/下行过渡也可能与训练分布不同。

因此，绕圈是当前奖励下的一个“安全局部最优解”，而不是命令发生错误。

## 推荐的改进方向

### 推荐：明确世界系目标或路点

为每个环境提供位于台阶下方的目标点，并使用世界系进度奖励：

```text
奖励 = 到目标点距离的减少量
     + 速度在目标方向上的投影
     - 偏航角速度惩罚
     - 跌倒、打滑等惩罚
```

这样机器人原地绕圈时不会缩短目标距离，反而会因偏航受到惩罚；只有朝下台阶方向实际推进才会获得主要奖励。

### 次选：固定世界朝向

启用 `heading_command`，生成固定的世界系 heading（指向台阶下方），并为偏航跟踪提供非零权重。此方案能明显减少绕圈，但它仍是“固定朝向行走”，不如目标点进度奖励明确。

### 辅助项：惩罚偏航

可增加 `ang_vel_z_l2`，或者为零 yaw 命令增加非零的 `track_ang_vel_z_exp` 权重，以抑制持续转圈。

不要只依赖这项惩罚：模型可能变成缓慢转圈、横向移动或其他规避方式，仍然没有学会下台阶。

## 已观察到的训练问题

此前 Go2 曾出现：

- 静止或三腿支撑、抬起一条腿仍可获得较高奖励；
- 对 gap 退缩或不愿通过；
- 线速度跟踪很差但 yaw 相关项占主导；
- 在早期调整后，线速度误差有所下降，但仍有大量接触终止。

训练日志中的高 episode reward 不应单独作为成功指标；需要同时检查：

- `Metrics/base_velocity/error_vel_xy`；
- 实际到目标/路线的世界系位移；
- 超时比例与接触终止比例；
- 各个奖励项的相对量级；
- Play 中跨越台阶与 gap 的视频表现。

## 下一步建议

1. 先决定任务目标是否应为“通用局部速度跟踪”，还是“从台阶顶端到指定下方目标点”。
2. 若目标是稳定通过固定台阶/gap，优先实现世界系路点与进度奖励，而不是仅调大局部速度奖励。
3. 训练新模型后，再用固定台阶 Play 场景进行可视化验证；不要以旧 checkpoint 的绕圈表现判断修改是否生效。
4. 保留当前已有工作树修改，并在提交前使用 `git diff` 确认仅包含预期内容。
