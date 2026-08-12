# Go2 训练摘要（2026-08-12）

## 环境

- 项目：`/home/brave/isaaclab_pj/AME_Locomotion`
- Conda 环境：`isaaclab`
- Isaac Lab：`0.54.4`；RSL-RL：`3.0.1`

`train.py` 与 `play.py` 已适配当前 RSL-RL：使用 `isaaclab_rl.utils.pretrained_checkpoint`，并移除 RSL-RL 3.x 不支持的 `share_cnn_encoders`。

## Go2 当前设计

- 前向速度指令：机身系 `0.5–1.5 m/s`，10 秒重采样；
- 固定航向：世界系 `+x`，由 heading controller 转为角速度指令；
- 终点：reset 起点前方 `3 m`；只有超时才按完成进度给奖励；
- Play 地形：固定 `0.15 m` 金字塔台阶。

## 奖励重点

- 使用 `wmp_cnn` 风格的限幅线速度跟踪、足端腾空时间、滑移、绊脚与边缘接触惩罚；
- 不使用静态关节同步、足端空中时间方差或固定相位 trot 约束；
- 目标是让策略根据台阶落脚情况自适应，而不是用固定关节角或固定周期“伪造”步态。

## 验证

训练时重点观察：

- `Metrics/base_velocity/error_vel_xy`；
- 到世界系 `+x` 终点的实际进度；
- timeout 与接触终止占比；
- 后腿是否离地、足端是否踩在台阶边缘；
- Play 的固定台阶视频。

旧 checkpoint 不会反映新的命令和奖励设计，需重新训练或从较早阶段继续训练。
