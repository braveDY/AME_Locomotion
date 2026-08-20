# AGENTS.md - AME Locomotion 开发与服务器训练规范

## 一、 基础行为与沟通准则
- **中文回复**：始终使用中文回复用户。
- **审慎修改**：修改代码前必须先阅读相关文件，充分理解上下文。
- **副作用操作预告**：执行有副作用的操作（删除、覆盖文件、`git push` 等）前先向用户说明计划。
- **Git 提交规范**：Git 提交信息必须用英文，严格遵循 Conventional Commits 规范（如 `feat:`, `fix:`, `tune:`, `refactor:`, `chore:`）。
- **网络代理**：本地网络请求或 GitHub 操作超时时，使用 `/home/brave/docs/scripts/proxy.sh` 代理解决（端口 7897）。
- **容器方案**：当前项目使用容器开发验证，`src/elevation_mapping/em_gpu_humble.sh` 为局部高程建图容器启动脚本。

---

## 二、 本地开发到 GitHub 同步流程
每次在本地修改代码验证后，必须执行完整的提交和推送流程：
1. **代码暂存与提交**：
   ```bash
   git add <modified_files>
   git commit -m "<type>(<scope>): <english message>"
   ```
2. **通过代理推送至远端**：
   ```bash
   git -c http.proxy=socks5h://127.0.0.1:7897 -c https.proxy=socks5h://127.0.0.1:7897 push origin <branch_name>
   ```

---

## 三、 服务器端环境与代码同步流程
- **服务器主机**：`ssh lab`（用户 `ubuntu20@927`，显卡 RTX 4090 24GB）
- **宿主机项目路径**：`/home/ubuntu20/.braveDY/robot_rl/AME_Locomotion`
- **容器内项目路径**：`/robot_rl/AME_Locomotion`

### 服务器代码拉取
在服务器端拉取最新提交前，先清理未跟踪临时改动，再拉取最新代码：
```bash
ssh lab "cd /home/ubuntu20/.braveDY/robot_rl/AME_Locomotion && git checkout -- . && git pull origin <branch_name>"
```

---

## 四、 Docker 容器启动与管理规范
- **容器名**：`isaaclab_braveDY`
- **镜像**：`isaac-lab-base:latest`
- **工作空间挂载**：宿主机 `/home/ubuntu20/.braveDY/robot_rl` 挂载至容器 `/robot_rl`

> [!IMPORTANT]
> 容器启动时必须指定 `--entrypoint bash` 并配合 `-c "tail -f /dev/null"` 保持后台常驻。
> **严禁直接运行默认 entrypoint**（否则会在后台隐式自启 Isaac Sim streaming 服务，占用 3GB 显存与 300% CPU）。

### 容器创建/重置标准命令：
```bash
docker rm -f isaaclab_braveDY 2>/dev/null || true
docker run -d \
  --name isaaclab_braveDY \
  --gpus all \
  --network host \
  --ipc host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e ACCEPT_EULA=Y \
  -e PRIVACY_CONSENT=Y \
  -e DISPLAY=:0 \
  -e QT_X11_NO_MITSHM=1 \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /home/ubuntu20/.Xauthority:/root/.Xauthority:rw \
  -v /home/ubuntu20/.braveDY/robot_rl:/robot_rl \
  -w /robot_rl \
  --entrypoint bash \
  isaac-lab-base:latest \
  -c "tail -f /dev/null"
```

---

## 五、 Tmux 后台训练启动与监控流程

### 1. Python 解释器路径
容器内执行训练必须使用 IsaacLab 专用的解释器包装脚本：
`/workspace/isaaclab/_isaac_sim/python.sh` （挂载了 Omniverse Kit 与 USD 核心依赖）。

### 2. 启动 Tmux 训练会话
> [!IMPORTANT]
> **清理旧进程与显存保护**：
> 仅执行 `tmux kill-session` 无法保证容器内 `python` 进程被完全终止。若存在残留孤儿进程占用显存，会导致 PhysX 报 `CUDA error: out of memory` 或 `PhysX Internal CUDA error (code 2)`。
> 因此启动前必须在容器内显式执行 `pkill -9 -f python`，并确认显存已完全释放。

清理旧会话与残留进程，并在后台新建 `train` 会话（日志写入 `train.log`）：
```bash
ssh lab "tmux kill-session -t train 2>/dev/null || true; docker exec isaaclab_braveDY pkill -9 -f python 2>/dev/null || true; sleep 2; tmux new-session -d -s train 'docker exec -w /robot_rl/AME_Locomotion isaaclab_braveDY /workspace/isaaclab/_isaac_sim/python.sh scripts/rsl_rl/train.py --task AME-Go2-Custom-v0 --headless --num_envs 1024 --max_iterations 10000 2>&1 | tee /home/ubuntu20/.braveDY/robot_rl/AME_Locomotion/train.log'; sleep 2; tmux ls"
```

### 3. 监控与日常维护
- **附加并查看 tmux 实时控制台**：
  ```bash
  ssh lab
  tmux attach -t train
  ```
  *(脱离会话使用 `Ctrl + B` 然后按 `D`，不可按 `Ctrl + C` 避免中断训练)*
- **实时查看日志流**：
  ```bash
  ssh lab "tail -f /home/ubuntu20/.braveDY/robot_rl/AME_Locomotion/train.log"
  ```
- **GPU 资源状态监控**：
  ```bash
  ssh lab "nvidia-smi"
  ```
  *(正常训练状态下 RTX 4090 显存占用约 10~15GB，GPU 利用率处于 60%~100%，无多余僵尸进程)*

---

## 六、 常见故障与排查（Troubleshooting）

### 1. PhysX CUDA OOM / Internal CUDA error (code 2)
- **现象**：日志输出 `PhysX failed to allocate GPU memory - aborting simulation` 或 `CUDA error: out of memory: mGpuContactPairsDev`。
- **原因**：上一次训练的 Isaac Sim 进程未在容器内彻底退出，占用显存导致新进程无法分配 CUDA 内存。
- **解决办法**：
  1. 在容器内强杀所有 python 进程：
     ```bash
     ssh lab "docker exec isaaclab_braveDY pkill -9 -f python"
     ```
  2. 若仍未释放，执行**第四节的标准命令**重置/重建 `isaaclab_braveDY` 容器。
  3. 执行 `ssh lab "nvidia-smi"` 确认显存空闲 > 20GB 后再重新启动训练。

### 2. Git 拉取冲突或未跟踪修改阻止同步
- **现象**：`git pull origin <branch>` 提示本地有未暂存修改或冲突。
- **解决办法**：
  ```bash
  ssh lab "cd /home/ubuntu20/.braveDY/robot_rl/AME_Locomotion && git checkout -- . && git clean -fd && git pull origin <branch>"
  ```
