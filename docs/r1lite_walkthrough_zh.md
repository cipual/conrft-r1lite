# R1Lite ConRFT 使用流程

本文档整理 R1Lite 当前支持的两条数据到训练链路。

- 链路 A：在 Gym 环境中用 SpaceMouse 采集示教数据，可选做 transition
  replay 验证，然后进行 offline 和 online ConRFT 训练。
- 链路 B：用官方主从臂 / SARM 数据采集，转换到 LeRobot，训练 SARM
  reward model，重新标注并导出 ConRFT 数据，可选做 replay 验证，然后进行
  offline 和 online ConRFT 训练。

英文版见 [r1lite_walkthrough.md](./r1lite_walkthrough.md)。

## 通用准备

`RWRL` 环境用于 ConRFT、Octo embeddings、replay、机器人 env、actor 和
learner 训练：

```bash
source /home/ps/Applications/miniforge3/etc/profile.d/conda.sh
conda activate RWRL
cd /home/ps/VLA-RL/conrft-r1lite
export ROBOT=http://192.168.12.12:8001
```

`lerobot` 环境用于 LeRobot 数据集导出、SARM 标注、SARM reward model 训练和
SARM progress 计算：

```bash
source /home/ps/Applications/miniforge3/etc/profile.d/conda.sh
conda activate lerobot
cd /home/ps/VLA-RL/conrft-r1lite
export HF_LEROBOT_HOME=/home/ps/VLA-RL/conrft-r1lite/data/lerobot
```

只要涉及在线 env 采集、在线 replay、actor rollout 或 online training，都需要
先启动 R1Lite body service。实验配置优先读取 `ROBOT` 环境变量，其次使用
`config.yaml` 里的 `env.server_url`。

## 共同约定

flatten 后的 proprio state 使用统一的 `gym_sorted` 顺序。离线 PKL 和在线 env
observation 必须一致。

双臂任务顺序为：

```text
left/gripper_pose, left/joint_pos, left/joint_vel, left/tcp_pose, left/tcp_vel,
right/gripper_pose, right/joint_pos, right/joint_vel, right/tcp_pose, right/tcp_vel,
torso_pos
```

R1Lite EEF action 语义为：

- 平移：归一化 delta，env 内乘以 `control.xyz_scale`
- 旋转：归一化 `euler_left` XYZ delta，env 内乘以 `control.rot_scale`
- 夹爪：归一化后的下一帧绝对夹爪目标，范围为 `[-1, 1]`

env 中旋转执行公式为：

```python
Rotation.from_euler("xyz", delta) * current_rotation
```

旧版 SARM/LeRobot 转换如果生成的是 `rotvec_right` action 标签，需要重新生成
PKL 后再 replay 或训练。

## 链路 A：Gym + SpaceMouse 示教

这个链路适合 reward 可以直接由 Gym env 给出的任务。参考任务是
`r1lite_reach_target`。

### A1. 配置任务

主要文件：

- [config.yaml](../examples/experiments/r1lite_reach_target/config.yaml)
- [config.py](../examples/experiments/r1lite_reach_target/config.py)
- [wrapper.py](../examples/experiments/r1lite_reach_target/wrapper.py)

关键配置项：

| 配置组 | 含义 | 常见取值 |
| --- | --- | --- |
| `env.server_url` | robot service URL，会被 `ROBOT` 覆盖 | `http://192.168.12.12:8001/` |
| `env.max_episode_length` | 单条 rollout 最大步数 | 正整数，常用 `1000` |
| `env.default_mode` | body service 控制模式 | `ee_pose_servo` |
| `env.reset_*` | reset 位姿 / 关节和到位阈值 | 按任务设置 |
| `env.abs_pose_limit_low/high` | action target 安全边界 | 米 + 弧度 |
| `control.hz` | env 控制频率 | 正浮点数，常用 `10.0` |
| `control.xyz_scale` | 一个归一化 xyz action 对应的米数 | 正浮点数，常用 `0.03` |
| `control.rot_scale` | 一个归一化旋转 action 对应的弧度 | 正浮点数，常用 `0.20` |
| `train.arm` | 任务使用的机械臂 | `left`、`right`、`dual`；定点任务用 `right` |
| `train.image_keys` | Octo 使用的图像观测 | 定点任务用 `image_primary,image_wrist` |
| `train.octo_path` | Octo checkpoint | `hf://rail-berkeley/octo-small-1.5` |
| `task.*` | reward 和成功条件 | target pose、阈值、reward 权重 |
| `teleop.*` | SpaceMouse 标定和死区 | 非负浮点数 |
| `gripper.fixed_open` | 固定夹爪张开并忽略夹爪动作 | 定点任务为 `true` |

### A2. 采集示教数据

从实验目录启动：

```bash
cd /home/ps/VLA-RL/conrft-r1lite/examples/experiments/r1lite_reach_target
export SUCCESS_COUNT=20
export OUTPUT_DIR=/home/ps/VLA-RL/conrft-r1lite/data/transition/r1lite_reach_target
bash run_record_demos_octo.sh
```

等价的直接命令：

```bash
cd /home/ps/VLA-RL/conrft-r1lite/examples
python record_demos_r1lite_octo.py \
  --exp_name=r1lite_reach_target \
  --successes_needed=20 \
  --output_dir=/home/ps/VLA-RL/conrft-r1lite/data/transition/r1lite_reach_target
```

`run_record_demos_octo.sh` 参数：

| 名称 | 类型 / 范围 | 含义 |
| --- | --- | --- |
| `SUCCESS_COUNT` | 正整数 | 需要保留的成功轨迹数量 |
| `OUTPUT_DIR` | 路径 | ConRFT `.pkl` 示教数据输出目录 |
| 额外 CLI 参数 | 透传 | 追加给 `record_demos_r1lite_octo.py` |

`record_demos_r1lite_octo.py` 入口参数：

| 参数 | 类型 / 范围 | 含义 |
| --- | --- | --- |
| `--exp_name` | `examples/experiments` 下的实验名 | 选择 config 和 env wrapper |
| `--successes_needed` | 正整数 | 收到多少条成功轨迹后停止 |
| `--reward_scale` | 浮点数 | 计算 `mc_returns` 时的 reward 乘数 |
| `--reward_bias` | 浮点数 | 计算 `mc_returns` 时的 reward 偏置 |
| `--output_dir` | 路径或省略 | 默认写到 `examples/experiments/<exp_name>/demo_data` |
| `--reset_wait_sec` | 非负秒数 | reset 后等待操作员稳定场景的时间 |

输出是 ConRFT transition PKL，包含 observations、actions、rewards、dones、
`mc_returns`、Octo `embeddings` 和 `next_embeddings`。

### A3. 可选：Replay 验证

生成 PKL 后、训练前，建议先跑 replay。

列出轨迹：

```bash
cd /home/ps/VLA-RL/conrft-r1lite/examples
python replay_transition_r1lite.py \
  --exp_name=r1lite_reach_target \
  --input_file=/path/to/reach_target_demos.pkl \
  --list_only
```

offline action replay 应该能用接近 0 的误差重建 recorded next TCP pose：

```bash
python replay_transition_r1lite.py \
  --exp_name=r1lite_reach_target \
  --input_file=/path/to/reach_target_demos.pkl \
  --trajectory_index=0 \
  --exec_mode=offline \
  --replay_mode=action \
  --output_csv=/tmp/reach_replay_errors.csv \
  --output_summary_json=/tmp/reach_replay_summary.json
```

Replay 脚本参数：

| 参数 | 类型 / 可选值 | 含义 |
| --- | --- | --- |
| `--exp_name` | 实验名 | 选择 env config 和 action 语义 |
| `--input_file` | `.pkl` 路径 | 要检查或 replay 的 transition 文件 |
| `--trajectory_index` | 非负整数 | 按 `dones` 切分后的轨迹编号 |
| `--all_trajectories` | flag | 仅 offline，处理全部轨迹 |
| `--list_only` | flag | 只打印轨迹摘要后退出 |
| `--exec_mode` | `offline`、`online` | 只计算，或真实下发机器人 |
| `--replay_mode` | `action`、`state` | 积分 action，或发送记录的 state target |
| `--offline_reference` | `teacher_forced`、`rollout` | 每步用 recorded current state，或用积分出的 target |
| `--start_step` | 非负整数 | 从第几步开始 |
| `--max_steps` | 正整数或省略 | 最多 replay 多少步 |
| `--no_reset_before` | flag | 仅 online，replay 前不 reset |
| `--no_reset_after` | flag | 仅 online，replay 后不 reset |
| `--reset_wait_sec` | 非负秒数 | reset 后等待时间 |
| `--log_every` | 正整数 | 每 N 步打印一次 |
| `--output_csv` | 路径或省略 | 每步误差表 |
| `--output_npz` | 路径或省略 | 数值数组 |
| `--output_summary_json` | 路径或省略 | 误差统计摘要 |

### A4. Offline 训练

运行 learner-only 预训练：

```bash
cd /home/ps/VLA-RL/conrft-r1lite/examples/experiments/r1lite_reach_target
export DEMO_PATH=/path/to/reach_target_demos.pkl
export CHECKPOINT_PATH=/home/ps/VLA-RL/conrft-r1lite/examples/experiments/r1lite_reach_target/conrft
bash run_learner_conrft_pretrain.sh
```

`run_learner_conrft_pretrain.sh` 从 `config.yaml` 的 `offline_training` 读取默认值。

| 环境变量 / flag | 类型 / 范围 | 含义 |
| --- | --- | --- |
| `DEMO_PATH` / `--demo_path` | 路径；直接 CLI 可重复传 | offline demo PKL |
| `CHECKPOINT_PATH` / `--checkpoint_path` | 路径 | checkpoint 输出目录 |
| `PRETRAIN_STEPS` / `--pretrain_steps` | 正整数 | offline learner update 步数 |
| `Q_WEIGHT` / `--q_weight` | 非负浮点数 | actor Q-guidance 权重 |
| `BC_WEIGHT` / `--bc_weight` | 非负浮点数 | behavior cloning 权重 |
| `TRAIN_DEBUG` / `--debug` | bool | 为 true 时关闭 W&B |
| `XLA_PYTHON_CLIENT_MEM_FRACTION` | `(0, 1]` 浮点数 | JAX 显存占用比例 |

训练主脚本参数：

| 参数 | 可选值 / 类型 | 含义 |
| --- | --- | --- |
| `--exp_name` | 实验名 | 选择 config |
| `--learner` | flag | 启动 learner 进程 |
| `--actor` | flag | 启动 actor 进程 |
| `--ip` | host/IP | actor 连接的 learner 地址 |
| `--demo_path` | 可重复路径 | learner 加载的 demo buffer |
| `--checkpoint_path` | 路径 | checkpoint 读写目录 |
| `--seed` | 整数 | 随机种子 |
| `--gamma` | `[0, 1]` 浮点数 | 折扣因子 |
| `--reward_scale`、`--reward_bias`、`--reward_neg` | 浮点数 | reward 变换 |
| `--eval_checkpoint_step` | 非负整数 | evaluation 模式使用的 checkpoint step |
| `--eval_n_trajs` | 正整数 | evaluation 轨迹数量 |

### A5. Online 训练

learner 和 actor 分别开两个终端。

Learner：

```bash
cd /home/ps/VLA-RL/conrft-r1lite/examples/experiments/r1lite_reach_target
export DEMO_PATH=/path/to/reach_target_demos.pkl
export CHECKPOINT_PATH=/home/ps/VLA-RL/conrft-r1lite/examples/experiments/r1lite_reach_target/conrft
bash run_learner_conrft.sh
```

Actor：

```bash
cd /home/ps/VLA-RL/conrft-r1lite/examples/experiments/r1lite_reach_target
export CHECKPOINT_PATH=/home/ps/VLA-RL/conrft-r1lite/examples/experiments/r1lite_reach_target/conrft
bash run_actor_conrft.sh --ip=localhost
```

online 默认值来自 `config.yaml` 的 `online_training`。

| 配置项 | 类型 / 范围 | 含义 |
| --- | --- | --- |
| `online_training.checkpoint_path` | 路径 | 从哪个 offline checkpoint 继续 |
| `online_training.demo_path` | 路径 | online 阶段继续混合的 demo PKL |
| `online_training.pretrain_steps` | 正整数 | resume 时使用的 pretrain step 边界 |
| `online_training.batch_size` | 正整数 | learner batch size |
| `online_training.training_starts` | 非负整数 | 收到多少 online transitions 后开始更新 |
| `online_training.steps_per_update` | 正整数 | 更新频率 |
| `online_training.learner.q_weight/bc_weight` | 非负浮点数 | online actor loss 权重 |
| `online_training.actor.xla_mem_fraction` | `(0, 1]` 浮点数 | actor JAX 显存占用比例 |

## 链路 B：官方主从臂 + SARM Reward

这个链路适合长程、视觉 reward 更自然的任务。参考任务是
`r1lite_dual_mango_box`。

### B1. RAW Episode 导出到 LeRobot

在 `lerobot` 环境运行：

```bash
cd /home/ps/VLA-RL/conrft-r1lite
python examples/sarm/export_rosbag_to_lerobot_sarm.py \
  --input_dirs=/home/ps/VLA-RL/conrft-r1lite/data/RAW/r1lite_dual_mango_box \
  --task_name=r1lite_dual_mango_box \
  --task_desc="左臂抓住白色的框放在右臂的周围，右臂抓住发红的芒果，把它放入框内，然后左右机械臂复位。" \
  --fps=10 \
  --action_space=eef \
  --output_repo_id=r1lite_dual_mango_box \
  --output_dir=/home/ps/VLA-RL/conrft-r1lite/data/lerobot/r1lite_dual_mango_box \
  --overwrite
```

导出脚本参数：

| 参数 | 类型 / 可选值 | 含义 |
| --- | --- | --- |
| `--input_dirs` | 逗号分隔路径 | RAW episode 目录或父目录 |
| `--raw_dir_glob` | glob 字符串 | 输入为父目录时匹配子目录，默认 `*_RAW` |
| `--recursive` | flag | 递归扫描父目录 |
| `--task_name` | 字符串 | 写入数据集 metadata 的任务名 |
| `--task_desc` | 字符串 | 语言任务描述 |
| `--fps` | 正浮点数 | 重采样频率 |
| `--action_space` | `eef`、`joint` | LeRobot 中保存的 action label 类型 |
| `--output_repo_id` | 字符串 | 本地或 Hugging Face 风格的数据集 id |
| `--output_dir` / `--root` | 路径 | 本地数据集目录 |
| `--overwrite` | flag | 先删除已有输出目录 |
| `--no_videos` | flag | 直接存图片，不写 MP4 |
| `--vcodec` | codec 字符串 | 视频编码，默认 `h264` |
| `--image_writer_threads` | 正整数 | LeRobot 图片写入线程数 |
| `--<key>_topic` | ROS topic | 覆盖 `head`、`left_wrist`、`right_wrist`、TCP、joint、gripper topic |
| `--dry_run_manifest` | JSON 路径 | 只写 manifest，不创建数据集 |

用于 SARM 标注时建议保留视频列，不要使用 `--no_videos`。

### B2. 标注并训练 SARM Reward Model

手动标注 UI：

```bash
cd /home/ps/VLA-RL/conrft-r1lite
python examples/sarm/manual_annotate_sarm.py \
  --dataset_root=/home/ps/VLA-RL/conrft-r1lite/data/lerobot/r1lite_dual_mango_box \
  --video_key=observation.images.head \
  --port=8020
```

手动标注参数：

| 参数 | 类型 / 范围 | 含义 |
| --- | --- | --- |
| `--dataset_root` | 路径 | 本地 LeRobotDataset 根目录 |
| `--repo_id` | 字符串 | 配合 `--root` / `HF_LEROBOT_HOME` 查找数据集 |
| `--root` | 路径 | LeRobot cache root |
| `--annotations_file` | 路径 | sidecar JSON，默认在数据集 `meta/` 下 |
| `--task_desc` | 字符串 | 给标注工具显示的任务描述 |
| `--fps` | 正浮点数或省略 | 覆盖数据集 fps |
| `--video_key` | 列名 | 标注 UI 使用的视频流 |
| `--episodes` | 整数列表 | 只标注指定 episode |
| `--sparse_subtasks` | 逗号分隔字符串 | sparse SARM 标签 |
| `--dense_subtasks` | 逗号分隔字符串 | dense SARM 标签 |
| `--overwrite_subtasks` | flag | 替换已有 subtask 定义 |
| `--no_backup` | flag | 不备份 parquet / proportion 文件 |
| `--prepare_only` | flag | 只写 sidecar 后退出 |
| `--host` | host | UI 绑定地址 |
| `--port` | `1..65535` 整数 | UI 端口 |
| `--no_browser` | flag | 不自动打开浏览器 |

在 LeRobot 仓库中训练 SARM：

```bash
cd /home/ps/VLA-RL/lerobot
export HF_LEROBOT_HOME=/home/ps/VLA-RL/conrft-r1lite/data/lerobot
export PYTHONPATH=/home/ps/VLA-RL/lerobot/src:$PYTHONPATH

lerobot-train \
  --dataset.repo_id=r1lite_dual_mango_box \
  --policy.type=sarm \
  --policy.annotation_mode=dual \
  --policy.state_key=observation.state \
  --policy.frame_gap=10 \
  --policy.push_to_hub=false \
  --output_dir=/home/ps/VLA-RL/conrft-r1lite/examples/sarm/outputs/train/r1lite_dual_mango_box_sarm \
  --batch_size=4 \
  --steps=5000 \
  --num_workers=6
```

关键训练参数：

| 参数 | 类型 / 范围 | 含义 |
| --- | --- | --- |
| `--dataset.repo_id` | 数据集 id | 训练使用的 LeRobot 数据集 |
| `--policy.type` | `sarm` | reward model policy 类型 |
| `--policy.annotation_mode` | `sparse`、`dense`、`dual` | 训练哪些 annotation head |
| `--policy.state_key` | 列名 | state 列，通常是 `observation.state` |
| `--policy.frame_gap` | 正整数 | SARM 输入的时间间隔 |
| `--output_dir` | 路径 | 训练输出目录 |
| `--batch_size` | 正整数 | batch size |
| `--steps` | 正整数 | 训练步数 |
| `--num_workers` | 非负整数 | dataloader worker 数 |

计算逐帧 SARM progress：

```bash
cd /home/ps/VLA-RL/lerobot
python -m lerobot.policies.sarm.compute_rabc_weights \
  --dataset-repo-id r1lite_dual_mango_box \
  --reward-model-path /home/ps/VLA-RL/conrft-r1lite/examples/sarm/outputs/train/r1lite_dual_mango_box_sarm/checkpoints/005000/pretrained_model \
  --head-mode dense \
  --output-path /home/ps/VLA-RL/conrft-r1lite/data/lerobot/r1lite_dual_mango_box/sarm_progress.parquet \
  --output-dir /home/ps/VLA-RL/conrft-r1lite/examples/sarm/outputs/rabc_viz \
  --stride 1
```

Progress 计算参数：

| 参数 | 类型 / 可选值 | 含义 |
| --- | --- | --- |
| `--dataset-repo-id` | 数据集 id | LeRobot 数据集 id |
| `--reward-model-path` | 路径 | 训练好的 SARM `pretrained_model` 目录 |
| `--head-mode` | `sparse`、`dense` | 使用哪个 reward head 计算 progress |
| `--output-path` | parquet 路径 | 输出 `sarm_progress.parquet` |
| `--output-dir` | 路径 | 可选可视化输出目录 |
| `--stride` | 正整数 | 每 N 帧评估一次 |

### B3. 导出 SARM Reward ConRFT PKL

在 `lerobot` 环境运行：

```bash
cd /home/ps/VLA-RL/conrft-r1lite
python examples/sarm/relabel_rosbag_or_conrft_with_sarm_reward.py \
  --source_lerobot_dataset=/home/ps/VLA-RL/conrft-r1lite/data/lerobot/r1lite_dual_mango_box \
  --config_yaml=/home/ps/VLA-RL/conrft-r1lite/examples/experiments/r1lite_dual_mango_box/config.yaml \
  --output_pkl=/home/ps/VLA-RL/conrft-r1lite/data/transition/r1lite_dual_mango_box/r1lite_dual_mango_box_sarm_reward_no_octo.pkl \
  --head_mode=dense \
  --success_threshold=0.95 \
  --success_reward=10.0 \
  --include_images
```

Relabel / export 参数：

| 参数 | 类型 / 可选值 | 含义 |
| --- | --- | --- |
| `--source_lerobot_dataset` | 路径 | 包含 `sarm_progress.parquet` 的 LeRobotDataset |
| `--input_pkl` | 路径或省略 | legacy 模式：只重写已有 ConRFT PKL reward |
| `--progress_parquet` | 路径或省略 | 显式 progress 文件；默认在 dataset root 下 |
| `--sarm_model` | 路径或省略 | 只记录 metadata |
| `--head_mode` | `sparse`、`dense` | 使用哪个 progress 列 |
| `--output_pkl` | 路径 | 输出 ConRFT PKL |
| `--success_threshold` | 浮点数，通常 `0..1` | progress 达到多少算成功 |
| `--success_reward` | 浮点数 | 成功 bonus |
| `--gamma` | `[0, 1]` 浮点数 | Monte Carlo return 折扣 |
| `--reward_scale`、`--reward_bias` | 浮点数 | dense reward 变换 |
| `--reward_clip_low/high` | 浮点数 | progress delta 缩放前裁剪范围 |
| `--no_truncate_after_success` | flag | 成功后继续保留后续帧 |
| `--obs_horizon` | 正整数 | state / image 历史长度 |
| `--include_images` | flag | 输出 PKL 包含图像历史 |
| `--image_key_map` | `out=column,...` | ConRFT 图像 key 到 LeRobot 视频列的映射 |
| `--image_max_width` | 整数 | 图像缩放宽度；`<=0` 保持原分辨率 |
| `--embedding_mode` | `none`、`zeros` | smoke test 的占位 embedding 模式 |
| `--allow_zero_embeddings` | flag | 使用 `--embedding_mode=zeros` 时必须传 |
| `--config_yaml` | 路径 | 读取 `control.*` 和 `gripper.*` scale |
| `--action_space` | `eef` | ConRFT env action 类型 |
| `--xyz_scale`、`--rot_scale` | 正浮点数 | 覆盖 YAML action scale |
| `--gripper_max` | 正浮点数 | 映射到归一化 `+1` 的真实夹爪值 |
| `--no_clip_action` | flag | 不把归一化 action label 裁剪到 `[-1, 1]` |
| `--max_episodes`、`--max_transitions` | 正整数或省略 | 限制导出数量 |

生成的 action 使用 `euler_left` 旋转语义，state layout 为 canonical
`gym_sorted`。

在 `RWRL` 环境添加真实 Octo embeddings：

```bash
cd /home/ps/VLA-RL/conrft-r1lite
python examples/sarm/add_octo_embeddings_to_conrft_pkl.py \
  --exp_name=r1lite_dual_mango_box \
  --input_pkl=/home/ps/VLA-RL/conrft-r1lite/data/transition/r1lite_dual_mango_box/r1lite_dual_mango_box_sarm_reward_no_octo.pkl \
  --output_pkl=/home/ps/VLA-RL/conrft-r1lite/data/transition/r1lite_dual_mango_box/r1lite_dual_mango_box_sarm_reward.pkl
```

Embedding 脚本参数：

| 参数 | 类型 / 可选值 | 含义 |
| --- | --- | --- |
| `--exp_name` | 实验名 | 选择 image keys 和 Octo config |
| `--input_pkl` | 路径 | 没有真实 embeddings 的 PKL |
| `--output_pkl` | 路径 | 最终训练 PKL |
| `--image_keys` | 逗号分隔 key 或省略 | 默认使用实验 config |
| `--jax_platform` | `auto`、`cpu`、`gpu` | JAX backend 选择 |
| `--xla_mem_fraction` | `(0, 1]` 浮点数 | 默认 JAX GPU 显存比例 |
| `--start_trajectory` | 非负整数 | 从第几条轨迹开始处理 |
| `--max_trajectories` | 正整数或省略 | 最多处理多少条轨迹 |
| `--max_transitions_per_trajectory` | 正整数或省略 | 每条轨迹最多处理多少 transition |

### B4. 可选：Replay 验证

训练前建议先跑 offline action replay：

```bash
cd /home/ps/VLA-RL/conrft-r1lite/examples
python replay_transition_r1lite.py \
  --exp_name=r1lite_dual_mango_box \
  --input_file=/home/ps/VLA-RL/conrft-r1lite/data/transition/r1lite_dual_mango_box/r1lite_dual_mango_box_sarm_reward.pkl \
  --trajectory_index=0 \
  --exec_mode=offline \
  --replay_mode=action \
  --output_csv=/tmp/mango_replay_errors.csv \
  --output_summary_json=/tmp/mango_replay_summary.json
```

有效 PKL 中，action 积分出来的 target 应该和 recorded next TCP pose 在数值精度
范围内一致。

### B5. Offline 训练

运行 learner-only 预训练：

```bash
cd /home/ps/VLA-RL/conrft-r1lite/examples/experiments/r1lite_dual_mango_box
export DEMO_PATH=/home/ps/VLA-RL/conrft-r1lite/data/transition/r1lite_dual_mango_box/r1lite_dual_mango_box_sarm_reward.pkl
export CHECKPOINT_PATH=/home/ps/VLA-RL/conrft-r1lite/examples/experiments/r1lite_dual_mango_box/conrft_sarm
bash run_learner_conrft_pretrain.sh
```

offline 参数和链路 A 相同。默认值来自
[config.yaml](../examples/experiments/r1lite_dual_mango_box/config.yaml) 的
`offline_training`。

### B6. 使用 SARM Reward 的 Online 训练

在 `lerobot` 环境启动 SARM progress sidecar：

```bash
cd /home/ps/VLA-RL/conrft-r1lite
export PYTHONPATH=/home/ps/VLA-RL/lerobot/src:$PYTHONPATH
python examples/sarm/sarm_progress_sidecar.py \
  --reward_model_path=/home/ps/VLA-RL/conrft-r1lite/examples/sarm/outputs/train/r1lite_dual_mango_box_sarm/checkpoints/005000/pretrained_model \
  --host=127.0.0.1 \
  --port=8010 \
  --device=cuda \
  --default_head_mode=dense
```

Sidecar 参数：

| 参数 | 类型 / 可选值 | 含义 |
| --- | --- | --- |
| `--reward_model_path` | 路径 | 训练好的 SARM `pretrained_model` 目录 |
| `--host` | host | 绑定地址 |
| `--port` | `1..65535` 整数 | HTTP 端口 |
| `--device` | torch device 字符串 | `cuda`、`cpu`、`cuda:0` 等 |
| `--default_head_mode` | `sparse`、`dense` | 请求中没传 head 时使用哪个 |
| `--log_level` | 日志级别 | `INFO`、`DEBUG` 等 |

检查实验中的 reward model 配置：

```yaml
reward_model:
  enabled: true
  log_only: true
  endpoint_url: "http://127.0.0.1:8010"
  head_mode: "dense"
```

建议先用 `log_only: true` 只记录 SARM 诊断，不替换 env reward。确认无误后，
如果希望 online RL 使用 SARM reward，再改为 `log_only: false`。

learner 和 actor 分别在两个 `RWRL` 终端启动。

Learner：

```bash
cd /home/ps/VLA-RL/conrft-r1lite/examples/experiments/r1lite_dual_mango_box
export DEMO_PATH=/home/ps/VLA-RL/conrft-r1lite/data/transition/r1lite_dual_mango_box/r1lite_dual_mango_box_sarm_reward.pkl
export CHECKPOINT_PATH=/home/ps/VLA-RL/conrft-r1lite/examples/experiments/r1lite_dual_mango_box/conrft_sarm
bash run_learner_conrft.sh
```

Actor：

```bash
cd /home/ps/VLA-RL/conrft-r1lite/examples/experiments/r1lite_dual_mango_box
export CHECKPOINT_PATH=/home/ps/VLA-RL/conrft-r1lite/examples/experiments/r1lite_dual_mango_box/conrft_sarm
bash run_actor_conrft.sh --ip=localhost
```

online 参数和链路 A 相同，默认值来自 mango 任务的 `online_training` 配置段。

