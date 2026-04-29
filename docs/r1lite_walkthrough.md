# R1Lite ConRFT Walkthrough

This document describes two supported data-to-training paths for R1Lite.

- Path A: collect demonstrations inside the Gym environment with SpaceMouse,
  optionally replay the saved transitions, then run offline and online ConRFT.
- Path B: collect official leader-follower / SARM data, convert it to
  LeRobot, train a SARM reward model, relabel/export ConRFT data, optionally
  replay the transitions, then run offline and online ConRFT.

The Chinese version is [r1lite_walkthrough_zh.md](./r1lite_walkthrough_zh.md).

## Common Setup

Use `RWRL` for ConRFT, Octo embeddings, replay, robot envs, and actor/learner
training:

```bash
source /home/ps/Applications/miniforge3/etc/profile.d/conda.sh
conda activate RWRL
cd /home/ps/VLA-RL/conrft-r1lite
export ROBOT=http://192.168.12.12:8001
```

Use `lerobot` for LeRobot dataset export, SARM annotation/training, and SARM
progress computation:

```bash
source /home/ps/Applications/miniforge3/etc/profile.d/conda.sh
conda activate lerobot
cd /home/ps/VLA-RL/conrft-r1lite
export HF_LEROBOT_HOME=/home/ps/VLA-RL/conrft-r1lite/data/lerobot
```

The R1Lite body service must be running before online env collection, online
replay, actor rollout, or online training. The experiment configs read
`ROBOT` first, then fall back to `env.server_url` in `config.yaml`.

## Shared Conventions

The canonical flattened proprio layout is `gym_sorted`. Offline PKLs and online
env observations must match this order.

For dual-arm tasks:

```text
left/gripper_pose, left/joint_pos, left/joint_vel, left/tcp_pose, left/tcp_vel,
right/gripper_pose, right/joint_pos, right/joint_vel, right/tcp_pose, right/tcp_vel,
torso_pos
```

R1Lite EEF action semantics are:

- translation: normalized delta, multiplied by `control.xyz_scale`
- rotation: normalized `euler_left` XYZ delta, multiplied by `control.rot_scale`
- gripper: normalized absolute next gripper target in `[-1, 1]`

The env applies rotation as:

```python
Rotation.from_euler("xyz", delta) * current_rotation
```

Existing SARM/LeRobot-derived PKLs generated with older `rotvec_right` action
labels should be regenerated before replay or training.

## Path A: Gym + SpaceMouse Demonstrations

Use this path for tasks that can be rewarded directly by the Gym env. The
reference example is `r1lite_reach_target`.

### A1. Configure The Task

Main files:

- [config.yaml](../examples/experiments/r1lite_reach_target/config.yaml)
- [config.py](../examples/experiments/r1lite_reach_target/config.py)
- [wrapper.py](../examples/experiments/r1lite_reach_target/wrapper.py)

Important config groups:

| Group | Meaning | Typical values |
| --- | --- | --- |
| `env.server_url` | robot service URL, overridden by `ROBOT` | `http://192.168.12.12:8001/` |
| `env.max_episode_length` | max rollout length | positive int, often `1000` |
| `env.default_mode` | body-service control mode | `ee_pose_servo` |
| `env.reset_*` | reset pose/joint and reset tolerances | task-specific |
| `env.abs_pose_limit_low/high` | action target safety box | meters + radians |
| `control.hz` | env control frequency | positive float, often `10.0` |
| `control.xyz_scale` | meters per normalized xyz action unit | positive float, often `0.03` |
| `control.rot_scale` | radians per normalized rotation action unit | positive float, often `0.20` |
| `train.arm` | active arm | `left`, `right`, or `dual`; reach target uses `right` |
| `train.image_keys` | image observations used by Octo | reach target uses `image_primary,image_wrist` |
| `train.octo_path` | Octo checkpoint | `hf://rail-berkeley/octo-small-1.5` |
| `task.*` | reward and success definition | target pose, tolerances, reward weights |
| `teleop.*` | SpaceMouse calibration/deadzone | non-negative floats |
| `gripper.fixed_open` | keep gripper open and ignore gripper actions | `true` for reach target |

### A2. Collect Demonstrations

Run from the experiment directory:

```bash
cd /home/ps/VLA-RL/conrft-r1lite/examples/experiments/r1lite_reach_target
export SUCCESS_COUNT=20
export OUTPUT_DIR=/home/ps/VLA-RL/conrft-r1lite/data/transition/r1lite_reach_target
bash run_record_demos_octo.sh
```

Equivalent direct command:

```bash
cd /home/ps/VLA-RL/conrft-r1lite/examples
python record_demos_r1lite_octo.py \
  --exp_name=r1lite_reach_target \
  --successes_needed=20 \
  --output_dir=/home/ps/VLA-RL/conrft-r1lite/data/transition/r1lite_reach_target
```

`run_record_demos_octo.sh` parameters:

| Name | Type / range | Meaning |
| --- | --- | --- |
| `SUCCESS_COUNT` | positive int | number of successful trajectories to keep |
| `OUTPUT_DIR` | path | output directory for ConRFT `.pkl` demos |
| extra CLI args | forwarded | appended to `record_demos_r1lite_octo.py` |

`record_demos_r1lite_octo.py` entry parameters:

| Argument | Type / range | Meaning |
| --- | --- | --- |
| `--exp_name` | experiment name in `examples/experiments` | selects config and env wrapper |
| `--successes_needed` | positive int | stop after this many successful demos |
| `--reward_scale` | float | multiplier used when computing `mc_returns` |
| `--reward_bias` | float | additive reward bias for `mc_returns` |
| `--output_dir` | path or omitted | defaults to `examples/experiments/<exp_name>/demo_data` |
| `--reset_wait_sec` | non-negative float seconds | wait after reset before the next rollout |

Output is a ConRFT transition PKL with observations, actions, rewards, dones,
`mc_returns`, Octo `embeddings`, and `next_embeddings`.

### A3. Optional Replay Validation

Use replay after generating a PKL and before training.

List trajectories:

```bash
cd /home/ps/VLA-RL/conrft-r1lite/examples
python replay_transition_r1lite.py \
  --exp_name=r1lite_reach_target \
  --input_file=/path/to/reach_target_demos.pkl \
  --list_only
```

Offline action replay should reconstruct recorded next TCP poses with near-zero
error:

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

Replay script parameters:

| Argument | Type / choices | Meaning |
| --- | --- | --- |
| `--exp_name` | experiment name | selects env config and action semantics |
| `--input_file` | `.pkl` path | transition file to inspect or replay |
| `--trajectory_index` | int >= 0 | trajectory after splitting by `dones` |
| `--all_trajectories` | flag | offline only, process every trajectory |
| `--list_only` | flag | print trajectory summary and exit |
| `--exec_mode` | `offline`, `online` | compute only, or command the robot |
| `--replay_mode` | `action`, `state` | integrate actions, or send recorded state targets |
| `--offline_reference` | `teacher_forced`, `rollout` | use recorded current state each step, or integrated target |
| `--start_step` | int >= 0 | first step to replay |
| `--max_steps` | positive int or omitted | cap number of replayed steps |
| `--no_reset_before` | flag | online only, skip reset before replay |
| `--no_reset_after` | flag | online only, skip reset after replay |
| `--reset_wait_sec` | non-negative float | wait after reset |
| `--log_every` | positive int | print every N steps |
| `--output_csv` | path or omitted | per-step error table |
| `--output_npz` | path or omitted | numeric arrays |
| `--output_summary_json` | path or omitted | summary statistics |

### A4. Offline Training

Run learner-only pretraining:

```bash
cd /home/ps/VLA-RL/conrft-r1lite/examples/experiments/r1lite_reach_target
export DEMO_PATH=/path/to/reach_target_demos.pkl
export CHECKPOINT_PATH=/home/ps/VLA-RL/conrft-r1lite/examples/experiments/r1lite_reach_target/conrft
bash run_learner_conrft_pretrain.sh
```

`run_learner_conrft_pretrain.sh` reads defaults from
`offline_training` in `config.yaml`.

| Env var / flag | Type / range | Meaning |
| --- | --- | --- |
| `DEMO_PATH` / `--demo_path` | path, repeatable in direct CLI | offline demo PKL |
| `CHECKPOINT_PATH` / `--checkpoint_path` | path | checkpoint output directory |
| `PRETRAIN_STEPS` / `--pretrain_steps` | positive int | number of offline learner updates |
| `Q_WEIGHT` / `--q_weight` | float >= 0 | actor Q-guidance weight |
| `BC_WEIGHT` / `--bc_weight` | float >= 0 | behavior cloning weight |
| `TRAIN_DEBUG` / `--debug` | bool | disables W&B when true |
| `XLA_PYTHON_CLIENT_MEM_FRACTION` | float in `(0, 1]` | JAX memory fraction |

Direct training script flags:

| Argument | Choices / type | Meaning |
| --- | --- | --- |
| `--exp_name` | experiment name | selects config |
| `--learner` | flag | run learner process |
| `--actor` | flag | run actor process |
| `--ip` | host/IP | learner address used by actors |
| `--demo_path` | repeatable path | demo buffers loaded by learner |
| `--checkpoint_path` | path | load/save checkpoint directory |
| `--seed` | int | random seed |
| `--gamma` | float in `[0, 1]` | discount |
| `--reward_scale`, `--reward_bias`, `--reward_neg` | float | reward transforms |
| `--eval_checkpoint_step` | int >= 0 | checkpoint step for evaluation mode |
| `--eval_n_trajs` | positive int | number of eval trajectories |

### A5. Online Training

Start learner and actor in separate terminals.

Learner:

```bash
cd /home/ps/VLA-RL/conrft-r1lite/examples/experiments/r1lite_reach_target
export DEMO_PATH=/path/to/reach_target_demos.pkl
export CHECKPOINT_PATH=/home/ps/VLA-RL/conrft-r1lite/examples/experiments/r1lite_reach_target/conrft
bash run_learner_conrft.sh
```

Actor:

```bash
cd /home/ps/VLA-RL/conrft-r1lite/examples/experiments/r1lite_reach_target
export CHECKPOINT_PATH=/home/ps/VLA-RL/conrft-r1lite/examples/experiments/r1lite_reach_target/conrft
bash run_actor_conrft.sh --ip=localhost
```

Online defaults come from `online_training` in `config.yaml`.

| Config field | Type / range | Meaning |
| --- | --- | --- |
| `online_training.checkpoint_path` | path | offline checkpoint to resume |
| `online_training.demo_path` | path | demo PKL mixed with online data |
| `online_training.pretrain_steps` | positive int | checkpoint step boundary for resume |
| `online_training.batch_size` | positive int | learner batch size |
| `online_training.training_starts` | int >= 0 | online transitions before learner updates |
| `online_training.steps_per_update` | positive int | update cadence |
| `online_training.learner.q_weight/bc_weight` | float >= 0 | online actor loss weights |
| `online_training.actor.xla_mem_fraction` | float in `(0, 1]` | actor JAX memory fraction |

## Path B: Official Leader-Follower + SARM Reward

Use this path when the task reward is long-horizon or visual. The reference
example is `r1lite_dual_mango_box`.

### B1. Export RAW Episodes To LeRobot

Run in the `lerobot` environment:

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

Export script parameters:

| Argument | Type / choices | Meaning |
| --- | --- | --- |
| `--input_dirs` | comma-separated paths | RAW episode dirs or parent dirs |
| `--raw_dir_glob` | glob string | child pattern when parent dirs are passed, default `*_RAW` |
| `--recursive` | flag | recursively scan parent dirs |
| `--task_name` | string | task name saved in dataset metadata |
| `--task_desc` | string | language task description |
| `--fps` | positive float | resampling frequency |
| `--action_space` | `eef`, `joint` | action labels to store in LeRobot |
| `--output_repo_id` | string | local/HF-style LeRobot dataset id |
| `--output_dir` / `--root` | path | local dataset directory |
| `--overwrite` | flag | remove existing output dataset first |
| `--no_videos` | flag | store images directly instead of videos |
| `--vcodec` | codec string | video codec, default `h264` |
| `--image_writer_threads` | positive int | LeRobot image writer threads |
| `--<key>_topic` | ROS topic | override `head`, `left_wrist`, `right_wrist`, TCP, joint, gripper topics |
| `--dry_run_manifest` | JSON path | write manifest only, do not create dataset |

For SARM annotation, keep videos enabled. Browser-based annotation expects
video columns.

### B2. Annotate And Train The SARM Reward Model

Manual annotation UI:

```bash
cd /home/ps/VLA-RL/conrft-r1lite
python examples/sarm/manual_annotate_sarm.py \
  --dataset_root=/home/ps/VLA-RL/conrft-r1lite/data/lerobot/r1lite_dual_mango_box \
  --video_key=observation.images.head \
  --port=8020
```

Manual annotation parameters:

| Argument | Type / range | Meaning |
| --- | --- | --- |
| `--dataset_root` | path | local LeRobotDataset root |
| `--repo_id` | string | LeRobot repo id, used with `--root` / `HF_LEROBOT_HOME` |
| `--root` | path | LeRobot cache root |
| `--annotations_file` | path | sidecar JSON, defaults under dataset `meta/` |
| `--task_desc` | string | task instruction shown to annotation tools |
| `--fps` | positive float or omitted | override dataset fps |
| `--video_key` | column name | video stream for annotation UI |
| `--episodes` | list of ints | restrict annotation to selected episodes |
| `--sparse_subtasks` | comma-separated strings | sparse SARM labels |
| `--dense_subtasks` | comma-separated strings | dense SARM labels |
| `--overwrite_subtasks` | flag | replace existing subtask definitions |
| `--no_backup` | flag | skip parquet/proportion backup |
| `--prepare_only` | flag | write sidecar and exit |
| `--host` | host | UI bind host |
| `--port` | int in `1..65535` | UI port |
| `--no_browser` | flag | do not open browser automatically |

Train SARM from the LeRobot repo:

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

Key SARM training parameters:

| Argument | Type / range | Meaning |
| --- | --- | --- |
| `--dataset.repo_id` | dataset id | LeRobot dataset to train on |
| `--policy.type` | `sarm` | reward model policy type |
| `--policy.annotation_mode` | `sparse`, `dense`, `dual` | which annotation heads to train |
| `--policy.state_key` | column name | state column, normally `observation.state` |
| `--policy.frame_gap` | positive int | temporal frame gap for SARM inputs |
| `--output_dir` | path | training output directory |
| `--batch_size` | positive int | training batch size |
| `--steps` | positive int | training steps |
| `--num_workers` | int >= 0 | dataloader workers |

Compute per-frame SARM progress:

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

Progress computation parameters:

| Argument | Type / choices | Meaning |
| --- | --- | --- |
| `--dataset-repo-id` | dataset id | LeRobot dataset id |
| `--reward-model-path` | path | trained SARM `pretrained_model` directory |
| `--head-mode` | `sparse`, `dense` | reward head used for progress |
| `--output-path` | parquet path | writes `sarm_progress.parquet` |
| `--output-dir` | path | optional visual output directory |
| `--stride` | positive int | evaluate every N frames |

### B3. Export SARM-Reward ConRFT PKL

Run in the `lerobot` environment:

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

Relabel/export parameters:

| Argument | Type / choices | Meaning |
| --- | --- | --- |
| `--source_lerobot_dataset` | path | LeRobotDataset containing `sarm_progress.parquet` |
| `--input_pkl` | path or omitted | legacy mode: relabel existing ConRFT PKL rewards only |
| `--progress_parquet` | path or omitted | explicit progress file; defaults under dataset root |
| `--sarm_model` | path or omitted | metadata only |
| `--head_mode` | `sparse`, `dense` | progress column to consume |
| `--output_pkl` | path | output ConRFT PKL |
| `--success_threshold` | float, usually `0..1` | progress value treated as success |
| `--success_reward` | float | bonus added on success |
| `--gamma` | float in `[0, 1]` | Monte Carlo return discount |
| `--reward_scale`, `--reward_bias` | float | dense reward transform |
| `--reward_clip_low/high` | float | clip progress delta before scaling |
| `--no_truncate_after_success` | flag | keep frames after success |
| `--obs_horizon` | positive int | state/image history length |
| `--include_images` | flag | include image histories in output PKL |
| `--image_key_map` | `out=column,...` | map ConRFT image keys to LeRobot video columns |
| `--image_max_width` | int | resize images; `<=0` keeps original resolution |
| `--embedding_mode` | `none`, `zeros` | placeholder embedding behavior for smoke tests |
| `--allow_zero_embeddings` | flag | required with `--embedding_mode=zeros` |
| `--config_yaml` | path | reads `control.*` and `gripper.*` scales |
| `--action_space` | `eef` | ConRFT env action type |
| `--xyz_scale`, `--rot_scale` | positive float | override YAML action scales |
| `--gripper_max` | positive float | physical gripper value mapped to normalized `+1` |
| `--no_clip_action` | flag | do not clip normalized labels to `[-1, 1]` |
| `--max_episodes`, `--max_transitions` | positive int or omitted | limited export runs |

The generated actions use `euler_left` rotation semantics and canonical
`gym_sorted` state layout.

Add real Octo embeddings in the `RWRL` environment:

```bash
cd /home/ps/VLA-RL/conrft-r1lite
python examples/sarm/add_octo_embeddings_to_conrft_pkl.py \
  --exp_name=r1lite_dual_mango_box \
  --input_pkl=/home/ps/VLA-RL/conrft-r1lite/data/transition/r1lite_dual_mango_box/r1lite_dual_mango_box_sarm_reward_no_octo.pkl \
  --output_pkl=/home/ps/VLA-RL/conrft-r1lite/data/transition/r1lite_dual_mango_box/r1lite_dual_mango_box_sarm_reward.pkl
```

Embedding script parameters:

| Argument | Type / choices | Meaning |
| --- | --- | --- |
| `--exp_name` | experiment name | selects image keys and Octo config |
| `--input_pkl` | path | PKL without real embeddings |
| `--output_pkl` | path | final training PKL |
| `--image_keys` | comma-separated keys or omitted | defaults to experiment config |
| `--jax_platform` | `auto`, `cpu`, `gpu` | JAX backend selection |
| `--xla_mem_fraction` | float in `(0, 1]` | default JAX GPU memory fraction |
| `--start_trajectory` | int >= 0 | first trajectory to process |
| `--max_trajectories` | positive int or omitted | cap trajectories |
| `--max_transitions_per_trajectory` | positive int or omitted | cap transitions per trajectory |

### B4. Optional Replay Validation

Offline action replay is the recommended check before training:

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

For a valid PKL, action-integrated targets should match recorded next TCP poses
up to numerical precision.

### B5. Offline Training

Run learner-only pretraining:

```bash
cd /home/ps/VLA-RL/conrft-r1lite/examples/experiments/r1lite_dual_mango_box
export DEMO_PATH=/home/ps/VLA-RL/conrft-r1lite/data/transition/r1lite_dual_mango_box/r1lite_dual_mango_box_sarm_reward.pkl
export CHECKPOINT_PATH=/home/ps/VLA-RL/conrft-r1lite/examples/experiments/r1lite_dual_mango_box/conrft_sarm
bash run_learner_conrft_pretrain.sh
```

The same offline parameters from Path A apply. Defaults come from
`offline_training` in
[config.yaml](../examples/experiments/r1lite_dual_mango_box/config.yaml).

### B6. Online Training With SARM Reward

Start the SARM progress sidecar in the `lerobot` environment:

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

Sidecar parameters:

| Argument | Type / choices | Meaning |
| --- | --- | --- |
| `--reward_model_path` | path | trained SARM `pretrained_model` directory |
| `--host` | host | bind host |
| `--port` | int in `1..65535` | HTTP port |
| `--device` | torch device string | `cuda`, `cpu`, `cuda:0`, etc. |
| `--default_head_mode` | `sparse`, `dense` | head used when request omits it |
| `--log_level` | logging level | `INFO`, `DEBUG`, etc. |

Check the experiment reward model config:

```yaml
reward_model:
  enabled: true
  log_only: true
  endpoint_url: "http://127.0.0.1:8010"
  head_mode: "dense"
```

Use `log_only: true` first to record SARM diagnostics without replacing the env
reward. Set `log_only: false` when you intentionally want online RL to train on
SARM reward.

Start learner and actor in separate `RWRL` terminals.

Learner:

```bash
cd /home/ps/VLA-RL/conrft-r1lite/examples/experiments/r1lite_dual_mango_box
export DEMO_PATH=/home/ps/VLA-RL/conrft-r1lite/data/transition/r1lite_dual_mango_box/r1lite_dual_mango_box_sarm_reward.pkl
export CHECKPOINT_PATH=/home/ps/VLA-RL/conrft-r1lite/examples/experiments/r1lite_dual_mango_box/conrft_sarm
bash run_learner_conrft.sh
```

Actor:

```bash
cd /home/ps/VLA-RL/conrft-r1lite/examples/experiments/r1lite_dual_mango_box
export CHECKPOINT_PATH=/home/ps/VLA-RL/conrft-r1lite/examples/experiments/r1lite_dual_mango_box/conrft_sarm
bash run_actor_conrft.sh --ip=localhost
```

Online parameters are the same as Path A, with defaults from the mango
`online_training` config section.

