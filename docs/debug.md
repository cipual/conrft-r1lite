# R1Lite Debug Replay

This document is for real-robot replay diagnostics only. It is intentionally separate from the RL env, experiment configs, and training walkthrough.

Debug replay uses physical actions:

- EEF action: `actions["eef_delta"]` is `[dx, dy, dz, rx, ry, rz]` where translation is meters and rotation is a rotvec in radians.
- Joint action: `actions["joint_delta"]` is six joint deltas in radians.

The debug scripts only consume physical debug transitions and send commands through the body service.

## Setup

```bash
cd /home/robot/VLA-RL/conrft-r1lite/examples/debug/consistance
source /home/robot/Applications/miniforge3/etc/profile.d/conda.sh
conda activate RWRL
```

Make sure the body service is reachable:

```bash
curl -s http://192.168.12.12:8001/health
```

For joint replay, the body service must forward `joint_target` in `ee_pose_servo` mode to `/motion_target/target_joint_state_arm_right`.

## EEF Debug Transition

Convert the official RAW bag into a physical EEF-delta debug transition:

```bash
python convert_official_teleop_to_eef_debug_transition_r1lite.py \
  --input_dir=/home/robot/VLA-RL/conrft-r1lite/20260409/RB251106041_20260409152555451_RAW \
  --arm=right \
  --control_hz=10 \
  --output_file=/home/robot/VLA-RL/conrft-r1lite/examples/debug/consistance/r1lite_reach_target_official_teleop_eef_debug_transition.pkl
```

This file is not the RL training pkl. It is a debug pkl whose action is already a physical EEF delta.

## EEF Action Vs Pose Target Replay

Run the real-robot EEF comparison:

```bash
python debug_compare_eef_replay_r1lite.py \
  --input_file=/home/robot/VLA-RL/conrft-r1lite/examples/debug/consistance/r1lite_reach_target_official_teleop_eef_debug_transition.pkl \
  --trajectory_index=0 \
  --arm=right \
  --server_url=http://192.168.12.12:8001/ \
  --control_hz=10 \
  --log_every=10
```

What it does:

- `eef_action replay`: reads current real `tcp_pose`, adds `actions["eef_delta"]`, and sends the resulting `pose_target`.
- `pose_target replay`: sends recorded `next_observations["tcp_pose"]` directly.
- Both modes use body service HTTP only.

Default outputs:

- `/home/robot/VLA-RL/conrft-r1lite/examples/debug/consistance/r1lite_debug_eef_replay_compare_3d.png`
- `/home/robot/VLA-RL/conrft-r1lite/examples/debug/consistance/r1lite_debug_eef_replay_compare_errors.csv`
- `/home/robot/VLA-RL/conrft-r1lite/examples/debug/consistance/r1lite_debug_eef_replay_compare.npz`

Important CSV columns:

- `eef_action_pos_err_m`
- `eef_action_ori_err_rad`
- `pose_target_pos_err_m`
- `pose_target_ori_err_rad`
- `eef_delta_x_m` to `eef_delta_rz_rad`

## Joint Debug Transition

Convert the official RAW bag into a physical joint-delta debug transition:

```bash
python convert_official_teleop_to_joint_transition_r1lite.py \
  --input_dir=/home/robot/VLA-RL/conrft-r1lite/20260409/RB251106041_20260409152555451_RAW \
  --arm=right \
  --control_hz=10 \
  --output_file=/home/robot/VLA-RL/conrft-r1lite/examples/debug/consistance/r1lite_reach_target_official_teleop_joint_transition.pkl
```

This file is not the RL training pkl. It is a debug pkl whose action is already a physical joint delta.

## Joint Action Vs Pose Target Replay

Run the real-robot joint-space comparison:

```bash
python debug_compare_joint_replay_r1lite.py \
  --input_file=/home/robot/VLA-RL/conrft-r1lite/examples/debug/consistance/r1lite_reach_target_official_teleop_joint_transition.pkl \
  --trajectory_index=0 \
  --arm=right \
  --server_url=http://192.168.12.12:8001/ \
  --control_hz=10 \
  --mode=ee_pose_servo \
  --log_every=10
```

What it does:

- `joint_action replay`: reads current real `joint_pos`, adds `actions["joint_delta"]`, and sends the resulting `joint_target`.
- `pose_target replay`: sends recorded `next_observations["tcp_pose"]` directly.
- Both modes use body service HTTP only.

Default outputs:

- `/home/robot/VLA-RL/conrft-r1lite/examples/debug/consistance/r1lite_debug_joint_replay_compare_3d.png`
- `/home/robot/VLA-RL/conrft-r1lite/examples/debug/consistance/r1lite_debug_joint_replay_compare_errors.csv`
- `/home/robot/VLA-RL/conrft-r1lite/examples/debug/consistance/r1lite_debug_joint_replay_compare.npz`

Important CSV columns:

- `joint_action_pos_err_m`
- `joint_action_ori_err_rad`
- `pose_target_pos_err_m`
- `pose_target_ori_err_rad`
- `joint_delta_1` to `joint_delta_6`
- `target_joint_1` to `target_joint_6`
- `joint_action_joint_1` to `joint_action_joint_6`

## Flow Alignment

The EEF and joint checks are aligned:

| Flow | Action replay reference | Action command | Pose target replay |
| --- | --- | --- | --- |
| EEF | current real `tcp_pose` | physical EEF delta to `pose_target` | recorded `next tcp_pose` |
| Joint | current real `joint_pos` | physical joint delta to `joint_target` | recorded `next tcp_pose` |

This makes both action replays local delta rollouts from the current measured robot state, while both pose-target replays use recorded absolute EEF targets.

## Frequency Sweep

If joint replay is sensitive to command rate, run the same command with different `--control_hz` values:

```bash
python debug_compare_joint_replay_r1lite.py \
  --input_file=/home/robot/VLA-RL/conrft-r1lite/examples/debug/consistance/r1lite_reach_target_official_teleop_joint_transition.pkl \
  --trajectory_index=0 \
  --arm=right \
  --server_url=http://192.168.12.12:8001/ \
  --control_hz=1 \
  --mode=ee_pose_servo \
  --log_every=10 \
  --output_image_3d=/home/robot/VLA-RL/conrft-r1lite/examples/debug/consistance/r1lite_debug_joint_replay_compare_3d_hz1.png \
  --output_csv=/home/robot/VLA-RL/conrft-r1lite/examples/debug/consistance/r1lite_debug_joint_replay_compare_errors_hz1.csv \
  --output_npz=/home/robot/VLA-RL/conrft-r1lite/examples/debug/consistance/r1lite_debug_joint_replay_compare_hz1.npz
```

Compare `--control_hz=1`, `2`, `5`, and `10`.

## Sanity Checks

Check body service state:

```bash
curl -s http://192.168.12.12:8001/health
```

For joint replay, after sending a command, verify:

- `commands.right.desired_joint` is not `null`
- `commands.right.last_sent_target` contains `关节位置 [...]`

If needed, verify the ROS topic from a ROS 2 shell:

```bash
ros2 topic echo /motion_target/target_joint_state_arm_right
```

Seeing messages confirms the body service is publishing joint targets. If the robot does not move, inspect the downstream joint controller or official joint tracker chain.

## Interpretation

These scripts test real-world replay behavior, not training-label validity.

Offline consistency can be perfect while real replay still differs because real replay includes:

- controller delay
- tracking error
- hidden controller state
- command frequency limits
- topic feedback synchronization

Use the CSV and 3D plots to compare action replay against pose-target replay under the same body service execution path.
