# Official Teleop To ConRFT Conversion

This note describes how to convert Galaxea/official teleoperation recordings
(`RAW.json + metadata.yaml + .mcap`) into the RL transition format expected by
the maintained ConRFT pipeline.

## Goal

The conversion should be driven by the target experiment config instead of
being hard-coded for one task. In particular, the converter should read the
task's `config.yaml` / `config.py` to decide:

- control frequency (`control.hz`)
- single-arm vs dual-arm mode
- selected arm (`left` / `right`)
- whether the task uses the gripper
- image aliases (`image_keys`)
- reward / success logic if task-specific reward conversion is enabled

## Input Format

Official teleop data is expected in the standard layout:

- `<episode>_RAW.json`
- `<episode>_RAW/metadata.yaml`
- `<episode>_RAW/<episode>_RAW.mcap`

For the inspected sample, the bag contains at least:

- right arm feedback:
  - `/hdas/feedback_arm_right`
  - `/hdas/feedback_gripper_right`
- right arm commanded targets:
  - `/motion_target/target_joint_state_arm_right`
  - `/motion_target/target_position_gripper_right`
- right arm end-effector pose:
  - `/motion_control/pose_ee_arm_right`
- camera streams:
  - `/hdas/camera_head/right_raw/image_raw_color/compressed`
  - `/hdas/camera_head/left_raw/image_raw_color/compressed`
  - `/hdas/camera_wrist_right/color/image_raw/compressed`

Important observed property of the sample bag:

- `/motion_target/target_pose_arm_right` has `message_count = 0`
- `/motion_target/target_joint_state_arm_right` is populated

So the official teleop sample is effectively joint-target driven, not
pose-target driven.

## Topic -> RL Field Mapping

For the current `r1lite_reach_target` single-arm right-arm task:

| Source topic | Meaning | RL transition field |
|---|---|---|
| `/hdas/camera_head/right_raw/image_raw_color/compressed` or `/hdas/camera_head/left_raw/image_raw_color/compressed` | head image | `observations["image_primary"]` |
| `/hdas/camera_wrist_right/color/image_raw/compressed` | right wrist image | `observations["image_wrist"]` |
| `/motion_control/pose_ee_arm_right` | current EE pose | `observations["state"]["tcp_pose"]` |
| `/hdas/feedback_arm_right` | current right-arm joint feedback | `joint_pos` and, by differencing, `joint_vel` |
| `/hdas/feedback_gripper_right` | current gripper feedback | `gripper_pose` |
| `/motion_target/target_joint_state_arm_right` | teleop command target | action reconstruction source |
| `/motion_target/target_position_gripper_right` | gripper target | optional action reconstruction source |

Currently unavailable or not trustworthy in the inspected sample:

- `tcp_force`
- `tcp_torque`
- reliable `joint_effort`

These should currently be filled conservatively and annotated as unavailable
unless a verified source is added.

## Recommended Conversion Stages

### Stage 1: Bag -> Task-Aligned RL Episode

This stage should:

- load the task config (`exp_name` or config path)
- inspect the bag metadata and validate required topics
- choose the active arm / camera topics from config
- build a unified time axis using the task's target control frequency
- align state, target, and image messages onto that time axis
- reconstruct `observations`, `actions`, `next_observations`, `done`, `mask`
- optionally compute rewards using the task wrapper logic
- add `mc_returns` using the same post-processing helper as online demo recording

This stage should keep original images unresized whenever possible.

### Stage 2: RL Episode -> Model-Specific Training Input

This stage is model-dependent and may:

- resize images for Octo or another VLA
- add `mc_returns`
- add `embeddings`
- add `next_embeddings`

Keeping Stage 1 model-agnostic avoids throwing away image information too early.

## Time Axis And Frequency

The converter should not assume the bag frequency equals the RL training
frequency. Instead:

1. read original timestamps from the bag
2. use `control.hz` from the target task config as the desired RL step rate
3. resample to that rate when building transitions

For `r1lite_reach_target`, this currently means `10 Hz`.

## Action Reconstruction Strategy

Because the inspected official teleop sample is driven by joint targets instead
of pose targets, the most practical action reconstruction for the current R1Lite
ConRFT setup is:

1. use consecutive end-effector poses from `/motion_control/pose_ee_arm_right`
2. compute pose delta between time `t` and `t+1`
3. convert that delta to the RL action convention:
   - `dx, dy, dz`
   - `droll, dpitch, dyaw`
4. append a gripper action if the target task uses gripper control

For fixed-gripper tasks like `r1lite_reach_target`, the final gripper dimension
may be set to a constant neutral/open-compatible value and excluded from
task-specific semantics.

## Reward / Done Alignment With Online Demo Recording

The maintained online demo script is:

- [record_demos_r1lite_octo.py](../examples/record_demos_r1lite_octo.py)

Its reward and done behavior comes from the real env wrapper stack, especially:

- [wrapper.py](../examples/experiments/r1lite_reach_target/wrapper.py)

For `r1lite_reach_target`, the offline official-teleop conversion now follows
the same task semantics:

- reward is computed from the next-step end-effector pose
- the same position/orientation errors are used
- the same success threshold is used
- the same dense penalty + success bonus structure is used
- `done=True` on success, or on the final converted step
- `mc_returns` are computed with the same helper used by online demo recording

This keeps the converted official-teleop demos aligned with the demos produced
by `run_record_demos_octo.sh`, except that the official-teleop path starts from
recorded bag data rather than a live env rollout.

## Script Structure

Recommended converter structure:

1. config loader
   - reads `exp_name` or `config.yaml`
   - resolves arm mode, image keys, target `hz`, gripper usage

2. bag reader
   - loads `metadata.yaml`
   - validates presence of required topics
   - exposes timestamped reads for images / state / targets

3. transition builder
   - generates the resampled timeline
   - constructs observations
   - reconstructs actions
   - computes rewards / done when requested
   - adds `mc_returns`
   - writes task-aligned `.pkl`

## Current Script

The maintained conversion entry point is:

- [convert_official_teleop_to_conrft_demo.py](../examples/convert_official_teleop_to_conrft_demo.py)

The current version is a `rosbags`-backed converter with an optional stage-2
embedding pass:

- config-driven
- metadata-aware
- topic-validation ready
- reads selected ROS bag topics through `rosbags`
- decodes compressed head / wrist images
- reconstructs task-aligned observations and 7D actions
- computes task reward / success and `mc_returns`
- optionally adds Octo `embeddings` and `next_embeddings` with `--with_embeddings`

Current scope and limits:

- tested design target is single-arm `r1lite_reach_target`
- keeps original image resolution in the output transitions
- stage-2 Octo processing is optional and enabled explicitly

## Conversion Environment

The conversion environment should be kept separate from the main training
environment when possible. The goal of this environment is:

- inspect `metadata.yaml`
- read `.mcap` / rosbag2 payloads
- decode compressed images
- build task-aligned RL transitions

For this workflow, a full ROS2 runtime is not required by default. A lighter
Python-only stack is preferred first.

### Recommended Minimal Dependencies

- `python>=3.10`
- `numpy`
- `pyyaml`
- `opencv-python`
- `pillow`
- `tqdm`
- `rosbags`

These are enough for:

- metadata parsing
- bag reading
- compressed image decoding
- timestamp alignment
- transition serialization

### Optional Dependencies

- `mcap`
  Useful if we decide to read MCAP directly at a lower level instead of relying
  on `rosbags`.
- `rclpy`
- `rosbag2_py`
- `cv_bridge`

These ROS2 packages are only recommended if the Python-only reader path proves
insufficient for the official teleop bags or custom message decoding.

### Recommended Conda Environment

```bash
conda create -n r1lite-convert python=3.10 -y
conda activate r1lite-convert
python -m pip install --upgrade pip
python -m pip install \
    numpy \
    pyyaml \
    opencv-python \
    pillow \
    tqdm \
    rosbags
```

### Optional Add-Ons

If later needed, install these on top of the minimal environment:

```bash
python -m pip install mcap
```

If a ROS2-backed reader is eventually required, install ROS2 separately and
then add the matching Python packages from that ROS2 distribution instead of
mixing arbitrary wheels into the conda environment.

### Why This Is Separate From `RWRL`

The maintained `RWRL` environment is already responsible for:

- ConRFT training
- Octo / JAX
- online actor / learner execution

Mixing ROS2 bag-reading dependencies directly into that environment increases
the risk of package conflicts. The conversion environment should therefore stay
small and focused, and produce an intermediate task-aligned `.pkl` that the
training environment can consume later.
