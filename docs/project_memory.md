# Project Memory

This document records the working context, decisions, and known pitfalls accumulated while integrating R1Lite, ConRFT, debug replay, LeRobot, and SARM reward modeling.

## Repositories

- Main RL / robot task repo: `/home/robot/VLA-RL/conrft-r1lite`
- Robot body service repo: `/home/robot/VLA-RL/r1lite`
- LeRobot sidecar repo: `/home/robot/VLA-RL/lerobot`
- Main ConRFT remote: `github.com:cipual/conrft-r1lite.git`
- Local LeRobot fork remote: `https://github.com/cipual/lerobot.git`

The LeRobot fork is intentionally maintained because upstream LeRobot needed local SARM fixes for this project.

## Environments

Use two separate conda environments.

`RWRL` is for ConRFT, Octo embeddings, robot envs, replay, conversion, and training.

`lerobot` is for LeRobotDataset export, SARM annotation, SARM reward model training, and SARM progress computation.

Do not mix the two environments. Several bugs came from dependency differences around PyTorch, Transformers, CLIP, video encoding, and JAX/TensorFlow packages.

Quick setup is documented in `docs/environment_setup.md`.

## Robot And Body Service

The debug and replay tools should use the R1Lite body service HTTP API, not raw ROS2 subscriptions from the debug scripts.

Important body service endpoint:

```text
http://192.168.12.12:8001/
```

Useful checks:

```bash
curl -s http://192.168.12.12:8001/health
curl -s http://192.168.12.12:8001/state
```

The body service supports `/action` ownership rules:

- `policy` can send `/action`.
- `debug` can send `/reset`, `/recover`, `/clear_fault`, and `/brake`.
- `teleop` can send `/action` when `teleop_source=spacemouse`.

For joint target under `ee_pose_servo`, the old server behavior showed that explicit velocity values such as `velocities=[0.5] * 6` are needed for joint control to visibly execute.

## R1Lite Reach Target Replay Findings

The official teleop converted data showed a large accumulated error when replayed in EEF `action` mode.

Key observation:

- `replay_mode=pose_target` succeeds because each step sends the recorded target pose directly.
- `replay_mode=action` initially behaved like chunk size 1: read current robot state, add one action delta, command the next pose, then repeat.
- This makes the rollout sensitive to small execution/state-estimation differences because every next target is anchored to the current measured state.
- `pose_target` behaves like a very large action chunk because it follows the recorded pose sequence instead of reintegrating from noisy measured state.

Important interpretation:

- The action itself can be consistent with adjacent recorded observations offline.
- Real robot execution can still drift when every step reanchors to the live measured TCP pose.
- Chunked action replay reduces this sensitivity by accumulating a short sequence from a reference pose rather than from every newly read live state.

The previous deadzone attempt did not solve the problem and was reverted.

## Debug Replay Design

Debug replay must be fully decoupled from RL envs.

Debug scripts should not import or depend on:

- `gym`
- `R1LiteArmEnv`
- `CONFIG_MAPPING`
- experiment YAML
- `xyz_scale`
- `rot_scale`
- RL normalized action scale
- `replay_transition_r1lite.py`

Debug transition actions are already physical quantities.

EEF debug action:

- Key: `actions["eef_delta"]`
- Translation unit: meters
- Rotation unit: radians
- Rotation representation should be documented in `infos["action_type"]`

Joint debug action:

- Key: `actions["joint_delta"]`
- Unit: radians

Debug PKL is not the RL training PKL. A debug PKL is only for consistency checks and replay diagnostics.

Current debug folder convention:

```text
conrft-r1lite/examples/debug/<task_name>/consistance/
```

The spelling `consistance` is currently used in the project paths.

## Debug Replay Commands

EEF debug conversion should produce a debug transition from official RAW data without action scale.

EEF debug replay compare should compare:

- `action replay`: current real TCP pose plus physical EEF delta.
- `pose_target replay`: recorded `next_observations["tcp_pose"]`.

Joint debug conversion should produce physical joint deltas.

Joint debug replay compare should compare:

- `joint action replay`: current real joint pose plus physical joint delta.
- `pose_target replay`: recorded target, depending on the script mode.

Debug compare outputs should include:

- 3D trajectory comparison figure.
- Per-step error CSV.
- Raw replay arrays in NPZ.

## Joint Vs EEF Replay Findings

Joint-space action replay and pose-target replay looked much more consistent than EEF action replay.

This suggests the main issue is not simply "robot did not wait long enough" or "state was read too early". Running slower, such as `--control_hz=1`, can improve some joint tests, but it does not fully explain the EEF accumulated error.

Likely contributing factors:

- EEF delta composition is more sensitive to pose representation and controller behavior.
- TCP pose state estimation has more visible bias/noise than joint state.
- Reanchoring every EEF action to live TCP pose compounds small errors.
- Joint targets may be tracked more directly by the low-level controller.

## Chunked Replay Decision

For action replay diagnostics, action mode should support chunk size.

Important semantic correction:

- A chunk of size `K` does not mean "execute only the final pose once".
- It means execute `K` commands.
- Each of the `K` targets is computed by integrating actions from the same chunk reference pose, not by repeatedly reading live robot state after every action.

`pose_target` mode should not consume the chunk-size parameter.

## Data Layout

The project is moving toward keeping collected and converted datasets under:

```text
conrft-r1lite/data/
```

Examples:

```text
conrft-r1lite/data/raw/<task_name>/
conrft-r1lite/data/lerobot/<task_name>/
conrft-r1lite/data/transition/<task_name>/
```

Debug PKLs remain under:

```text
conrft-r1lite/examples/debug/<task_name>/consistance/
```

because they are not training data.

## R1Lite Dual Mango Box Task

Task description:

```text
左臂抓住白色的框放在右臂的周围，右臂抓住发红的芒果，把它放入框内，然后左右机械臂复位。
```

Reset is considered part of the task, not a segment to crop away.

The SARM progress endpoint should be "both robot arms return to the reset pose", not "mango enters the box".

Assumption:

- Reset means the robot arms/grippers return to a safe or initial pose.
- The mango and white box do not need to be restored to their original physical positions.

## SARM Pipeline

The SARM data chain should start from RAW rosbag, not from ConRFT transition PKL.

Primary flow:

```text
RAW rosbag -> LeRobotDataset -> SARM annotation -> SARM model -> sarm_progress.parquet -> ConRFT PKL with SARM reward
```

For reward-model-dependent long-horizon tasks, generating a ConRFT PKL directly from `LeRobotDataset + sarm_progress.parquet` is preferred.

Reason:

- A generic RAW rosbag to ConRFT PKL conversion needs a reward definition.
- For simple tasks such as reach target, sparse 0/1 reward is easy.
- For long-horizon manipulation, reward is produced by the reward model, so the reward model output should define the transition reward from the start.

## Canonical State Layout

ConRFT transition state layout is now intentionally a single ABI:

```text
gym_sorted
```

Meaning:

- `SERLObsWrapper` sorts `proprio_keys` before creating the flattened state space.
- online env observations, offline conversion scripts, replay diagnostics, and training PKLs must all use the same sorted key order.
- replay tools should not guess between multiple possible state layouts. If a PKL does not match `gym_sorted`, regenerate it.

Single-arm canonical order:

```text
gripper_pose, tcp_force, tcp_pose, tcp_torque, tcp_vel
```

Dual mango canonical order:

```text
left/gripper_pose, left/joint_pos, left/joint_vel, left/tcp_pose, left/tcp_vel,
right/gripper_pose, right/joint_pos, right/joint_vel, right/tcp_pose, right/tcp_vel,
torso_pos
```

Important consequence:

- Older `r1lite_dual_mango_box` PKLs generated before this decision may be in config-order layout. Do not train or online-finetune from those files.
- Regenerate the SARM reward PKL from LeRobot/SARM progress with the latest `examples/sarm/relabel_rosbag_or_conrft_with_sarm_reward.py`.
- Then run `examples/sarm/add_octo_embeddings_to_conrft_pkl.py` to add real Octo embeddings.
- Use `examples/replay_transition_r1lite.py --exec_mode=offline --replay_mode=action` before training; a canonical PKL should reproduce recorded next TCP pose with near-zero error.

## LeRobot Export

The export script supports converting a folder containing many `_RAW` episode directories.

LeRobot export should include:

- `observation.images.head`
- `observation.images.left_wrist`
- `observation.images.right_wrist`
- `observation.state`
- `action`
- `timestamp`
- `episode_index`
- `frame_index`
- `task`

Action is physical dual-arm delta by default.

Action spaces:

- `eef`: EEF delta plus gripper delta.
- `joint`: joint delta plus gripper delta.

Known export details:

- Wrist cameras may be `360x640`, while head may be `720x1280`.
- Dataset features must match actual image shapes.
- LeRobot may concatenate episode videos into a long video file.
- Temporary raw image frames can disappear after successful video encoding because LeRobot cleans intermediate frame folders.
- PyAV can fail if FPS is passed as a float; use an integer or rational-compatible FPS.

## Manual Annotation

Manual SARM annotation should load LeRobotDataset, not RAW data.

Reason:

- VLM annotation uses LeRobot format.
- Manual annotation should use the same dataset representation to avoid mismatched labels.

Manual annotation UI should:

- Load existing annotations if present.
- Load unlabeled trajectories if no annotation exists.
- Support episode selection.
- Jump to selected episode start/end correctly, even when videos are concatenated globally.
- After setting one subtask end, auto-fill the next subtask start.

Annotation JSON is stored in the LeRobot dataset metadata area, for example:

```text
data/lerobot/r1lite_dual_mango_box/meta/sarm_manual_annotations.json
```

## Sparse, Dense, And Dual SARM Labels

Sparse and dense labels are different granularities of subtask annotation.

Sparse labels:

- Coarser task phases.
- Useful for stage-level progress.

Dense labels:

- Finer-grained progress states.
- Useful for smoother reward signal.

Dual mode can train two heads with different granularity, but it is not mandatory.

For the current manually labeled dense-only experiment, use:

```text
policy.annotation_mode=dense_only
head-mode dense
```

`head-mode sparse` means use the sparse SARM head when computing progress/RABC weights. If training was dense-only, use dense.

## SARM Training Notes

The local LeRobot fork includes SARM-related fixes:

- `processor_sarm.py` handles Transformers versions where CLIP returns `BaseModelOutputWithPooling` or similar ModelOutput objects instead of plain tensors.
- `compute_rabc_weights.py` uses a non-interactive Matplotlib backend to avoid Tk crashes.
- `compute_rabc_weights.py` defaults to no upload and requires explicit `--push-to-hub`.

Training was slow with larger batch sizes because image/video decoding and CLIP preprocessing became bottlenecks. Smaller batch size, such as 2, was faster in observed wall-clock step time.

If reducing batch size, total optimizer steps may need to increase if the goal is to keep the same number of seen samples.

## SARM Progress And Reward

The current reward construction is based on progress delta:

```text
reward_t = progress_{t+1} - progress_t
```

At success:

- If progress crosses `success_threshold`, terminal success is triggered.
- A `success_reward`, for example `10.0`, can be added.

This means `critic/rewards` in training can look noisy because it is logging sampled batch rewards from a replay buffer. Even if the reward function is deterministic, minibatch sampling across many timesteps and episodes produces a spiky reward plot.

Reward scaling and clipping are controlled by experiment YAML / training config, not by the raw SARM progress file itself.

The relabel script was changed toward reading experiment YAML because action/state/image conventions and reward scaling should stay aligned with the task config.

## ConRFT PKL From SARM

For long-horizon SARM tasks, the intended path is:

```text
LeRobotDataset + sarm_progress.parquet -> ConRFT transition PKL
```

This avoids requiring a hand-defined sparse reward before the reward model exists.

The generated PKL must be compatible with existing ConRFT training loaders.

Important compatibility points:

- Image observations must be present and valid if training uses images.
- Octo embeddings must be present or intentionally zero-filled only if the downstream trainer accepts that.
- Action dimensionality must match the experiment config.
- State/proprio dimensionality must match the experiment config.
- State/proprio layout must be canonical `gym_sorted`.
- R1Lite ConRFT EEF rotation action must use the repo-default `euler_left`
  convention: env applies `Rotation.from_euler("xyz", delta) * current_rotation`.
- Rewards, dones, masks, and mc_returns must be finite.
- The direct LeRobot-to-ConRFT exporter reorders LeRobot dual-arm state into canonical `gym_sorted` before writing the PKL.
- The direct LeRobot-to-ConRFT exporter reconstructs EEF translation and
  `euler_left` rotation actions from current/next recorded TCP poses, so the
  action labels stay coupled to the state trajectory.
- The gripper action is reconstructed from the canonical next-state gripper field, so state order and action semantics stay coupled.
- PKLs generated under the old SARM `rotvec_right` action convention must be
  regenerated before replay, offline training, or online initialization.

There was a previous issue where a generated SARM PKL was interpreted as thousands of tiny trajectories with two frames each. This indicates the PKL trajectory structure must be validated carefully before training.

Use the PKL camera visualization script to sanity-check trajectory grouping and images.

Required validation after export:

- preview camera frames with `examples/visualize_pkl_cameras.py`
- verify `infos["state_layout"] == "gym_sorted"`
- verify rewards, dones, mc_returns, and success flags
- verify image keys match experiment config
- run `replay_transition_r1lite.py` in offline `action` mode and confirm the target-to-recorded TCP error is near zero
- run a short offline learner load test before a long training run

## Octo Embedding Notes

`examples/sarm/add_octo_embeddings_to_conrft_pkl.py` should be run in `RWRL`.

The SARM mango visual PKL can be very large; current sample output is around 18 GB before embeddings. A native `段错误 (核心已转储)` during the first Octo trajectory is usually not a Python exception. In the observed RWRL environment, the direct cause was TensorFlow and JAX both touching CUDA before TensorFlow GPU visibility was disabled.

Current fix:

- the script sets conservative defaults before importing JAX/Octo:
  - `XLA_PYTHON_CLIENT_PREALLOCATE=false`
  - `XLA_PYTHON_CLIENT_MEM_FRACTION=0.35`
  - `TF_FORCE_GPU_ALLOW_GROWTH=true`
- for GPU runs, the script imports TensorFlow first and runs `tf.config.set_visible_devices([], "GPU")`, matching Octo's own training/finetuning scripts; JAX then owns CUDA exclusively.
- `--jax_platform=cpu` also sets `JAX_PLATFORMS=cpu`, which is required by the current JAX version to skip CUDA backend initialization.
- Octo embeddings are immediately copied from device arrays to CPU `float32` numpy arrays to avoid retaining one GPU buffer per transition.
- GPU smoke tests passed for the first 2 transitions, 50 transitions, and the full first trajectory of 595 transitions after this fix.
- if GPU still segfaults on another machine, first run the partial diagnostic commands below, then fall back to `--jax_platform=cpu` only as a last resort.

Recommended diagnostic command:

```bash
python examples/sarm/add_octo_embeddings_to_conrft_pkl.py \
  --exp_name=r1lite_dual_mango_box \
  --input_pkl=/home/ps/VLA-RL/conrft-r1lite/data/transition/r1lite_dual_mango_box/r1lite_dual_mango_box_sarm_reward_no_octo.pkl \
  --output_pkl=/tmp/r1lite_dual_mango_box_sarm_reward_first_traj.pkl \
  --max_trajectories=1 \
  --jax_platform=cpu
```

Do not use a partial output file for training. It is only for validating that Octo embedding generation works.

## PKL Camera Visualization

A utility exists to load a ConRFT PKL and export camera videos.

This is useful for checking:

- Whether images are stored correctly.
- Whether trajectory length is correct.
- Whether the loaded PKL is actually one episode or accidentally split into many tiny episodes.

If the output video is only a few KB and contains two frames, the PKL trajectory structure is probably wrong or the script loaded a tiny trajectory.

## Training Observations

Earlier SARM-reward RL training showed unstable losses:

- Large `actor/bc_loss` and `actor/actor_loss` spikes were linked to action scaling / action dimensionality / data mismatch issues.
- After fixing the relabel / config alignment, BC loss decreased normally.
- `actor/q_loss` can remain high or plateau if critic target values are noisy, if reward deltas are sparse/spiky, or if Q scale is poorly matched to reward scale.
- `critic/td_loss` can look low while Q-related actor terms remain unhelpful, because the critic can fit bootstrapped targets without giving a clean policy improvement signal.

Important diagnostic plots:

- `critic/rewards`
- `critic/target_qs`
- `critic/predicted_qs`
- `critic/random_action_values`
- `critic/next_action_values`
- `actor/bc_loss`
- `actor/q_loss`

If `random_action_values` overlap too much with policy action Q values, the critic may not be learning a useful action preference.

## Error Recovery Data

The current framework can store trajectories where time moving forward does not always mean task progress increases.

SARM dense progress can represent regressions if labels/model allow it.

However, there is no confirmed full rewind augmentation pipeline currently integrated for teaching "bad recovery" explicitly.

Potential future direction:

- Add backward / rewind augmented transitions.
- Preserve negative progress deltas instead of clipping them away.
- Train reward/Q functions to distinguish recovery from forward task progress.

## Git And Remote State

Recent pushed commits:

ConRFT repo:

```text
f14f74d Simplify environment setup quick start
```

LeRobot fork:

```text
1715022f Patch SARM for local R1Lite workflow
```

The LeRobot local branch now tracks `origin/main` from `cipual/lerobot`.

When rebasing LeRobot against the fork, use:

```bash
GIT_LFS_SKIP_SMUDGE=1 git rebase origin/main
```

This avoids LFS smudge failures from large upstream test artifacts.

## Known Pitfalls

- Do not use RL training PKL as debug PKL.
- Do not apply `xyz_scale` or `rot_scale` inside debug replay.
- Do not debug replay through `R1LiteArmEnv`.
- Do not assume `pose_target` and `action` replay diagnose the same failure mode.
- Do not assume slower control Hz alone fixes EEF drift.
- Do not mix `RWRL` and `lerobot` conda environments.
- Do not run LeRobot SARM scripts from the ConRFT repo path unless imports are configured correctly.
- Use `python -m lerobot...` for LeRobot module scripts when relative imports are involved.
- For dense-only SARM training, use dense head in progress computation.
- Do not push datasets, RAW bags, videos, checkpoints, or generated transition PKLs to Git.

## Useful Project Documents

- `docs/environment_setup.md`: Quick Start environment setup.
- `docs/debug.md`: Debug replay and consistency check workflow.
- `docs/r1lite_walkthrough.md`: R1Lite task walkthrough and training/replay commands.
- `docs/r1lite_conrft_schema.md`: ConRFT PKL schema notes.
- `examples/sarm/README.md`: SARM export, annotation, training, relabel, and reward-model notes.
