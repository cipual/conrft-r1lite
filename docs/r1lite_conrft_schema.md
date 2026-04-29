# R1Lite ConRFT Training Schema

This note defines the default single-arm schema for connecting the current
`r1lite` body service to the existing ConRFT training stack.

## Observation schema

The raw HTTP env response is adapted into:

```python
{
    "state": {
        "tcp_pose": float32[7],
        "tcp_vel": float32[6],
        "tcp_force": float32[3],
        "tcp_torque": float32[3],
        "gripper_pose": float32[1],
        "joint_pos": float32[6],
        "joint_vel": float32[6],
        "joint_effort": float32[6],
    },
    "images": {
        "image_primary": uint8[256, 256, 3],  # mapped from head camera
        "image_wrist": uint8[256, 256, 3],    # mapped from selected arm wrist camera
    },
}
```

### Current Field Sources In `R1LiteArmEnv`

The single-arm env currently reads every observation field from the robot HTTP
`/state` response through
[envs.py](../serl_robot_infra/r1lite_env/envs.py).

Field-by-field source mapping:

| ConRFT field | Current source in `/state` | Notes |
|---|---|---|
| `state.tcp_pose` | `raw["state"][arm]["tcp_pose"]` | Real robot/server value |
| `state.tcp_vel` | `raw["state"][arm]["tcp_vel"]` | Real robot/server value |
| `state.tcp_force` | `raw["state"][arm]["tcp_force"]` | Present in schema, but only as reliable as the server-side source |
| `state.tcp_torque` | `raw["state"][arm]["tcp_torque"]` | Present in schema, but only as reliable as the server-side source |
| `state.gripper_pose` | `raw["state"][arm]["gripper_pose"]` | Real robot/server value |
| `state.joint_pos` | `raw["state"][arm]["joint_pos"]` | Real robot/server value |
| `state.joint_vel` | `raw["state"][arm]["joint_vel"]` | Real robot/server value |
| `state.joint_effort` | `raw["state"][arm]["joint_effort"]` | Present in schema, but only as reliable as the server-side source |
| `images.image_primary` | `raw["images"]["head"]` after env-side resize/remap | Mapped to `image_primary` by wrapper |
| `images.image_wrist` | `raw["images"]["left_wrist" / "right_wrist"]` after env-side resize/remap | Mapped to `image_wrist` by wrapper |

Important clarification:

- the current env schema exposes `tcp_force`, `tcp_torque`, and `joint_effort`
  because the training stack expects these keys to exist
- however, whether these values are **real measured signals** depends entirely
  on what the robot service is actually returning at `/state`
- they should not be assumed to be trustworthy force/torque/effort signals
  unless the upstream robot server path has been verified separately

For the official teleop bag we inspected, the available bag topics do **not**
currently provide usable TCP wrench samples:

- `/hdas/feedback_right_arm_wrench` exists but has `message_count = 0`
- `/hdas/feedback_left_arm_wrench` exists but has `message_count = 0`

So for official-teleop conversion, `tcp_force` / `tcp_torque` should currently
be treated as unavailable unless a different bag confirms non-empty wrench
topics or another state source is added.

For ConRFT, the canonical flattened proprio ABI is `gym_sorted`.
`SERLObsWrapper` sorts `proprio_keys` before flattening; conversion scripts must
write offline PKLs in the same order.

For single-arm tasks, the default canonical proprio subset is:

```python
["gripper_pose", "tcp_force", "tcp_pose", "tcp_torque", "tcp_vel"]
```

After `SERLObsWrapper` and `ChunkingWrapper(obs_horizon=2)`, training consumes:

```python
{
    "state": float32[2, 20],
    "image_primary": uint8[2, 256, 256, 3],
    "image_wrist": uint8[2, 128, 128, 3],
}
```

For the dual mango task, the canonical flattened order is:

```python
[
    "left/gripper_pose", "left/joint_pos", "left/joint_vel", "left/tcp_pose", "left/tcp_vel",
    "right/gripper_pose", "right/joint_pos", "right/joint_vel", "right/tcp_pose", "right/tcp_vel",
    "torso_pos",
]
```

Do not mix older config-order dual mango PKLs with the current online env.

## Action schema

The policy action stays as the standard single-arm normalized command:

```python
action: float32[7]
```

Semantics:

- `action[:3]`: xyz delta in normalized space
- `action[3:6]`: rpy delta in normalized space
- `action[6]`: gripper command in `[-1, 1]`

The env converts the action to the current R1Lite HTTP payload:

- `pose_target`: current TCP pose plus scaled delta, then clipped to the configured per-arm safety box before publish
- `gripper`: `clip((action[6] + 1) * 50, 0, 100)`
- `preset`: default `free_space`
- `owner`: `policy`

The maintained R1Lite env now follows the Franka env pattern for:

- fixed reset pose with optional randomization around the reset center
- per-arm Cartesian/orientation safety limits in env space
- policy-side HIL intervention through the env wrapper layer
- reset poses are forwarded through `/reset` and clipped again on the robot service side

For `r1lite_reach_target`, the default config keeps `RANDOM_RESET = False` for
bring-up, but exposes the same `RANDOM_XY_RANGE` / `RANDOM_RZ_RANGE` interface
as the official Franka tasks.

## Demo transition schema

ConRFT learner still expects transition `.pkl` files instead of raw HDF5:

```python
{
    "observations": obs_t,
    "actions": action_t,
    "next_observations": obs_t1,
    "rewards": reward_t,
    "masks": 1.0 - done_t,
    "dones": done_t,
    "mc_returns": return_to_go_t,
    "embeddings": octo_embedding_t,
    "next_embeddings": octo_embedding_t1,
}
```

The Octo embedding helper now uses the configured image keys instead of the
Franka-specific `side_policy_256` and `wrist_1`.

### Conversion Note

When converting official teleop bags into RL transitions, do **not** resize
images in the first conversion stage unless you are explicitly targeting a
specific model input format.

Recommended split:

1. official teleop bag -> task-aligned RL transition data with original images
2. RL transition data -> model-specific resized tensors (for example Octo)

This keeps the conversion output model-agnostic and avoids throwing away image
information too early.

## First task: reach target pose

The first recommended R1Lite task is `r1lite_reach_target`.

- Control mode: single-arm, 7D normalized action
- Reward: dense pose error penalty plus sparse success reward
- Success: end-effector position and orientation both within thresholds
- Demo collection: SpaceMouse intervention through the existing R1Lite teleop wrapper

Default success thresholds:

- position error <= `0.03 m`
- orientation error <= `0.35 rad`

Default target pose:

- right arm: `[0.43, 0.20, 0.28, 0.0, 1.0, 0.0, 0.0]`
- left arm: `[0.43, -0.20, 0.28, 0.0, 1.0, 0.0, 0.0]`

## Recommended R1Lite workflow

For the maintained end-to-end flow, use:

- walkthrough: [r1lite_walkthrough.md](./r1lite_walkthrough.md)
- demo recording entry point: [record_demos_r1lite_octo.py](../examples/record_demos_r1lite_octo.py)
- experiment launchers:
  [run_record_demos_octo.sh](../examples/experiments/r1lite_reach_target/run_record_demos_octo.sh),
  [run_learner_conrft_pretrain.sh](../examples/experiments/r1lite_reach_target/run_learner_conrft_pretrain.sh),
  [run_learner_conrft.sh](../examples/experiments/r1lite_reach_target/run_learner_conrft.sh),
  [run_actor_conrft.sh](../examples/experiments/r1lite_reach_target/run_actor_conrft.sh)
