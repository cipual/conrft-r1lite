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

For ConRFT, the default proprio subset is:

```python
["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose"]
```

After `SERLObsWrapper` and `ChunkingWrapper(obs_horizon=2)`, training consumes:

```python
{
    "state": float32[2, 20],
    "image_primary": uint8[2, 256, 256, 3],
    "image_wrist": uint8[2, 256, 256, 3],
}
```

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

- `pose_target`: current TCP pose plus scaled delta
- `gripper`: `clip((action[6] + 1) * 50, 0, 100)`
- `preset`: default `free_space`
- `owner`: `policy`

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
