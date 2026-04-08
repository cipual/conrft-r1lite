# Training On R1Lite Walkthrough

This walkthrough mirrors the official Franka pipeline, but uses the current
R1Lite HTTP service, single-arm reach-target scaffold, and SpaceMouse
intervention tooling in `conrft-r1lite`.

## Scope

The current end-to-end path is built around
`examples/experiments/r1lite_reach_target`.

- task: single-arm reach target
- reward: geometric reward from end-effector pose error
- teleoperation: SpaceMouse through `R1LiteTeleopInterventionWrapper`
- control owner during HIL: `policy`
- direct teleop owner: `teleop` with `teleop_source="spacemouse"`

## 0. Activate The Environment

Use the `RWRL` conda environment before running any of the scripts below:

```bash
source /home/robot/Applications/miniforge3/etc/profile.d/conda.sh
conda activate RWRL
```

The provided `run_*.sh` scripts export the required `PYTHONPATH` and a writable
`MPLCONFIGDIR` automatically.

Set the robot service address before demo collection or training:

```bash
export ROBOT=http://192.168.12.12:8001
```

## 1. Start The Robot Service

On the robot machine, start the R1Lite body service and verify that:

- `GET /state` returns valid arm state and images
- `GET /health` reports the expected `active_mode`
- `POST /action` works for `owner="policy"`

The inference-side quick checks are documented in [README.md](../README.md).

## 2. Configure The Experiment

Review [config.yaml](../examples/experiments/r1lite_reach_target/config.yaml) first.
The Python file [config.py](../examples/experiments/r1lite_reach_target/config.py)
now mainly contains loading logic and wrapper wiring.

Key values to confirm:

- `SERVER_URL`
- `DEFAULT_MODE`
- `DEFAULT_PRESET`
- `RESET_LEFT_POSE` / `RESET_RIGHT_POSE`
- `reset_left_joint` / `reset_right_joint`
- `env.reset_settle_sec`
- reset completion thresholds under `env.reset_*`
- `RANDOM_RESET`, `RANDOM_XY_RANGE`, `RANDOM_RZ_RANGE`
- `ABS_POSE_LIMIT_LOW` / `ABS_POSE_LIMIT_HIGH`
- `TrainConfig.arm`
- `TrainConfig.image_keys`
- `train.setup_mode`
- `TrainConfig.task_desc`
- `TrainConfig.octo_path`
- `offline_training.batch_size`
- `offline_training.pretrain_steps`
- `offline_training.pretrain.q_weight` / `offline_training.pretrain.bc_weight`
- `offline_training.learner.q_weight` / `offline_training.learner.bc_weight`
- `offline_training.pretrain.xla_mem_fraction`
- `task.target_left_pose` / `task.target_right_pose`
- `task.position_tolerance_m` / `task.orientation_tolerance_rad`
- `task.success_reward` and dense reward weights
- `teleop.calibrate_seconds`, `teleop.trans_deadzone`, `teleop.rot_deadzone`
- `teleop.activate_threshold` / `teleop.release_threshold`
- `gripper.fixed_open` / `gripper.open_value`
- `control.hz` / `control.xyz_scale` / `control.rot_scale`
- `control.debug_effective_hz`

Default control mode is `ee_pose_servo`, which is the recommended mode for the
first end-to-end run because SpaceMouse commands are expressed as end-effector
pose deltas.

About `SERVER_URL`:

- `r1lite_reach_target` now reads `SERVER_URL` from the `ROBOT` environment
  variable first
- if `ROBOT` is unset, it falls back to `http://127.0.0.1:8001/`
- otherwise the default comes from `config.yaml`

About `octo_path`:

- the default value now follows the Octo repo recommendation:
  `hf://rail-berkeley/octo-small-1.5`
- `OctoModel.load_pretrained(...)` will download and cache the checkpoint under
  the current user automatically
- if you already have a local checkpoint, override it with:

```bash
export OCTO_PATH=/path/to/your/octo_checkpoint
```

- if you want to pre-download the model manually, the upstream Octo examples and
  README use the same Hugging Face identifier:
  `hf://rail-berkeley/octo-small-1.5`

Advanced note:

- if you want to use a different experiment config file entirely, set:

```bash
export R1LITE_REACH_CONFIG=/path/to/your/config.yaml
```

## 3. Verify SpaceMouse Teleop

Before recording demos, verify direct teleoperation:

```bash
export R1LITE_REACH_CONFIG=/home/robot/VLA-RL/conrft-r1lite/examples/experiments/r1lite_reach_target/config.yaml
cd serl_robot_infra
python -m r1lite_env.spacemouse_teleop --server-url "$ROBOT" --arm right
```

Recommended checks:

- robot responds smoothly to 6DoF input
- if `gripper.fixed_open=true`, the reach task keeps the gripper open and ignores SpaceMouse gripper buttons during demo recording
- `GET /health` shows `command_owner=teleop`
- `GET /health` shows `active_teleop_source=spacemouse`

If this fails, fix teleop first. Demo recording and HIL training both depend on
the same SpaceMouse path.

## 4. Record Demonstrations

R1Lite demos are collected directly from the env, not from rosbag conversion.
The dedicated script is:

```bash
cd examples/experiments/r1lite_reach_target
bash run_record_demos_octo.sh
```

If you want to force a specific local checkpoint for this step:

```bash
cd examples/experiments/r1lite_reach_target
export OCTO_PATH=/path/to/your/octo_checkpoint
bash run_record_demos_octo.sh
```

Equivalent direct command:

```bash
cd examples
python record_demos_r1lite_octo.py \
    --exp_name=r1lite_reach_target \
    --successes_needed=20
```

What the script does:

- creates the real R1Lite env
- enables `R1LiteTeleopInterventionWrapper`
- rolls out zero policy actions and records `info["intervene_action"]` when the
  operator takes over
- keeps only successful trajectories
- adds `mc_returns`
- adds Octo `embeddings`
- adds `next_embeddings`
- writes a ConRFT-ready `.pkl` file to
  `examples/experiments/r1lite_reach_target/demo_data`

## 5. Stage I: Cal-ConRFT Pretrain

Set `DEMO_PATH` to the demo file recorded in the previous step:

```bash
cd examples/experiments/r1lite_reach_target
export DEMO_PATH=./demo_data/r1lite_reach_target_20_demos_<timestamp>.pkl
bash run_learner_conrft_pretrain.sh
```

This stage is learner-only. It uses the demonstration buffer to calibrate the
policy before online exploration.

Default script behavior:

- defaults now come from `offline_training` in
  [config.yaml](../examples/experiments/r1lite_reach_target/config.yaml)
- commonly tuned fields are:
  - `offline_training.batch_size`
  - `offline_training.pretrain_steps`
  - `offline_training.pretrain.q_weight`
  - `offline_training.pretrain.bc_weight`
  - `offline_training.pretrain.xla_mem_fraction`

### W&B Curves During Offline Pretrain

The offline learner logs `update_info` from
[conrft_single_octo_cp.py](../serl_launcher/serl_launcher/agents/continuous/conrft_single_octo_cp.py)
to W&B. The most important curves are:

- `actor_loss`
  The total policy loss used to update the actor. It is the weighted sum of
  behavior cloning loss and Q-guidance loss:
  `actor_loss = bc_weight * bc_loss + q_weight * (-q_loss)`.
  This is a mixed objective, so its absolute value is less important than its
  long-term trend and stability.

- `bc_loss`
  The behavior cloning / consistency loss. This tells you how well the policy
  matches demo actions. In offline pretrain this is usually one of the most
  important curves. A gradual decrease is generally a good sign.

- `mse`
  Mean squared error between predicted actions and demo actions. This is the
  easiest “is the policy imitating the demos” indicator to read. It should
  usually decrease during successful offline warm-up.

- `q_loss`
  Logged as the mean Q value of the current policy actions. Because the actor
  objective uses `-Q`, a larger `q_loss` here usually means the critic thinks
  the actor’s actions are better. Read it together with `bc_loss`, not alone.

- `q_weight`
  The current weight of the Q-guidance term in actor training.

- `bc_weight`
  The current weight of the behavior cloning term in actor training.

- `critic_loss`
  Total critic loss. In Cal-QL this includes both TD loss and CQL penalty:
  `critic_loss = td_loss + cql_alpha * cql_loss`.
  This curve can be noisy; watch for explosions or sustained divergence.

- `td_loss`
  Temporal-difference regression loss for the critic. This shows how well the
  critic fits bootstrap targets from offline data.

- `cql_loss`
  Conservative Q-Learning penalty. It pushes the critic to avoid assigning
  unrealistically high values to unsupported actions. This is expected to be
  non-zero in offline training.

- `cql_alpha`
  Weight applied to the conservative penalty. In the current implementation it
  is fixed by config, so this curve is mainly useful for confirming the run is
  using the expected setting.

- `cql_diff`
  The raw Q-gap used before clipping in the CQL term. Large positive values
  typically mean the critic is trying to score out-of-distribution actions too
  highly; very unstable swings here often correlate with critic instability.

- `calql_bound_rate`
  Fraction of sampled Q values that fall below the Monte Carlo return lower
  bound used by Cal-QL. This is a Cal-QL-specific diagnostic and is mainly for
  relative comparison between runs, not for a strict target value.

- `predicted_qs`
  Mean Q predicted by the critic on the batch actions.

- `target_qs`
  Mean bootstrap target used to train the critic.

- `rewards`
  Mean batch reward from demo transitions. In pure offline pretrain on a fixed
  dataset, this should usually stay roughly stationary. It is a dataset sanity
  check, not a learning-progress metric.

- `actor_lr`, `critic_lr`
  Optimizer learning rates. These are useful mainly to confirm schedules or
  debug unexpected training behavior.

- `timer/*`
  Average runtime breakdowns logged from the training loop. These help diagnose
  slow data loading, replay sampling, or update steps, but they are not RL
  quality metrics.

How to read the curves quickly:

- If `bc_loss` and `mse` go down while `critic_loss` stays bounded, offline
  pretrain is usually behaving reasonably.
- If `critic_loss`, `td_loss`, or `cql_diff` suddenly explode, the critic is
  likely unstable; try smaller `offline_training.batch_size` first.
- If `bc_loss` barely moves, inspect the demo quality, action scaling, and
  whether the dataset matches the current observation schema.
- If W&B shows `rewards` drifting a lot during pure offline pretrain, that is
  often a sign of dataset mixing or loading problems rather than actual policy
  improvement.

## 6. Stage II: HIL-ConRFT Online Training

Run both threads:

```bash
cd examples/experiments/r1lite_reach_target
export DEMO_PATH=./demo_data/r1lite_reach_target_20_demos_<timestamp>.pkl
bash run_learner_conrft.sh
bash run_actor_conrft.sh
```

During actor rollout:

- the actor owns `owner="policy"`
- SpaceMouse intervention happens inside the env wrapper
- intervention transitions are inserted into the intervention/demo data store
- online rollouts are mixed with offline demos by the learner

Use SpaceMouse whenever the policy drifts or explores inefficiently.

## 7. Recommended Files And Outputs

Important inputs:

- experiment config:
  [config.py](../examples/experiments/r1lite_reach_target/config.py)
- reward wrapper:
  [wrapper.py](../examples/experiments/r1lite_reach_target/wrapper.py)
- demo script:
  [record_demos_r1lite_octo.py](../examples/record_demos_r1lite_octo.py)

Expected outputs:

- demos:
  `examples/experiments/r1lite_reach_target/demo_data/*.pkl`
- checkpoints:
  `examples/experiments/r1lite_reach_target/conrft/`

## 8. Current Gaps Compared To The Full Franka Workflow

The current R1Lite path is intentionally simpler than the Franka banana task:

- no classifier data collection step
- no learned visual reward classifier for success detection
- reward comes from geometric pose error

This makes `r1lite_reach_target` the recommended first task for bringing up the
full ConRFT actor/learner loop on R1Lite.
