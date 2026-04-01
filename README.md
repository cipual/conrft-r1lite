# ConRFT R1Lite Inference Guide

This `README.md` is the running inference-side integration guide for `conrft-r1lite`. New teleop, monitor, and debugging features should be documented here as they are added.

The original research-oriented repository README has been preserved at [docs/research_readme_original.md](./docs/research_readme_original.md).

## Service Address

Set the robot service address first:

```bash
export ROBOT=http://127.0.0.1:8001
```

If the robot service runs on another machine:

```bash
export ROBOT=http://192.168.12.12:8001
```

## Permission Table

| Owner | Allowed endpoints | Purpose |
| --- | --- | --- |
| public | `GET /state`, `GET /health` | read-only inspection |
| policy | `POST /action` | inference env control, including policy rollout |
| teleop | `POST /action` | human teleoperation only; requires `teleop_source="spacemouse"` |
| debug | `POST /reset`, `POST /recover`, `POST /clear_fault`, `POST /brake` | maintenance and recovery only |

Rules:

- `POST /action` rejects `owner="debug"` with HTTP 403.
- `POST /action` requires `teleop_source="spacemouse"` when `owner="teleop"`.
- `POST /reset`, `POST /recover`, `POST /clear_fault`, and `POST /brake` reject `owner!="debug"` with HTTP 403.
- `policy` and `teleop` share the control-owner timeout lock. If one is active, the other receives HTTP 409 until the lock expires.

## Quick HTTP Checks

Use `--noproxy '*'` on the inference machine so `curl` matches the Python client behavior and talks to the robot service directly.

Read current state without image payload:

```bash
curl --noproxy '*' -s "$ROBOT/state" | jq '{state, meta}'
```

Read image payload only:

```bash
curl --noproxy '*' -s "$ROBOT/state" | jq '.images'
```

Read health:

```bash
curl --noproxy '*' -X GET "$ROBOT/health"
```

Send a policy action:

```bash
curl --noproxy '*' -X POST "$ROBOT/action" \
  -H "Content-Type: application/json" \
  -d '{
    "owner":"policy",
    "seq":1,
    "left":{
      "pose_delta":[0.0,0.0,0.005,0.0,0.0,0.0],
      "gripper":20.0,
      "preset":"free_space"
    }
  }'
```

Send a teleop action:

```bash
curl --noproxy '*' -X POST "$ROBOT/action" \
  -H "Content-Type: application/json" \
  -d '{
    "owner":"teleop",
    "teleop_source":"spacemouse",
    "seq":2,
    "right":{
      "pose_target":[0.35,0.25,0.32,0.0,1.0,0.0,0.0],
      "gripper":30.0,
      "preset":"free_space"
    }
  }'
```

Reset:

```bash
curl --noproxy '*' -X POST "$ROBOT/reset" \
  -H "Content-Type: application/json" \
  -d '{"owner":"debug"}'
```

Recover:

```bash
curl --noproxy '*' -X POST "$ROBOT/recover" \
  -H "Content-Type: application/json" \
  -d '{"owner":"debug"}'
```

Clear faults:

```bash
curl --noproxy '*' -X POST "$ROBOT/clear_fault" \
  -H "Content-Type: application/json" \
  -d '{"owner":"debug"}'
```

Enable brake:

```bash
curl --noproxy '*' -X POST "$ROBOT/brake" \
  -H "Content-Type: application/json" \
  -d '{"enabled": true, "owner":"debug"}'
```

Disable brake:

```bash
curl --noproxy '*' -X POST "$ROBOT/brake" \
  -H "Content-Type: application/json" \
  -d '{"enabled": false, "owner":"debug"}'
```

## Notes

- If the service was started in MIT mode, keep `mode` in `/action` aligned with the startup mode.
- HTTP 409 usually means the active control owner is still locked by another control session, or the requested mode does not match the startup mode.

## SpaceMouse Teleop

Run this from the inference machine workspace after the Python environment is ready:

```bash
cd ~/VLA-RL/conrft-r1lite/serl_robot_infra
python -m r1lite_env.spacemouse_teleop --server-url "$ROBOT" --arm right
```

Useful options:

- `--arm left|right|dual`
- `--hz 10`
- `--xyz-scale 0.03`
- `--rot-scale 0.20`
- `--preset free_space`
- `--calibrate-seconds 0.5`
- `--trans-deadzone 0.08`
- `--rot-deadzone 0.08`

Startup behavior:

- `spacemouse_teleop` performs a short zero-bias calibration on startup. Keep the SpaceMouse untouched during this window.
- Calibration logs are printed in the terminal, including estimated bias and deadzone configuration.

## GUI Monitor

Run this from the inference machine workspace:

```bash
cd ~/VLA-RL/conrft-r1lite/serl_robot_infra
python -m r1lite_env.monitor_gui --server-url "$ROBOT" --image-hz 5 --state-period 0.5
```

What the GUI shows:

- left panel: left arm command target, left arm state, `left_wrist` image
- center panel: shared `head` image and maintenance buttons
- right panel: right arm command target, right arm state, `right_wrist` image
- bottom logs: `info`, `warning`, `fault`

GUI controls:

- `Brake Toggle`
- `Reset`
- `Clear Fault`
- `Refresh Now`

GUI refresh behavior:

- images refresh continuously according to `--image-hz`
- text state and command panels refresh every `--state-period` seconds
- `info` logs include owner changes, teleop source changes, brake changes, and SpaceMouse teleop activation
- `warning` logs include freshness / validity warnings
- `fault` logs include body-service faults and request failures
