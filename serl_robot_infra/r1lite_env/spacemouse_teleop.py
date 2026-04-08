import argparse
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from franka_env.spacemouse.spacemouse_expert import SpaceMouseExpert
from r1lite_env.client import R1LiteClient


def _default_config_yaml() -> Optional[str]:
    env_path = os.environ.get("R1LITE_REACH_CONFIG")
    if env_path:
        return env_path
    # 从 serl_robot_infra/r1lite_env 回到仓库根目录，再定位 reach_target 的默认配置文件。
    candidate = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "experiments"
        / "r1lite_reach_target"
        / "config.yaml"
    )
    return str(candidate) if candidate.exists() else None


def _load_control_defaults(config_yaml: Optional[str]) -> Dict[str, float]:
    if not config_yaml:
        return {}
    path = Path(config_yaml)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        return {}
    control = data.get("control", {})
    return control if isinstance(control, dict) else {}


def _estimate_idle_bias(expert: SpaceMouseExpert, duration_sec: float) -> np.ndarray:
    if duration_sec <= 0.0:
        action, _ = expert.get_action()
        return np.zeros_like(np.asarray(action, dtype=np.float64))

    deadline = time.time() + duration_sec
    samples = []
    while time.time() < deadline:
        action, _ = expert.get_action()
        samples.append(np.asarray(action, dtype=np.float64))
        time.sleep(0.01)
    if not samples:
        action, _ = expert.get_action()
        return np.zeros_like(np.asarray(action, dtype=np.float64))
    return np.mean(np.stack(samples, axis=0), axis=0)


def _apply_deadzone(action: np.ndarray, trans_deadzone: float, rot_deadzone: float) -> np.ndarray:
    filtered = np.asarray(action, dtype=np.float64).copy()
    if filtered.shape[0] >= 3:
        filtered[:3][np.abs(filtered[:3]) < trans_deadzone] = 0.0
    if filtered.shape[0] >= 6:
        filtered[3:6][np.abs(filtered[3:6]) < rot_deadzone] = 0.0
    if filtered.shape[0] >= 9:
        filtered[6:9][np.abs(filtered[6:9]) < trans_deadzone] = 0.0
    if filtered.shape[0] >= 12:
        filtered[9:12][np.abs(filtered[9:12]) < rot_deadzone] = 0.0
    return filtered


def _has_button_press(buttons) -> bool:
    return any(bool(value) for value in buttons)


def _pose_target_from_action(tcp_pose: np.ndarray, action: np.ndarray, xyz_scale: float, rot_scale: float) -> np.ndarray:
    pose = np.asarray(tcp_pose, dtype=np.float64).copy()
    delta = np.asarray(action[:6], dtype=np.float64)
    pose[:3] = pose[:3] + delta[:3] * xyz_scale
    pose[3:] = (Rotation.from_euler("xyz", delta[3:] * rot_scale) * Rotation.from_quat(pose[3:])).as_quat()
    return pose


def _single_arm_gripper(buttons) -> Optional[float]:
    if len(buttons) >= 1 and buttons[0]:
        return 0.0
    if len(buttons) >= 2 and buttons[-1]:
        return 100.0
    return None


def _dual_arm_grippers(buttons) -> Tuple[Optional[float], Optional[float]]:
    if len(buttons) != 4:
        return None, None
    left = 0.0 if buttons[0] else 100.0 if buttons[1] else None
    right = 0.0 if buttons[2] else 100.0 if buttons[3] else None
    return left, right


def _pose_matches(raw_pose: np.ndarray, accepted_pose: np.ndarray, atol: float = 1e-4) -> bool:
    return bool(np.allclose(np.asarray(raw_pose, dtype=np.float64), np.asarray(accepted_pose, dtype=np.float64), atol=atol))


def _format_pose(pose: np.ndarray) -> str:
    pose = np.asarray(pose, dtype=np.float64)
    return np.array2string(pose, precision=4, suppress_small=True)


def _build_single_arm_payload(
    state: Dict,
    arm: str,
    action: np.ndarray,
    buttons,
    seq: int,
    mode: str,
    preset: str,
    xyz_scale: float,
    rot_scale: float,
) -> Dict:
    tcp_pose = np.asarray(state["state"][arm]["tcp_pose"], dtype=np.float64)
    arm_payload = {
        "pose_target": _pose_target_from_action(tcp_pose, action[:6], xyz_scale, rot_scale).tolist(),
        "preset": preset,
    }
    gripper = _single_arm_gripper(buttons)
    if gripper is not None:
        arm_payload["gripper"] = gripper
    return {
        "owner": "teleop",
        "teleop_source": "spacemouse",
        "mode": mode,
        "seq": seq,
        arm: arm_payload,
    }


def _build_dual_arm_payload(
    state: Dict,
    action: np.ndarray,
    buttons,
    seq: int,
    mode: str,
    preset: str,
    xyz_scale: float,
    rot_scale: float,
) -> Dict:
    left_pose = np.asarray(state["state"]["left"]["tcp_pose"], dtype=np.float64)
    right_pose = np.asarray(state["state"]["right"]["tcp_pose"], dtype=np.float64)
    left_gripper, right_gripper = _dual_arm_grippers(buttons)

    left_payload = {
        "pose_target": _pose_target_from_action(left_pose, action[:6], xyz_scale, rot_scale).tolist(),
        "preset": preset,
    }
    right_payload = {
        "pose_target": _pose_target_from_action(right_pose, action[6:12], xyz_scale, rot_scale).tolist(),
        "preset": preset,
    }
    if left_gripper is not None:
        left_payload["gripper"] = left_gripper
    if right_gripper is not None:
        right_payload["gripper"] = right_gripper

    return {
        "owner": "teleop",
        "teleop_source": "spacemouse",
        "mode": mode,
        "seq": seq,
        "left": left_payload,
        "right": right_payload,
    }


def run() -> None:
    default_config_yaml = _default_config_yaml()
    control_defaults = _load_control_defaults(default_config_yaml)
    parser = argparse.ArgumentParser(description="Control R1Lite via SpaceMouse teleop owner")
    parser.add_argument("--config-yaml", default=default_config_yaml, help="Optional YAML config file to load control.hz / xyz_scale / rot_scale defaults")
    parser.add_argument("--server-url", default="http://127.0.0.1:8001/")
    parser.add_argument("--arm", default="right", choices=["left", "right", "dual"])
    parser.add_argument("--hz", type=float, default=float(control_defaults.get("hz", 10.0)))
    parser.add_argument("--xyz-scale", type=float, default=float(control_defaults.get("xyz_scale", 0.03)))
    parser.add_argument("--rot-scale", type=float, default=float(control_defaults.get("rot_scale", 0.20)))
    parser.add_argument("--preset", default="free_space")
    parser.add_argument("--mode", default=None, help="Defaults to the current service active_mode")
    parser.add_argument("--calibrate-seconds", type=float, default=0.5, help="Keep the SpaceMouse still at startup to estimate zero bias")
    parser.add_argument("--trans-deadzone", type=float, default=0.08, help="Deadzone for translation axes after bias removal")
    parser.add_argument("--rot-deadzone", type=float, default=0.08, help="Deadzone for rotation axes after bias removal")
    parser.add_argument(
        "--debug-target-pose",
        action="store_true",
        help="Print the raw pose target, the pose accepted by the server, and whether they match.",
    )
    parser.add_argument(
        "--debug-log-hz",
        type=float,
        default=2.0,
        help="Maximum debug print frequency for target-pose comparison.",
    )
    parser.add_argument(
        "--debug-effective-hz",
        action="store_true",
        default=bool(control_defaults.get("debug_effective_hz", False)),
        help="Print the effective command loop frequency once per second.",
    )
    args = parser.parse_args()

    client = R1LiteClient(args.server_url)
    expert = SpaceMouseExpert()
    print(f"[teleop] calibrating SpaceMouse for {args.calibrate_seconds:.2f}s, keep it untouched...")
    bias = _estimate_idle_bias(expert, args.calibrate_seconds)
    print(f"[teleop] calibration complete")
    print(f"[teleop] SpaceMouse bias: {np.array2string(bias, precision=4, suppress_small=True)}")
    print(
        "[teleop] deadzone config: "
        f"trans={args.trans_deadzone:.3f}, rot={args.rot_deadzone:.3f}, "
        f"xyz_scale={args.xyz_scale:.3f}, rot_scale={args.rot_scale:.3f}"
    )
    print(
        f"[teleop] control config: config_yaml={args.config_yaml}, hz={args.hz:.2f}, "
        f"debug_effective_hz={bool(args.debug_effective_hz)}"
    )
    seq = 0
    last_log_time = 0.0
    last_debug_time = 0.0
    hz_counter = 0
    hz_log_start_time = time.time()
    step_dt = 1.0 / max(args.hz, 1e-6)

    try:
        while True:
            start_time = time.time()
            state = client.get_state()
            mode = args.mode or state["meta"]["mode"]
            action, buttons = expert.get_action()
            action = _apply_deadzone(
                np.asarray(action, dtype=np.float64) - bias,
                trans_deadzone=args.trans_deadzone,
                rot_deadzone=args.rot_deadzone,
            )

            if np.allclose(action, 0.0) and not _has_button_press(buttons):
                time.sleep(max(0.0, step_dt - (time.time() - start_time)))
                continue

            if args.arm == "dual":
                if len(action) < 12:
                    raise RuntimeError("dual arm teleop requires two SpaceMouse devices")
                left_raw_pose = _pose_target_from_action(
                    np.asarray(state["state"]["left"]["tcp_pose"], dtype=np.float64),
                    action[:6],
                    args.xyz_scale,
                    args.rot_scale,
                )
                right_raw_pose = _pose_target_from_action(
                    np.asarray(state["state"]["right"]["tcp_pose"], dtype=np.float64),
                    action[6:12],
                    args.xyz_scale,
                    args.rot_scale,
                )
                payload = _build_dual_arm_payload(
                    state=state,
                    action=action,
                    buttons=buttons,
                    seq=seq,
                    mode=mode,
                    preset=args.preset,
                    xyz_scale=args.xyz_scale,
                    rot_scale=args.rot_scale,
                )
            else:
                if len(action) < 6:
                    raise RuntimeError("failed to read a valid SpaceMouse 6DoF action")
                raw_pose = _pose_target_from_action(
                    np.asarray(state["state"][args.arm]["tcp_pose"], dtype=np.float64),
                    action[:6],
                    args.xyz_scale,
                    args.rot_scale,
                )
                payload = _build_single_arm_payload(
                    state=state,
                    arm=args.arm,
                    action=action,
                    buttons=buttons,
                    seq=seq,
                    mode=mode,
                    preset=args.preset,
                    xyz_scale=args.xyz_scale,
                    rot_scale=args.rot_scale,
                )

            response = client.post_action(payload)
            if args.debug_effective_hz:
                hz_counter += 1
                now = time.time()
                dt = now - hz_log_start_time
                if dt >= 1.0:
                    print(f"[teleop-loop] effective_control_hz={hz_counter / max(dt, 1e-6):.2f}")
                    hz_counter = 0
                    hz_log_start_time = now
            if time.time() - last_log_time > 1.0:
                print(
                    f"seq={seq} owner={response.get('owner')} teleop_source={response.get('teleop_source')} "
                    f"published={response.get('published')}"
                )
                last_log_time = time.time()

            if args.debug_target_pose and (time.time() - last_debug_time) >= (1.0 / max(args.debug_log_hz, 1e-6)):
                health = client.get_health()
                commands = health.get("commands", {})
                if args.arm == "dual":
                    accepted_left = commands.get("left", {}).get("desired_pose")
                    accepted_right = commands.get("right", {}).get("desired_pose")
                    if accepted_left is not None and accepted_right is not None:
                        accepted_left = np.asarray(accepted_left, dtype=np.float64)
                        accepted_right = np.asarray(accepted_right, dtype=np.float64)
                        left_match = _pose_matches(left_raw_pose, accepted_left)
                        right_match = _pose_matches(right_raw_pose, accepted_right)
                        print(
                            "[teleop-debug] left raw="
                            f"{_format_pose(left_raw_pose)} accepted={_format_pose(accepted_left)} "
                            f"match={left_match} clipped={not left_match}"
                        )
                        print(
                            "[teleop-debug] right raw="
                            f"{_format_pose(right_raw_pose)} accepted={_format_pose(accepted_right)} "
                            f"match={right_match} clipped={not right_match}"
                        )
                else:
                    accepted_pose = commands.get(args.arm, {}).get("desired_pose")
                    if accepted_pose is not None:
                        accepted_pose = np.asarray(accepted_pose, dtype=np.float64)
                        matches = _pose_matches(raw_pose, accepted_pose)
                        print(
                            "[teleop-debug] raw="
                            f"{_format_pose(raw_pose)} accepted={_format_pose(accepted_pose)} "
                            f"match={matches} clipped={not matches}"
                        )
                last_debug_time = time.time()
            seq += 1
            time.sleep(max(0.0, step_dt - (time.time() - start_time)))
    finally:
        expert.close()


if __name__ == "__main__":
    run()
