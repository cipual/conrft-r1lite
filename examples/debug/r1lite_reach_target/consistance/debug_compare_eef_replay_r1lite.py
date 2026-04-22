#!/usr/bin/env python3

import argparse
import csv
import pickle
import time
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation


def _ensure_examples_on_path():
    root = Path(__file__).resolve().parent
    repo_root = root.parent.parent.parent
    for path in (root, repo_root / "serl_robot_infra"):
        path_str = str(path)
        if path_str not in __import__("sys").path:
            __import__("sys").path.insert(0, path_str)


_ensure_examples_on_path()

from r1lite_env.client import R1LiteClient  # noqa: E402


def _load_transitions(pkl_path: Path) -> List[Dict]:
    with pkl_path.open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError(f"Expected non-empty transition list in {pkl_path}")
    return data


def _split_trajectories(transitions: List[Dict]) -> List[List[Dict]]:
    trajectories: List[List[Dict]] = []
    current: List[Dict] = []
    for transition in transitions:
        current.append(transition)
        if bool(transition.get("dones", False)):
            trajectories.append(current)
            current = []
    if current:
        trajectories.append(current)
    return trajectories


def _quat_angle_error_rad(current_xyzw: np.ndarray, target_xyzw: np.ndarray) -> float:
    if float(np.linalg.norm(current_xyzw)) < 1e-8 or float(np.linalg.norm(target_xyzw)) < 1e-8:
        return float("nan")
    current = Rotation.from_quat(current_xyzw)
    target = Rotation.from_quat(target_xyzw)
    delta = current.inv() * target
    return float(np.linalg.norm(delta.as_rotvec()))


def _target_pose_from_eef_delta(reference_pose: np.ndarray, eef_delta: np.ndarray) -> np.ndarray:
    target = np.asarray(reference_pose, dtype=np.float32).copy()
    target[:3] = target[:3] + np.asarray(eef_delta[:3], dtype=np.float32)
    target_rot = Rotation.from_quat(reference_pose[3:]) * Rotation.from_rotvec(eef_delta[3:6])
    target[3:] = target_rot.as_quat().astype(np.float32)
    return target


def _arms_for_replay(arm: str) -> List[str]:
    return ["left", "right"] if arm == "dual" else [arm]


def _obs_for_arm(transition: Dict, arm: str, obs_key: str) -> Dict:
    obs = transition[obs_key]
    if arm in obs and isinstance(obs[arm], dict):
        return obs[arm]
    return obs


def _action_for_arm(transition: Dict, arm: str) -> Dict:
    actions = transition["actions"]
    if arm in actions and isinstance(actions[arm], dict):
        return actions[arm]
    return actions


def _gripper_target_from_delta(current_gripper: np.ndarray, action: Dict) -> float | None:
    if "gripper_delta" in action:
        target = np.asarray(current_gripper, dtype=np.float32) + np.asarray(action["gripper_delta"], dtype=np.float32)
        return float(target.reshape(-1)[0])
    if "gripper_target" in action:
        return float(np.asarray(action["gripper_target"], dtype=np.float32).reshape(-1)[0])
    return None


def _recorded_pose_array(trajectory: List[Dict], arms: List[str]) -> Dict[str, np.ndarray]:
    return {
        arm: np.asarray(
            [np.asarray(_obs_for_arm(t, arm, "next_observations")["tcp_pose"], dtype=np.float32) for t in trajectory],
            dtype=np.float32,
        )
        for arm in arms
    }


def _plot_trajectories_3d(recorded_by_arm: Dict[str, np.ndarray], eef_by_arm: Dict[str, np.ndarray], pose_by_arm: Dict[str, np.ndarray], output_path: Path):
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    colors = {"left": ("#2ca02c", "#1f77b4", "#9467bd"), "right": ("#222222", "#ff7f0e", "#d62728")}
    for arm, recorded in recorded_by_arm.items():
        rec_color, action_color, target_color = colors.get(arm, ("#222222", "#1f77b4", "#d62728"))
        eef_actual = eef_by_arm[arm]
        pose_actual = pose_by_arm[arm]
        ax.plot(recorded[:, 0], recorded[:, 1], recorded[:, 2], label=f"{arm} recorded", color=rec_color, linewidth=2.2)
        ax.plot(eef_actual[:, 0], eef_actual[:, 1], eef_actual[:, 2], label=f"{arm} eef action replay", color=action_color, linewidth=1.8)
        ax.plot(pose_actual[:, 0], pose_actual[:, 1], pose_actual[:, 2], label=f"{arm} pose_target replay", color=target_color, linewidth=1.8)
        ax.scatter(recorded[0, 0], recorded[0, 1], recorded[0, 2], color=rec_color, marker="o", s=28)
        ax.scatter(recorded[-1, 0], recorded[-1, 1], recorded[-1, 2], color=rec_color, marker="x", s=40)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title("EEF-Delta Action vs Pose Target Replay Through Body Service (3D)")
    ax.legend(frameon=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _write_csv(
    trajectory: List[Dict],
    eef_pose_history: Dict[str, np.ndarray],
    eef_gripper_history: Dict[str, np.ndarray],
    pose_pose_history: Dict[str, np.ndarray],
    pose_gripper_history: Dict[str, np.ndarray],
    arms: List[str],
    output_csv: Path,
):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "step",
                "arm",
                "recorded_x",
                "recorded_y",
                "recorded_z",
                "recorded_qx",
                "recorded_qy",
                "recorded_qz",
                "recorded_qw",
                "eef_action_x",
                "eef_action_y",
                "eef_action_z",
                "eef_action_qx",
                "eef_action_qy",
                "eef_action_qz",
                "eef_action_qw",
                "eef_action_pos_err_m",
                "eef_action_ori_err_rad",
                "recorded_gripper",
                "eef_action_gripper",
                "eef_action_gripper_err",
                "pose_target_x",
                "pose_target_y",
                "pose_target_z",
                "pose_target_qx",
                "pose_target_qy",
                "pose_target_qz",
                "pose_target_qw",
                "pose_target_pos_err_m",
                "pose_target_ori_err_rad",
                "pose_target_gripper",
                "pose_target_gripper_err",
                "eef_delta_x_m",
                "eef_delta_y_m",
                "eef_delta_z_m",
                "eef_delta_rx_rad",
                "eef_delta_ry_rad",
                "eef_delta_rz_rad",
                "gripper_delta",
            ]
        )
        for idx, transition in enumerate(trajectory):
            for arm in arms:
                next_obs = _obs_for_arm(transition, arm, "next_observations")
                action = _action_for_arm(transition, arm)
                recorded = np.asarray(next_obs["tcp_pose"], dtype=np.float32)
                recorded_gripper = float(np.asarray(next_obs["gripper_pose"], dtype=np.float32).reshape(-1)[0])
                eef_actual = np.asarray(eef_pose_history[arm][idx], dtype=np.float32)
                pose_actual = np.asarray(pose_pose_history[arm][idx], dtype=np.float32)
                eef_gripper = float(eef_gripper_history[arm][idx])
                pose_gripper = float(pose_gripper_history[arm][idx])
                eef_delta = np.asarray(action["eef_delta"], dtype=np.float32)
                gripper_delta = float(np.asarray(action.get("gripper_delta", [0.0]), dtype=np.float32).reshape(-1)[0])
                writer.writerow(
                    [
                        idx + 1,
                        arm,
                        *recorded.tolist(),
                        *eef_actual.tolist(),
                        float(np.linalg.norm(eef_actual[:3] - recorded[:3])),
                        _quat_angle_error_rad(eef_actual[3:], recorded[3:]),
                        recorded_gripper,
                        eef_gripper,
                        abs(eef_gripper - recorded_gripper),
                        *pose_actual.tolist(),
                        float(np.linalg.norm(pose_actual[:3] - recorded[:3])),
                        _quat_angle_error_rad(pose_actual[3:], recorded[3:]),
                        pose_gripper,
                        abs(pose_gripper - recorded_gripper),
                        *eef_delta.tolist(),
                        gripper_delta,
                    ]
                )


def _run_replay(
    client: R1LiteClient,
    trajectory: List[Dict],
    arm: str,
    replay_mode: str,
    control_hz: float,
    log_every: int,
    reset_wait_sec: float,
) -> tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    arms = _arms_for_replay(arm)
    actual_pose_history: Dict[str, List[np.ndarray]] = {item: [] for item in arms}
    actual_gripper_history: Dict[str, List[float]] = {item: [] for item in arms}

    print(f"Resetting robot before {replay_mode} replay...")
    client.reset(owner="debug")
    time.sleep(max(0.0, reset_wait_sec))
    print(f"Replaying {len(trajectory)} step(s) in {replay_mode} mode through body service")

    for idx, transition in enumerate(trajectory):
        start = time.time()
        payload = {
            "mode": "ee_pose_servo",
            "owner": "policy",
        }
        state = client.get_state()
        if replay_mode == "eef_action":
            eef_delta_norms = []
            for item in arms:
                action = _action_for_arm(transition, item)
                current_pose = np.asarray(state["state"][item]["tcp_pose"], dtype=np.float32)
                current_gripper = np.asarray(state["state"][item]["gripper_pose"], dtype=np.float32)
                eef_delta = np.asarray(action["eef_delta"], dtype=np.float32)
                eef_delta_norms.append(float(np.linalg.norm(eef_delta)))
                command = {
                    "pose_target": _target_pose_from_eef_delta(current_pose, eef_delta).tolist(),
                    "preset": "free_space",
                }
                gripper = _gripper_target_from_delta(current_gripper, action)
                if gripper is not None:
                    command["gripper"] = gripper
                payload[item] = command
        elif replay_mode == "pose_target":
            for item in arms:
                next_obs = _obs_for_arm(transition, item, "next_observations")
                command = {
                    "pose_target": np.asarray(next_obs["tcp_pose"], dtype=np.float32).tolist(),
                    "preset": "free_space",
                }
                if "gripper_pose" in next_obs:
                    command["gripper"] = float(np.asarray(next_obs["gripper_pose"], dtype=np.float32).reshape(-1)[0])
                payload[item] = command
        else:
            raise ValueError(f"Unsupported replay_mode: {replay_mode}")

        client.post_action(payload)
        time.sleep(max(0.0, (1.0 / max(control_hz, 1e-6)) - (time.time() - start)))
        raw = client.get_state()
        if idx % max(1, log_every) == 0 or idx == len(trajectory) - 1:
            parts = []
        for item in arms:
            actual_pose = np.asarray(raw["state"][item]["tcp_pose"], dtype=np.float32)
            actual_gripper = float(np.asarray(raw["state"][item]["gripper_pose"], dtype=np.float32).reshape(-1)[0])
            actual_pose_history[item].append(actual_pose)
            actual_gripper_history[item].append(actual_gripper)
            recorded_next = np.asarray(_obs_for_arm(transition, item, "next_observations")["tcp_pose"], dtype=np.float32)
            pos_err = float(np.linalg.norm(actual_pose[:3] - recorded_next[:3]))
            ori_err = _quat_angle_error_rad(actual_pose[3:], recorded_next[3:])
            if idx % max(1, log_every) == 0 or idx == len(trajectory) - 1:
                parts.append(f"{item}:pos_err={pos_err:.4f}m ori_err={ori_err:.4f}rad")
        if idx % max(1, log_every) == 0 or idx == len(trajectory) - 1:
            extra = ""
            if replay_mode == "eef_action":
                extra = " eef_delta_norms=" + ",".join(f"{v:.4f}" for v in eef_delta_norms)
            print(
                f"[eef-compare] step={idx + 1}/{len(trajectory)} "
                f"replay_mode={replay_mode} {' '.join(parts)}{extra}"
            )

    print(f"Resetting robot after {replay_mode} replay...")
    client.reset(owner="debug")
    time.sleep(max(0.0, reset_wait_sec))
    return (
        {item: np.asarray(values, dtype=np.float32) for item, values in actual_pose_history.items()},
        {item: np.asarray(values, dtype=np.float32) for item, values in actual_gripper_history.items()},
    )


def main():
    parser = argparse.ArgumentParser(description="Compare physical EEF-delta action replay and pose-target replay through body service HTTP.")
    parser.add_argument("--input_file", required=True, help="Path to EEF debug transition .pkl")
    parser.add_argument("--trajectory_index", type=int, default=0)
    parser.add_argument("--arm", default="right", choices=("left", "right", "dual"))
    parser.add_argument("--server_url", default="http://192.168.12.12:8001/")
    parser.add_argument("--control_hz", type=float, default=10.0)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--reset_wait_sec", type=float, default=1.0)
    parser.add_argument(
        "--output_image_3d",
        default="/home/robot/VLA-RL/conrft-r1lite/examples/debug/r1lite_reach_target/consistance/r1lite_debug_eef_replay_compare_3d.png",
    )
    parser.add_argument(
        "--output_csv",
        default="/home/robot/VLA-RL/conrft-r1lite/examples/debug/r1lite_reach_target/consistance/r1lite_debug_eef_replay_compare_errors.csv",
    )
    parser.add_argument(
        "--output_npz",
        default="/home/robot/VLA-RL/conrft-r1lite/examples/debug/r1lite_reach_target/consistance/r1lite_debug_eef_replay_compare.npz",
    )
    args = parser.parse_args()

    transitions = _load_transitions(Path(args.input_file).expanduser().resolve())
    trajectories = _split_trajectories(transitions)
    if args.trajectory_index < 0 or args.trajectory_index >= len(trajectories):
        raise IndexError(f"trajectory_index={args.trajectory_index} out of range for {len(trajectories)} trajectories")
    trajectory = trajectories[args.trajectory_index]

    client = R1LiteClient(args.server_url)
    try:
        eef_pose_arr, eef_gripper_arr = _run_replay(
            client=client,
            trajectory=trajectory,
            arm=args.arm,
            replay_mode="eef_action",
            control_hz=args.control_hz,
            log_every=args.log_every,
            reset_wait_sec=args.reset_wait_sec,
        )
        pose_pose_arr, pose_gripper_arr = _run_replay(
            client=client,
            trajectory=trajectory,
            arm=args.arm,
            replay_mode="pose_target",
            control_hz=args.control_hz,
            log_every=args.log_every,
            reset_wait_sec=args.reset_wait_sec,
        )
    finally:
        client.close()

    arms = _arms_for_replay(args.arm)
    recorded = _recorded_pose_array(trajectory, arms)
    output_image_3d = Path(args.output_image_3d).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve()
    output_npz = Path(args.output_npz).expanduser().resolve()
    _plot_trajectories_3d(recorded, eef_pose_arr, pose_pose_arr, output_image_3d)
    _write_csv(trajectory, eef_pose_arr, eef_gripper_arr, pose_pose_arr, pose_gripper_arr, arms, output_csv)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_npz,
        **{f"{arm}_recorded": arr for arm, arr in recorded.items()},
        **{f"{arm}_eef_action_actual_pose": arr for arm, arr in eef_pose_arr.items()},
        **{f"{arm}_eef_action_actual_gripper": arr for arm, arr in eef_gripper_arr.items()},
        **{f"{arm}_pose_target_actual_pose": arr for arm, arr in pose_pose_arr.items()},
        **{f"{arm}_pose_target_actual_gripper": arr for arm, arr in pose_gripper_arr.items()},
    )
    print(f"Saved 3D trajectory comparison image to {output_image_3d}")
    print(f"Saved per-step error csv to {output_csv}")
    print(f"Saved raw replay arrays to {output_npz}")


if __name__ == "__main__":
    main()
