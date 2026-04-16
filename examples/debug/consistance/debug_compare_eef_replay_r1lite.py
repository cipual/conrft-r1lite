#!/usr/bin/env python3

import argparse
import csv
import pickle
import time
from pathlib import Path
from typing import Dict, List, Tuple

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


def _plot_trajectories_3d(recorded: np.ndarray, eef_actual: np.ndarray, pose_actual: np.ndarray, output_path: Path):
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(recorded[:, 0], recorded[:, 1], recorded[:, 2], label="recorded", color="#222222", linewidth=2.2)
    ax.plot(eef_actual[:, 0], eef_actual[:, 1], eef_actual[:, 2], label="eef action replay", color="#1f77b4", linewidth=1.8)
    ax.plot(pose_actual[:, 0], pose_actual[:, 1], pose_actual[:, 2], label="pose_target replay", color="#d62728", linewidth=1.8)
    ax.scatter(recorded[0, 0], recorded[0, 1], recorded[0, 2], color="#222222", marker="o", s=28)
    ax.scatter(recorded[-1, 0], recorded[-1, 1], recorded[-1, 2], color="#222222", marker="x", s=40)
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
    eef_pose_history: np.ndarray,
    pose_pose_history: np.ndarray,
    output_csv: Path,
):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "step",
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
                "pose_target_x",
                "pose_target_y",
                "pose_target_z",
                "pose_target_qx",
                "pose_target_qy",
                "pose_target_qz",
                "pose_target_qw",
                "pose_target_pos_err_m",
                "pose_target_ori_err_rad",
                "eef_delta_x_m",
                "eef_delta_y_m",
                "eef_delta_z_m",
                "eef_delta_rx_rad",
                "eef_delta_ry_rad",
                "eef_delta_rz_rad",
            ]
        )
        for idx, transition in enumerate(trajectory):
            recorded = np.asarray(transition["next_observations"]["tcp_pose"], dtype=np.float32)
            eef_actual = np.asarray(eef_pose_history[idx], dtype=np.float32)
            pose_actual = np.asarray(pose_pose_history[idx], dtype=np.float32)
            eef_delta = np.asarray(transition["actions"]["eef_delta"], dtype=np.float32)
            writer.writerow(
                [
                    idx + 1,
                    *recorded.tolist(),
                    *eef_actual.tolist(),
                    float(np.linalg.norm(eef_actual[:3] - recorded[:3])),
                    _quat_angle_error_rad(eef_actual[3:], recorded[3:]),
                    *pose_actual.tolist(),
                    float(np.linalg.norm(pose_actual[:3] - recorded[:3])),
                    _quat_angle_error_rad(pose_actual[3:], recorded[3:]),
                    *eef_delta.tolist(),
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
) -> np.ndarray:
    actual_pose_history: List[np.ndarray] = []

    print(f"Resetting robot before {replay_mode} replay...")
    client.reset(owner="debug")
    time.sleep(max(0.0, reset_wait_sec))
    print(f"Replaying {len(trajectory)} step(s) in {replay_mode} mode through body service")

    for idx, transition in enumerate(trajectory):
        start = time.time()
        if replay_mode == "eef_action":
            state = client.get_state()
            current_pose = np.asarray(state["state"][arm]["tcp_pose"], dtype=np.float32)
            eef_delta = np.asarray(transition["actions"]["eef_delta"], dtype=np.float32)
            pose_target = _target_pose_from_eef_delta(current_pose, eef_delta)
        elif replay_mode == "pose_target":
            pose_target = np.asarray(transition["next_observations"]["tcp_pose"], dtype=np.float32)
        else:
            raise ValueError(f"Unsupported replay_mode: {replay_mode}")

        client.post_action(
            {
                "mode": "ee_pose_servo",
                "owner": "policy",
                arm: {
                    "pose_target": pose_target.tolist(),
                    "preset": "free_space",
                },
            }
        )
        time.sleep(max(0.0, (1.0 / max(control_hz, 1e-6)) - (time.time() - start)))
        raw = client.get_state()
        actual_pose = np.asarray(raw["state"][arm]["tcp_pose"], dtype=np.float32)
        actual_pose_history.append(actual_pose)
        recorded_next = np.asarray(transition["next_observations"]["tcp_pose"], dtype=np.float32)
        pos_err = float(np.linalg.norm(actual_pose[:3] - recorded_next[:3]))
        ori_err = _quat_angle_error_rad(actual_pose[3:], recorded_next[3:])
        if idx % max(1, log_every) == 0 or idx == len(trajectory) - 1:
            extra = ""
            if replay_mode == "eef_action":
                extra = f" eef_delta_norm={float(np.linalg.norm(eef_delta)):.4f}"
            print(
                f"[eef-compare] step={idx + 1}/{len(trajectory)} "
                f"replay_mode={replay_mode} pos_err={pos_err:.4f}m ori_err={ori_err:.4f}rad{extra}"
            )

    print(f"Resetting robot after {replay_mode} replay...")
    client.reset(owner="debug")
    time.sleep(max(0.0, reset_wait_sec))
    return np.asarray(actual_pose_history, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Compare physical EEF-delta action replay and pose-target replay through body service HTTP.")
    parser.add_argument("--input_file", required=True, help="Path to EEF debug transition .pkl")
    parser.add_argument("--trajectory_index", type=int, default=0)
    parser.add_argument("--arm", default="right", choices=("left", "right"))
    parser.add_argument("--server_url", default="http://192.168.12.12:8001/")
    parser.add_argument("--control_hz", type=float, default=10.0)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--reset_wait_sec", type=float, default=1.0)
    parser.add_argument(
        "--output_image_3d",
        default="/home/robot/VLA-RL/conrft-r1lite/examples/debug/consistance/r1lite_debug_eef_replay_compare_3d.png",
    )
    parser.add_argument(
        "--output_csv",
        default="/home/robot/VLA-RL/conrft-r1lite/examples/debug/consistance/r1lite_debug_eef_replay_compare_errors.csv",
    )
    parser.add_argument(
        "--output_npz",
        default="/home/robot/VLA-RL/conrft-r1lite/examples/debug/consistance/r1lite_debug_eef_replay_compare.npz",
    )
    args = parser.parse_args()

    transitions = _load_transitions(Path(args.input_file).expanduser().resolve())
    trajectories = _split_trajectories(transitions)
    if args.trajectory_index < 0 or args.trajectory_index >= len(trajectories):
        raise IndexError(f"trajectory_index={args.trajectory_index} out of range for {len(trajectories)} trajectories")
    trajectory = trajectories[args.trajectory_index]

    client = R1LiteClient(args.server_url)
    try:
        eef_pose_arr = _run_replay(
            client=client,
            trajectory=trajectory,
            arm=args.arm,
            replay_mode="eef_action",
            control_hz=args.control_hz,
            log_every=args.log_every,
            reset_wait_sec=args.reset_wait_sec,
        )
        pose_pose_arr = _run_replay(
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

    recorded = np.asarray([np.asarray(t["next_observations"]["tcp_pose"], dtype=np.float32) for t in trajectory], dtype=np.float32)
    output_image_3d = Path(args.output_image_3d).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve()
    output_npz = Path(args.output_npz).expanduser().resolve()
    _plot_trajectories_3d(recorded[:, :3], eef_pose_arr[:, :3], pose_pose_arr[:, :3], output_image_3d)
    _write_csv(trajectory, eef_pose_arr, pose_pose_arr, output_csv)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_npz,
        recorded=recorded,
        eef_action_actual_pose=eef_pose_arr,
        pose_target_actual_pose=pose_pose_arr,
    )
    print(f"Saved 3D trajectory comparison image to {output_image_3d}")
    print(f"Saved per-step error csv to {output_csv}")
    print(f"Saved raw replay arrays to {output_npz}")


if __name__ == "__main__":
    main()
