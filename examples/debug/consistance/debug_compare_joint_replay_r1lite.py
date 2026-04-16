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
    for path in (
        root,
        repo_root / "serl_robot_infra",
    ):
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


def _plot_trajectories_3d(recorded: np.ndarray, joint_actual: np.ndarray, pose_actual: np.ndarray, output_path: Path):
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(recorded[:, 0], recorded[:, 1], recorded[:, 2], label="recorded", color="#222222", linewidth=2.2)
    ax.plot(joint_actual[:, 0], joint_actual[:, 1], joint_actual[:, 2], label="joint action replay", color="#1f77b4", linewidth=1.8)
    ax.plot(pose_actual[:, 0], pose_actual[:, 1], pose_actual[:, 2], label="pose_target replay", color="#d62728", linewidth=1.8)
    ax.scatter(recorded[0, 0], recorded[0, 1], recorded[0, 2], color="#222222", marker="o", s=28)
    ax.scatter(recorded[-1, 0], recorded[-1, 1], recorded[-1, 2], color="#222222", marker="x", s=40)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title("Joint-Delta Action vs Pose Target Replay Through Body Service (3D)")
    ax.legend(frameon=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _write_csv(
    trajectory: List[Dict],
    joint_pose_history: np.ndarray,
    joint_joint_history: np.ndarray,
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
                "joint_action_x",
                "joint_action_y",
                "joint_action_z",
                "joint_action_qx",
                "joint_action_qy",
                "joint_action_qz",
                "joint_action_qw",
                "joint_action_pos_err_m",
                "joint_action_ori_err_rad",
                "pose_target_x",
                "pose_target_y",
                "pose_target_z",
                "pose_target_qx",
                "pose_target_qy",
                "pose_target_qz",
                "pose_target_qw",
                "pose_target_pos_err_m",
                "pose_target_ori_err_rad",
                "joint_delta_1",
                "joint_delta_2",
                "joint_delta_3",
                "joint_delta_4",
                "joint_delta_5",
                "joint_delta_6",
                "target_joint_1",
                "target_joint_2",
                "target_joint_3",
                "target_joint_4",
                "target_joint_5",
                "target_joint_6",
                "joint_action_joint_1",
                "joint_action_joint_2",
                "joint_action_joint_3",
                "joint_action_joint_4",
                "joint_action_joint_5",
                "joint_action_joint_6",
            ]
        )
        for idx, transition in enumerate(trajectory):
            recorded = np.asarray(transition["next_observations"]["tcp_pose"], dtype=np.float32)
            joint_actual = np.asarray(joint_pose_history[idx], dtype=np.float32)
            pose_actual = np.asarray(pose_pose_history[idx], dtype=np.float32)
            joint_delta = np.asarray(
                transition["actions"].get(
                    "joint_delta",
                    np.asarray(transition["next_observations"]["joint_pos"], dtype=np.float32)
                    - np.asarray(transition["observations"]["joint_pos"], dtype=np.float32),
                ),
                dtype=np.float32,
            )
            target_joint = (
                np.asarray(transition["observations"]["joint_pos"], dtype=np.float32) + joint_delta
            ).astype(np.float32)
            actual_joint = np.asarray(joint_joint_history[idx], dtype=np.float32)
            writer.writerow(
                [
                    idx + 1,
                    *recorded.tolist(),
                    *joint_actual.tolist(),
                    float(np.linalg.norm(joint_actual[:3] - recorded[:3])),
                    _quat_angle_error_rad(joint_actual[3:], recorded[3:]),
                    *pose_actual.tolist(),
                    float(np.linalg.norm(pose_actual[:3] - recorded[:3])),
                    _quat_angle_error_rad(pose_actual[3:], recorded[3:]),
                    *joint_delta.tolist(),
                    *target_joint.tolist(),
                    *actual_joint.tolist(),
                ]
            )


def _run_replay(
    client: R1LiteClient,
    trajectory: List[Dict],
    arm: str,
    server_mode: str,
    replay_mode: str,
    control_hz: float,
    log_every: int,
    reset_wait_sec: float,
) -> Tuple[np.ndarray, np.ndarray]:
    actual_pose_history: List[np.ndarray] = []
    actual_joint_history: List[np.ndarray] = []

    print(f"Resetting robot before {replay_mode} replay...")
    client.reset(owner="debug")
    time.sleep(max(0.0, reset_wait_sec))
    print(f"Replaying {len(trajectory)} step(s) in {replay_mode} mode through body service")

    for idx, transition in enumerate(trajectory):
        start = time.time()
        if replay_mode == "joint_action":
            state = client.get_state()
            current_joint = np.asarray(state["state"][arm]["joint_pos"], dtype=np.float32)
            joint_delta = np.asarray(
                transition["actions"].get(
                    "joint_delta",
                    np.asarray(transition["next_observations"]["joint_pos"], dtype=np.float32)
                    - np.asarray(transition["observations"]["joint_pos"], dtype=np.float32),
                ),
                dtype=np.float32,
            )
            joint_target = (current_joint + joint_delta).astype(np.float32)
            payload = {
                "mode": server_mode,
                "owner": "policy",
                arm: {
                    "joint_target": joint_target.tolist(),
                    "preset": "free_space",
                },
            }
        elif replay_mode == "pose_target":
            payload = {
                "mode": server_mode,
                "owner": "policy",
                arm: {
                    "pose_target": np.asarray(transition["next_observations"]["tcp_pose"], dtype=np.float32).tolist(),
                    "preset": "free_space",
                },
            }
        else:
            raise ValueError(f"Unsupported replay_mode: {replay_mode}")

        client.post_action(payload)
        time.sleep(max(0.0, (1.0 / max(control_hz, 1e-6)) - (time.time() - start)))
        raw = client.get_state()
        actual_pose = np.asarray(raw["state"][arm]["tcp_pose"], dtype=np.float32)
        actual_joint = np.asarray(raw["state"][arm]["joint_pos"], dtype=np.float32)
        actual_pose_history.append(actual_pose)
        actual_joint_history.append(actual_joint)
        recorded_next = np.asarray(transition["next_observations"]["tcp_pose"], dtype=np.float32)
        pos_err = float(np.linalg.norm(actual_pose[:3] - recorded_next[:3]))
        ori_err = _quat_angle_error_rad(actual_pose[3:], recorded_next[3:])
        if idx % max(1, log_every) == 0 or idx == len(trajectory) - 1:
            extra = ""
            if replay_mode == "joint_action":
                extra = f" joint_delta_norm={float(np.linalg.norm(joint_delta)):.4f}"
            print(
                f"[joint-compare] step={idx + 1}/{len(trajectory)} "
                f"replay_mode={replay_mode} "
                f"mode={server_mode} "
                f"pos_err={pos_err:.4f}m ori_err={ori_err:.4f}rad{extra}"
            )

    print(f"Resetting robot after {replay_mode} replay...")
    client.reset(owner="debug")
    time.sleep(max(0.0, reset_wait_sec))
    return np.asarray(actual_pose_history, dtype=np.float32), np.asarray(actual_joint_history, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Compare physical joint-delta action replay and pose-target replay through body service HTTP.")
    parser.add_argument("--input_file", required=True, help="Path to joint transition .pkl")
    parser.add_argument("--trajectory_index", type=int, default=0)
    parser.add_argument("--arm", default="right", choices=("left", "right"))
    parser.add_argument("--server_url", default="http://192.168.12.12:8001/")
    parser.add_argument("--control_hz", type=float, default=10.0)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--reset_wait_sec", type=float, default=1.0)
    parser.add_argument("--mode", default="ee_pose_servo", choices=("ee_pose_servo", "mit_joint_compliance"))
    parser.add_argument(
        "--output_image_3d",
        default="/home/robot/VLA-RL/conrft-r1lite/examples/debug/consistance/r1lite_debug_joint_replay_compare_3d.png",
    )
    parser.add_argument(
        "--output_csv",
        default="/home/robot/VLA-RL/conrft-r1lite/examples/debug/consistance/r1lite_debug_joint_replay_compare_errors.csv",
    )
    parser.add_argument(
        "--output_npz",
        default="/home/robot/VLA-RL/conrft-r1lite/examples/debug/consistance/r1lite_debug_joint_replay_compare.npz",
    )
    args = parser.parse_args()

    input_file = Path(args.input_file).expanduser().resolve()
    transitions = _load_transitions(input_file)
    trajectories = _split_trajectories(transitions)
    if args.trajectory_index < 0 or args.trajectory_index >= len(trajectories):
        raise IndexError(f"trajectory_index={args.trajectory_index} out of range for {len(trajectories)} trajectories")
    trajectory = trajectories[args.trajectory_index]

    client = R1LiteClient(args.server_url)
    try:
        joint_pose_arr, joint_joint_arr = _run_replay(
            client=client,
            trajectory=trajectory,
            arm=args.arm,
            server_mode=args.mode,
            replay_mode="joint_action",
            control_hz=args.control_hz,
            log_every=args.log_every,
            reset_wait_sec=args.reset_wait_sec,
        )
        pose_pose_arr, _ = _run_replay(
            client=client,
            trajectory=trajectory,
            arm=args.arm,
            server_mode=args.mode,
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
    _plot_trajectories_3d(recorded[:, :3], joint_pose_arr[:, :3], pose_pose_arr[:, :3], output_image_3d)
    _write_csv(trajectory, joint_pose_arr, joint_joint_arr, pose_pose_arr, output_csv)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_npz,
        recorded=recorded,
        joint_action_actual_pose=joint_pose_arr,
        joint_action_actual_joint=joint_joint_arr,
        pose_target_actual_pose=pose_pose_arr,
    )
    print(f"Saved 3D trajectory comparison image to {output_image_3d}")
    print(f"Saved per-step error csv to {output_csv}")
    print(f"Saved raw replay arrays to {output_npz}")


if __name__ == "__main__":
    main()
