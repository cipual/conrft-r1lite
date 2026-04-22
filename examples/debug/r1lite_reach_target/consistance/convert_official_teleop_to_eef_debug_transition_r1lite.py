#!/usr/bin/env python3

import argparse
import pickle as pkl
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from scipy.spatial.transform import Rotation

from debug_rosbag_utils import build_debug_frames


def _rotvec_delta(current_xyzw: np.ndarray, next_xyzw: np.ndarray) -> np.ndarray:
    current = Rotation.from_quat(current_xyzw)
    nxt = Rotation.from_quat(next_xyzw)
    return (current.inv() * nxt).as_rotvec().astype(np.float32)


def _estimate_tcp_velocity(current_pose: np.ndarray, next_pose: np.ndarray, dt: float) -> np.ndarray:
    if dt <= 0:
        return np.zeros((6,), dtype=np.float32)
    linear = (next_pose[:3] - current_pose[:3]) / dt
    angular = _rotvec_delta(current_pose[3:], next_pose[3:]) / dt
    return np.concatenate([linear, angular], axis=0).astype(np.float32)


def _observation_from_frame(frame: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "timestamp_ns": int(frame["timestamp_ns"]),
        "tcp_pose": np.asarray(frame["tcp_pose"], dtype=np.float32).copy(),
        "tcp_vel": np.asarray(frame["tcp_vel"], dtype=np.float32).copy(),
        "joint_pos": np.asarray(frame["joint_pos"], dtype=np.float32).copy(),
        "joint_vel": np.asarray(frame["joint_vel"], dtype=np.float32).copy(),
        "gripper_pose": np.asarray(frame["gripper_pose"], dtype=np.float32).copy(),
    }


def _eef_action_from_frames(current: Dict[str, Any], nxt: Dict[str, Any]) -> Dict[str, np.ndarray]:
    eef_delta = np.concatenate(
        [
            np.asarray(nxt["tcp_pose"][:3], dtype=np.float32) - np.asarray(current["tcp_pose"][:3], dtype=np.float32),
            _rotvec_delta(np.asarray(current["tcp_pose"][3:], dtype=np.float32), np.asarray(nxt["tcp_pose"][3:], dtype=np.float32)),
        ],
        axis=0,
    ).astype(np.float32)
    gripper_delta = (
        np.asarray(nxt["gripper_pose"], dtype=np.float32) - np.asarray(current["gripper_pose"], dtype=np.float32)
    ).astype(np.float32)
    return {
        "eef_delta": eef_delta,
        "gripper_delta": gripper_delta,
        "gripper_target": np.asarray(nxt["gripper_pose"], dtype=np.float32).copy(),
    }


def _fill_tcp_velocities(frames: List[Dict[str, Any]]) -> None:
    for idx, frame in enumerate(frames):
        if idx < len(frames) - 1:
            dt = max(1e-6, (frames[idx + 1]["timestamp_ns"] - frame["timestamp_ns"]) / 1e9)
            next_pose = frames[idx + 1]["tcp_pose"]
        else:
            dt = max(1e-6, (frame["timestamp_ns"] - frames[idx - 1]["timestamp_ns"]) / 1e9) if idx > 0 else 0.1
            next_pose = frame["tcp_pose"]
        frame["tcp_vel"] = _estimate_tcp_velocity(frame["tcp_pose"], next_pose, dt)


def _build_eef_transitions(frames: List[Dict[str, Any]], arm: str) -> List[Dict[str, Any]]:
    if len(frames) < 2:
        raise ValueError("Need at least two aligned frames to build transitions.")

    _fill_tcp_velocities(frames)

    transitions: List[Dict[str, Any]] = []
    for idx in range(len(frames) - 1):
        current = frames[idx]
        nxt = frames[idx + 1]
        done = bool(idx == len(frames) - 2)
        transitions.append(
            {
                "observations": _observation_from_frame(current),
                "actions": _eef_action_from_frames(current, nxt),
                "next_observations": _observation_from_frame(nxt),
                "rewards": 0.0,
                "masks": float(1.0 - done),
                "dones": done,
                "infos": {
                    "conversion_source": "official_teleop_eef_debug_rosbags",
                    "action_type": "eef_delta_position_m_rotvec_rad_gripper_delta",
                    "arm": arm,
                    "timestamp_ns": int(current["timestamp_ns"]),
                    "next_timestamp_ns": int(nxt["timestamp_ns"]),
                },
            }
        )
    return transitions


def _align_dual_frames(left_frames: List[Dict[str, Any]], right_frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    left_ts = np.asarray([int(frame["timestamp_ns"]) for frame in left_frames], dtype=np.int64)
    right_ts = np.asarray([int(frame["timestamp_ns"]) for frame in right_frames], dtype=np.int64)
    start_ns = max(int(left_ts[0]), int(right_ts[0]))
    end_ns = min(int(left_ts[-1]), int(right_ts[-1]))
    if end_ns <= start_ns:
        raise ValueError("No overlapping left/right time interval for dual-arm transitions.")

    aligned: List[Dict[str, Any]] = []
    right_idx = 0
    for left_idx, ts in enumerate(left_ts):
        if ts < start_ns or ts > end_ns:
            continue
        while right_idx + 1 < len(right_ts) and abs(int(right_ts[right_idx + 1]) - int(ts)) <= abs(int(right_ts[right_idx]) - int(ts)):
            right_idx += 1
        if right_ts[right_idx] < start_ns or right_ts[right_idx] > end_ns:
            continue
        aligned.append(
            {
                "timestamp_ns": int(ts),
                "left": left_frames[left_idx],
                "right": right_frames[right_idx],
            }
        )
    if len(aligned) < 2:
        raise ValueError("Need at least two aligned left/right samples for dual-arm transitions.")
    return aligned


def _build_dual_eef_transitions(frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for arm in ("left", "right"):
        _fill_tcp_velocities([frame[arm] for frame in frames])

    transitions: List[Dict[str, Any]] = []
    for idx in range(len(frames) - 1):
        current = frames[idx]
        nxt = frames[idx + 1]
        done = bool(idx == len(frames) - 2)
        transitions.append(
            {
                "observations": {
                    "left": _observation_from_frame(current["left"]),
                    "right": _observation_from_frame(current["right"]),
                },
                "actions": {
                    "left": _eef_action_from_frames(current["left"], nxt["left"]),
                    "right": _eef_action_from_frames(current["right"], nxt["right"]),
                },
                "next_observations": {
                    "left": _observation_from_frame(nxt["left"]),
                    "right": _observation_from_frame(nxt["right"]),
                },
                "rewards": 0.0,
                "masks": float(1.0 - done),
                "dones": done,
                "infos": {
                    "conversion_source": "official_teleop_dual_eef_debug_rosbags",
                    "action_type": "dual_eef_delta_position_m_rotvec_rad_gripper_delta",
                    "arm": "dual",
                    "timestamp_ns": int(current["timestamp_ns"]),
                    "next_timestamp_ns": int(nxt["timestamp_ns"]),
                },
            }
        )
    return transitions


def main():
    parser = argparse.ArgumentParser(description="Convert official teleop RAW bag into a physical EEF-delta debug transition pkl.")
    parser.add_argument("--input_dir", required=True, help="Path to <episode>_RAW directory containing metadata.yaml and .mcap")
    parser.add_argument("--arm", default="right", choices=("left", "right", "dual"))
    parser.add_argument("--control_hz", type=float, default=10.0)
    parser.add_argument("--output_file", required=True, help="Where to write the EEF debug transition .pkl")
    args = parser.parse_args()

    if args.arm == "dual":
        left_frames = build_debug_frames(Path(args.input_dir), "left", args.control_hz, require_arm_target=False)
        right_frames = build_debug_frames(Path(args.input_dir), "right", args.control_hz, require_arm_target=False)
        transitions = _build_dual_eef_transitions(_align_dual_frames(left_frames, right_frames))
    else:
        frames = build_debug_frames(
            input_dir=Path(args.input_dir),
            arm=args.arm,
            control_hz=args.control_hz,
            require_arm_target=False,
        )
        transitions = _build_eef_transitions(frames, args.arm)
    output_file = Path(args.output_file).expanduser().resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("wb") as f:
        pkl.dump(transitions, f)

    print(f"Built {len(transitions)} EEF debug transition(s)")
    print(f"Wrote EEF debug transition pkl to {output_file}")


if __name__ == "__main__":
    main()
