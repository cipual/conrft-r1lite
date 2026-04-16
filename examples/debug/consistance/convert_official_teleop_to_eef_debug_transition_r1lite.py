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


def _build_eef_transitions(frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if len(frames) < 2:
        raise ValueError("Need at least two aligned frames to build transitions.")

    for idx, frame in enumerate(frames):
        if idx < len(frames) - 1:
            dt = max(1e-6, (frames[idx + 1]["timestamp_ns"] - frame["timestamp_ns"]) / 1e9)
            next_pose = frames[idx + 1]["tcp_pose"]
        else:
            dt = max(1e-6, (frame["timestamp_ns"] - frames[idx - 1]["timestamp_ns"]) / 1e9) if idx > 0 else 0.1
            next_pose = frame["tcp_pose"]
        frame["tcp_vel"] = _estimate_tcp_velocity(frame["tcp_pose"], next_pose, dt)

    transitions: List[Dict[str, Any]] = []
    for idx in range(len(frames) - 1):
        current = frames[idx]
        nxt = frames[idx + 1]
        eef_delta = np.concatenate(
            [
                np.asarray(nxt["tcp_pose"][:3], dtype=np.float32) - np.asarray(current["tcp_pose"][:3], dtype=np.float32),
                _rotvec_delta(np.asarray(current["tcp_pose"][3:], dtype=np.float32), np.asarray(nxt["tcp_pose"][3:], dtype=np.float32)),
            ],
            axis=0,
        ).astype(np.float32)
        done = bool(idx == len(frames) - 2)
        transitions.append(
            {
                "observations": _observation_from_frame(current),
                "actions": {
                    "eef_delta": eef_delta,
                },
                "next_observations": _observation_from_frame(nxt),
                "rewards": 0.0,
                "masks": float(1.0 - done),
                "dones": done,
                "infos": {
                    "conversion_source": "official_teleop_eef_debug_rosbags",
                    "action_type": "eef_delta_position_m_rotvec_rad",
                    "timestamp_ns": int(current["timestamp_ns"]),
                    "next_timestamp_ns": int(nxt["timestamp_ns"]),
                },
            }
        )
    return transitions


def main():
    parser = argparse.ArgumentParser(description="Convert official teleop RAW bag into a physical EEF-delta debug transition pkl.")
    parser.add_argument("--input_dir", required=True, help="Path to <episode>_RAW directory containing metadata.yaml and .mcap")
    parser.add_argument("--arm", default="right", choices=("left", "right"))
    parser.add_argument("--control_hz", type=float, default=10.0)
    parser.add_argument("--output_file", required=True, help="Where to write the EEF debug transition .pkl")
    args = parser.parse_args()

    frames = build_debug_frames(
        input_dir=Path(args.input_dir),
        arm=args.arm,
        control_hz=args.control_hz,
        require_arm_target=False,
    )
    transitions = _build_eef_transitions(frames)
    output_file = Path(args.output_file).expanduser().resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("wb") as f:
        pkl.dump(transitions, f)

    print(f"Built {len(transitions)} EEF debug transition(s)")
    print(f"Wrote EEF debug transition pkl to {output_file}")


if __name__ == "__main__":
    main()
