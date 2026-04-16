#!/usr/bin/env python3

import argparse
import pickle as pkl
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from debug_rosbag_utils import build_debug_frames


def _observation_from_frame(frame: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "timestamp_ns": int(frame["timestamp_ns"]),
        "tcp_pose": np.asarray(frame["tcp_pose"], dtype=np.float32).copy(),
        "joint_pos": np.asarray(frame["joint_pos"], dtype=np.float32).copy(),
        "joint_vel": np.asarray(frame["joint_vel"], dtype=np.float32).copy(),
        "gripper_pose": np.asarray(frame["gripper_pose"], dtype=np.float32).copy(),
    }


def _build_joint_transitions(frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if len(frames) < 2:
        raise ValueError("Need at least two aligned frames to build transitions.")

    transitions: List[Dict[str, Any]] = []
    for idx in range(len(frames) - 1):
        current = frames[idx]
        nxt = frames[idx + 1]
        action: Dict[str, np.ndarray] = {
            "joint_delta": (
                np.asarray(nxt["joint_pos"], dtype=np.float32)
                - np.asarray(current["joint_pos"], dtype=np.float32)
            ).astype(np.float32),
        }
        if "joint_target_pos" in current:
            action["joint_target_position"] = np.asarray(current["joint_target_pos"], dtype=np.float32).copy()
            action["joint_target_velocity"] = np.asarray(current["joint_target_vel"], dtype=np.float32).copy()

        done = bool(idx == len(frames) - 2)
        transitions.append(
            {
                "observations": _observation_from_frame(current),
                "actions": action,
                "next_observations": _observation_from_frame(nxt),
                "rewards": 0.0,
                "masks": float(1.0 - done),
                "dones": done,
                "infos": {
                    "conversion_source": "official_teleop_joint_debug_rosbags",
                    "action_type": "joint_delta_rad",
                    "timestamp_ns": int(current["timestamp_ns"]),
                    "next_timestamp_ns": int(nxt["timestamp_ns"]),
                },
            }
        )
    return transitions


def main():
    parser = argparse.ArgumentParser(description="Convert official teleop RAW bag into a physical joint-delta debug transition pkl.")
    parser.add_argument("--input_dir", required=True, help="Path to <episode>_RAW directory containing metadata.yaml and .mcap")
    parser.add_argument("--arm", default="right", choices=("left", "right"))
    parser.add_argument("--control_hz", type=float, default=10.0)
    parser.add_argument("--output_file", required=True, help="Where to write the joint debug transition .pkl")
    args = parser.parse_args()

    frames = build_debug_frames(
        input_dir=Path(args.input_dir),
        arm=args.arm,
        control_hz=args.control_hz,
        require_arm_target=False,
    )
    transitions = _build_joint_transitions(frames)
    output_file = Path(args.output_file).expanduser().resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("wb") as f:
        pkl.dump(transitions, f)

    print(f"Built {len(transitions)} joint debug transition(s)")
    print(f"Wrote joint debug transition pkl to {output_file}")


if __name__ == "__main__":
    main()
