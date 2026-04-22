#!/usr/bin/env python3

import shutil
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml


@dataclass
class TopicSeries:
    timestamps: np.ndarray
    values: List[Any]


@dataclass(frozen=True)
class DebugTopicMapping:
    tcp_pose_topic: str
    arm_feedback_topic: str
    arm_target_topic: str
    gripper_feedback_topic: str


def topic_mapping_for_arm(arm: str) -> DebugTopicMapping:
    if arm == "right":
        return DebugTopicMapping(
            tcp_pose_topic="/motion_control/pose_ee_arm_right",
            arm_feedback_topic="/hdas/feedback_arm_right",
            arm_target_topic="/motion_target/target_joint_state_arm_right",
            gripper_feedback_topic="/hdas/feedback_gripper_right",
        )
    if arm == "left":
        return DebugTopicMapping(
            tcp_pose_topic="/motion_control/pose_ee_arm_left",
            arm_feedback_topic="/hdas/feedback_arm_left",
            arm_target_topic="/motion_target/target_joint_state_arm_left",
            gripper_feedback_topic="/hdas/feedback_gripper_left",
        )
    raise ValueError(f"Unsupported arm: {arm}")


def resolve_metadata_path(input_dir: Path) -> Path:
    metadata_path = input_dir / "metadata.yaml"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.yaml not found under {input_dir}")
    if not list(input_dir.glob("*.mcap")):
        raise FileNotFoundError(f"No .mcap file found under {input_dir}")
    return metadata_path


def load_metadata(metadata_path: Path) -> dict:
    with metadata_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def topic_counts(metadata: dict) -> Dict[str, int]:
    info = metadata["rosbag2_bagfile_information"]["topics_with_message_count"]
    return {item["topic_metadata"]["name"]: int(item["message_count"]) for item in info}


def build_uniform_timeline(start_ns: int, end_ns: int, hz: float) -> np.ndarray:
    step_ns = int(round(1e9 / hz))
    if step_ns <= 0:
        raise ValueError(f"Invalid control_hz: {hz}")
    return np.arange(start_ns, end_ns + 1, step_ns, dtype=np.int64)


def prepare_rosbags_input_dir(input_dir: Path, metadata: dict) -> Tuple[Path, Optional[tempfile.TemporaryDirectory]]:
    bag_info = metadata["rosbag2_bagfile_information"]
    relative_paths = [str(x) for x in bag_info.get("relative_file_paths", [])]
    if not relative_paths:
        return input_dir, None

    missing = [name for name in relative_paths if not (input_dir / name).exists()]
    if not missing:
        return input_dir, None

    actual_mcap_files = sorted(input_dir.glob("*.mcap"))
    if len(actual_mcap_files) != 1:
        missing_list = ", ".join(missing)
        raise FileNotFoundError(
            f"metadata.yaml expects missing MCAP files [{missing_list}] under {input_dir}, "
            "and automatic compatibility fallback only supports bags with exactly one actual .mcap file."
        )

    temp_dir = tempfile.TemporaryDirectory(prefix="r1lite_debug_rosbags_compat_", dir="/tmp")
    compat_dir = Path(temp_dir.name)
    shutil.copy2(input_dir / "metadata.yaml", compat_dir / "metadata.yaml")
    actual_mcap = actual_mcap_files[0]
    for relative_name in relative_paths:
        target = compat_dir / relative_name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(actual_mcap)
    return compat_dir, temp_dir


def load_rosbags_reader():
    try:
        from rosbags.highlevel import AnyReader
    except ImportError as exc:
        raise RuntimeError("rosbags is required. Install it with: python -m pip install rosbags") from exc
    return AnyReader


def extract_pose_stamped(msg) -> np.ndarray:
    pose = msg.pose
    return np.asarray(
        [
            pose.position.x,
            pose.position.y,
            pose.position.z,
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ],
        dtype=np.float32,
    )


def extract_joint_state(msg, default_len: int = 6) -> Dict[str, np.ndarray]:
    position = np.asarray(list(getattr(msg, "position", [])), dtype=np.float32)
    velocity = np.asarray(list(getattr(msg, "velocity", [])), dtype=np.float32)
    effort = np.asarray(list(getattr(msg, "effort", [])), dtype=np.float32)

    def fit(vec: np.ndarray) -> np.ndarray:
        if vec.size >= default_len:
            return vec[:default_len].astype(np.float32)
        out = np.zeros((default_len,), dtype=np.float32)
        if vec.size > 0:
            out[: vec.size] = vec.astype(np.float32)
        return out

    return {
        "position": fit(position),
        "velocity": fit(velocity),
        "effort": fit(effort),
        "has_velocity": bool(velocity.size >= default_len),
        "has_effort": bool(effort.size >= default_len),
    }


def extract_gripper_joint_state(msg) -> Dict[str, float]:
    position = list(getattr(msg, "position", []))
    velocity = list(getattr(msg, "velocity", []))
    effort = list(getattr(msg, "effort", []))
    return {
        "position": float(position[0]) if position else 0.0,
        "velocity": float(velocity[0]) if velocity else 0.0,
        "effort": float(effort[0]) if effort else 0.0,
    }


def parse_message(topic: str, msg) -> Any:
    if "pose_ee_arm_" in topic or "target_pose_arm_" in topic:
        return extract_pose_stamped(msg)
    if "feedback_arm_" in topic or "target_joint_state_arm_" in topic:
        return extract_joint_state(msg, default_len=6)
    if "gripper" in topic or "feedback_hand_" in topic:
        return extract_gripper_joint_state(msg)
    return msg


def read_topic_series(input_dir: Path, metadata: dict, topics: Sequence[str]) -> Dict[str, TopicSeries]:
    AnyReader = load_rosbags_reader()
    topic_set = set(topics)
    buckets: Dict[str, List[Tuple[int, Any]]] = defaultdict(list)
    reader_input_dir, temp_dir = prepare_rosbags_input_dir(input_dir, metadata)
    try:
        with AnyReader([reader_input_dir]) as reader:
            connections = [conn for conn in reader.connections if conn.topic in topic_set]
            for connection, timestamp, rawdata in reader.messages(connections=connections):
                msg = reader.deserialize(rawdata, connection.msgtype)
                buckets[connection.topic].append((int(timestamp), parse_message(connection.topic, msg)))
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()

    result = {}
    for topic in topics:
        samples = buckets.get(topic, [])
        if not samples:
            continue
        result[topic] = TopicSeries(
            timestamps=np.asarray([item[0] for item in samples], dtype=np.int64),
            values=[item[1] for item in samples],
        )
    return result


def series_value_at(series: TopicSeries, timestamp_ns: int) -> Any:
    idx = int(np.searchsorted(series.timestamps, timestamp_ns, side="right") - 1)
    if idx < 0:
        idx = 0
    return series.values[idx]


def effective_interval(series_map: Dict[str, TopicSeries], topic_names: Sequence[str]) -> Tuple[int, int]:
    starts = []
    ends = []
    for topic in topic_names:
        series = series_map.get(topic)
        if series is None or len(series.timestamps) == 0:
            raise ValueError(f"Series is missing for topic: {topic}")
        starts.append(int(series.timestamps[0]))
        ends.append(int(series.timestamps[-1]))
    start_ns = max(starts)
    end_ns = min(ends)
    if end_ns <= start_ns:
        raise ValueError("No overlapping time interval across required topics")
    return start_ns, end_ns


def estimate_joint_velocity(current_joint: np.ndarray, next_joint: np.ndarray, dt: float) -> np.ndarray:
    if dt <= 0:
        return np.zeros_like(current_joint, dtype=np.float32)
    return ((next_joint - current_joint) / dt).astype(np.float32)


def estimate_tcp_velocity(current_pose: np.ndarray, next_pose: np.ndarray, rotvec_delta_fn, dt: float) -> np.ndarray:
    if dt <= 0:
        return np.zeros((6,), dtype=np.float32)
    linear = (next_pose[:3] - current_pose[:3]) / dt
    angular = rotvec_delta_fn(current_pose[3:], next_pose[3:]) / dt
    return np.concatenate([linear, angular], axis=0).astype(np.float32)


def build_debug_frames(input_dir: Path, arm: str, control_hz: float, require_arm_target: bool = False) -> List[Dict[str, Any]]:
    input_dir = input_dir.expanduser().resolve()
    metadata_path = resolve_metadata_path(input_dir)
    metadata = load_metadata(metadata_path)
    counts = topic_counts(metadata)
    mapping = topic_mapping_for_arm(arm)

    required_topics = [mapping.tcp_pose_topic, mapping.arm_feedback_topic]
    if require_arm_target:
        required_topics.append(mapping.arm_target_topic)
    for topic in required_topics:
        if counts.get(topic, 0) <= 0:
            raise ValueError(f"Required topic missing or empty: {topic}")

    selected_topics = [mapping.tcp_pose_topic, mapping.arm_feedback_topic]
    if counts.get(mapping.gripper_feedback_topic, 0) > 0:
        selected_topics.append(mapping.gripper_feedback_topic)
    if counts.get(mapping.arm_target_topic, 0) > 0:
        selected_topics.append(mapping.arm_target_topic)

    series_map = read_topic_series(input_dir, metadata, selected_topics)
    start_ns, end_ns = effective_interval(series_map, required_topics)
    timeline_ns = build_uniform_timeline(start_ns, end_ns, control_hz)

    frames: List[Dict[str, Any]] = []
    for timestamp_ns in timeline_ns:
        pose = series_value_at(series_map[mapping.tcp_pose_topic], int(timestamp_ns))
        joint_fb = series_value_at(series_map[mapping.arm_feedback_topic], int(timestamp_ns))
        gripper_fb = None
        if mapping.gripper_feedback_topic in series_map:
            gripper_fb = series_value_at(series_map[mapping.gripper_feedback_topic], int(timestamp_ns))

        frame = {
            "timestamp_ns": int(timestamp_ns),
            "tcp_pose": np.asarray(pose, dtype=np.float32),
            "joint_pos": np.asarray(joint_fb["position"], dtype=np.float32),
            "joint_vel_raw": np.asarray(joint_fb["velocity"], dtype=np.float32),
            "joint_has_velocity": bool(joint_fb["has_velocity"]),
            "joint_effort_raw": np.asarray(joint_fb["effort"], dtype=np.float32),
            "joint_has_effort": bool(joint_fb["has_effort"]),
            "gripper_pose": np.asarray(
                [gripper_fb["position"] if gripper_fb is not None else 0.0], dtype=np.float32
            ),
        }
        if mapping.arm_target_topic in series_map:
            joint_target = series_value_at(series_map[mapping.arm_target_topic], int(timestamp_ns))
            frame["joint_target_pos"] = np.asarray(joint_target["position"], dtype=np.float32)
            frame["joint_target_vel"] = np.asarray(joint_target["velocity"], dtype=np.float32)
        frames.append(frame)

    for idx, frame in enumerate(frames):
        if idx < len(frames) - 1:
            dt = max(1e-6, (frames[idx + 1]["timestamp_ns"] - frame["timestamp_ns"]) / 1e9)
            next_frame = frames[idx + 1]
        else:
            dt = max(1e-6, (frame["timestamp_ns"] - frames[idx - 1]["timestamp_ns"]) / 1e9) if idx > 0 else 1.0 / control_hz
            next_frame = frame
        if frame["joint_has_velocity"]:
            frame["joint_vel"] = frame["joint_vel_raw"].astype(np.float32)
        else:
            frame["joint_vel"] = estimate_joint_velocity(frame["joint_pos"], next_frame["joint_pos"], dt)
        if frame["joint_has_effort"]:
            frame["joint_effort"] = frame["joint_effort_raw"].astype(np.float32)
        else:
            frame["joint_effort"] = np.zeros((6,), dtype=np.float32)
    return frames
