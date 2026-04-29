#!/usr/bin/env python3

import argparse
import json
import os
import pickle as pkl
import shutil
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import gymnasium as gym
import numpy as np
import yaml
from scipy.spatial.transform import Rotation


def _ensure_examples_on_path():
    """允许独立脚本直接导入 experiments / r1lite_env / serl_launcher。"""
    root = Path(__file__).resolve().parent
    repo_root = root.parent
    for path in (
        root,
        repo_root / "serl_robot_infra",
        repo_root / "serl_launcher",
    ):
        path_str = str(path)
        if path_str not in os.sys.path:
            os.sys.path.insert(0, path_str)


_ensure_examples_on_path()

from data_util import (  # noqa: E402
    add_embeddings_to_trajectory,
    add_mc_returns_to_trajectory,
    add_next_embeddings_to_trajectory,
)
from experiments.mappings import CONFIG_MAPPING  # noqa: E402
from experiments.r1lite_reach_target.wrapper import (  # noqa: E402
    ReachTargetTaskConfig,
    quat_angle_error_rad,
)


@dataclass
class ConversionConfig:
    exp_name: str
    arm: str
    image_keys: Tuple[str, ...]
    proprio_keys: Tuple[str, ...]
    control_hz: float
    action_xyz_scale: float
    action_rot_scale: float
    setup_mode: str
    reward_neg: float
    discount: float
    uses_gripper: bool
    task_config: Optional[ReachTargetTaskConfig]
    obs_horizon: int
    task_desc: str
    octo_path: str


@dataclass
class TopicMapping:
    primary_image_topics: Tuple[str, ...]
    wrist_image_topics: Tuple[str, ...]
    ee_pose_topic: str
    arm_feedback_topic: str
    gripper_feedback_topic: Optional[str]
    arm_target_topic: Optional[str]
    gripper_target_topic: Optional[str]


@dataclass
class TopicSeries:
    timestamps: np.ndarray
    values: List[Any]


_PROPRIO_FIELD_SHAPES = {
    "tcp_pose": (7,),
    "tcp_vel": (6,),
    "tcp_force": (3,),
    "tcp_torque": (3,),
    "gripper_pose": (1,),
    "joint_pos": (6,),
    "joint_vel": (6,),
    "joint_effort": (6,),
}


def _load_task_config(exp_name: str) -> ConversionConfig:
    if exp_name not in CONFIG_MAPPING:
        raise ValueError(f"Unknown experiment: {exp_name}")
    cfg = CONFIG_MAPPING[exp_name]()
    env_cfg = cfg.get_environment(fake_env=True).unwrapped.config
    task_cfg = getattr(cfg, "task_config", None)
    return ConversionConfig(
        exp_name=exp_name,
        arm=cfg.arm,
        image_keys=tuple(cfg.image_keys),
        proprio_keys=tuple(sorted(getattr(cfg, "proprio_keys", []))),
        control_hz=float(getattr(env_cfg, "CONTROL_HZ", 10.0)),
        action_xyz_scale=float(np.asarray(getattr(env_cfg, "ACTION_SCALE"))[0]),
        action_rot_scale=float(np.asarray(getattr(env_cfg, "ACTION_SCALE"))[1]),
        setup_mode=str(cfg.setup_mode),
        reward_neg=float(getattr(cfg, "reward_neg", 0.0)),
        discount=float(getattr(cfg, "discount", 0.98)),
        uses_gripper="learned-gripper" in str(cfg.setup_mode),
        task_config=task_cfg,
        obs_horizon=2,
        task_desc=str(getattr(cfg, "task_desc", "")),
        octo_path=str(getattr(cfg, "octo_path", "")),
    )


def _resolve_bag_paths(input_dir: Path, meta_json: Optional[Path]) -> Tuple[Path, Path, Path]:
    metadata_path = input_dir / "metadata.yaml"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.yaml not found under {input_dir}")

    mcap_files = sorted(input_dir.glob("*.mcap"))
    if not mcap_files:
        raise FileNotFoundError(f"No .mcap file found under {input_dir}")
    mcap_path = mcap_files[0]

    if meta_json is None:
        candidate = input_dir.parent / f"{input_dir.name}.json"
        if not candidate.exists():
            raise FileNotFoundError(
                f"Could not infer RAW.json next to {input_dir}. Please pass --meta-json explicitly."
            )
        meta_json = candidate

    return metadata_path, mcap_path, meta_json


def _load_metadata(metadata_path: Path, meta_json: Path) -> Tuple[dict, dict]:
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = yaml.safe_load(f)
    with meta_json.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    return metadata, meta


def _prepare_rosbags_input_dir(input_dir: Path, metadata: dict) -> Tuple[Path, Optional[tempfile.TemporaryDirectory]]:
    """
    某些官方导出的 bag 目录里实际只有 `RAW.mcap`，但 metadata 里记录的是 `RAW_0.mcap`。
    rosbags 会严格按 metadata 找文件，这里自动构造一个兼容目录，避免用户手工改包。
    """
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

    temp_dir = tempfile.TemporaryDirectory(prefix="r1lite_rosbags_compat_", dir="/tmp")
    compat_dir = Path(temp_dir.name)
    shutil.copy2(input_dir / "metadata.yaml", compat_dir / "metadata.yaml")
    actual_mcap = actual_mcap_files[0]
    for relative_name in relative_paths:
        target = compat_dir / relative_name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(actual_mcap)
    return compat_dir, temp_dir


def _topic_counts(metadata: dict) -> Dict[str, int]:
    info = metadata["rosbag2_bagfile_information"]["topics_with_message_count"]
    return {item["topic_metadata"]["name"]: int(item["message_count"]) for item in info}


def _build_topic_mapping(cfg: ConversionConfig) -> TopicMapping:
    if cfg.arm == "right":
        return TopicMapping(
            primary_image_topics=(
                "/hdas/camera_head/right_raw/image_raw_color/compressed",
                "/hdas/camera_head/left_raw/image_raw_color/compressed",
            ),
            wrist_image_topics=("/hdas/camera_wrist_right/color/image_raw/compressed",),
            ee_pose_topic="/motion_control/pose_ee_arm_right",
            arm_feedback_topic="/hdas/feedback_arm_right",
            gripper_feedback_topic="/hdas/feedback_gripper_right",
            arm_target_topic="/motion_target/target_joint_state_arm_right",
            gripper_target_topic="/motion_target/target_position_gripper_right",
        )
    if cfg.arm == "left":
        return TopicMapping(
            primary_image_topics=(
                "/hdas/camera_head/left_raw/image_raw_color/compressed",
                "/hdas/camera_head/right_raw/image_raw_color/compressed",
            ),
            wrist_image_topics=("/hdas/camera_wrist_left/color/image_raw/compressed",),
            ee_pose_topic="/motion_control/pose_ee_arm_left",
            arm_feedback_topic="/hdas/feedback_arm_left",
            gripper_feedback_topic="/hdas/feedback_gripper_left",
            arm_target_topic="/motion_target/target_joint_state_arm_left",
            gripper_target_topic="/motion_target/target_position_gripper_left",
        )
    raise NotImplementedError("Dual-arm official teleop conversion is not implemented yet.")


def _pick_available_topic(candidates: Sequence[str], counts: Dict[str, int]) -> Optional[str]:
    for name in candidates:
        if counts.get(name, 0) > 0:
            return name
    return None


def _validate_required_topics(mapping: TopicMapping, counts: Dict[str, int]) -> Dict[str, str]:
    resolved = {}
    primary = _pick_available_topic(mapping.primary_image_topics, counts)
    wrist = _pick_available_topic(mapping.wrist_image_topics, counts)
    if primary is None:
        raise ValueError("No available head camera topic found for image_primary")
    if wrist is None:
        raise ValueError("No available wrist camera topic found for image_wrist")

    required = {
        "image_primary": primary,
        "image_wrist": wrist,
        "tcp_pose": mapping.ee_pose_topic,
        "joint_feedback": mapping.arm_feedback_topic,
    }
    for label, topic in required.items():
        if counts.get(topic, 0) <= 0:
            raise ValueError(f"Required topic missing or empty for {label}: {topic}")
        resolved[label] = topic

    if mapping.gripper_feedback_topic and counts.get(mapping.gripper_feedback_topic, 0) > 0:
        resolved["gripper_feedback"] = mapping.gripper_feedback_topic
    if mapping.arm_target_topic and counts.get(mapping.arm_target_topic, 0) > 0:
        resolved["arm_target"] = mapping.arm_target_topic
    if mapping.gripper_target_topic and counts.get(mapping.gripper_target_topic, 0) > 0:
        resolved["gripper_target"] = mapping.gripper_target_topic
    return resolved


def _build_uniform_timeline(start_ns: int, end_ns: int, hz: float) -> np.ndarray:
    step_ns = int(round(1e9 / hz))
    if step_ns <= 0:
        raise ValueError(f"Invalid target hz: {hz}")
    return np.arange(start_ns, end_ns + 1, step_ns, dtype=np.int64)


def _make_conversion_plan(
    cfg: ConversionConfig,
    metadata: dict,
    meta_json: dict,
    resolved_topics: Dict[str, str],
) -> dict:
    bag_duration_s = float(metadata["rosbag2_bagfile_information"]["duration"]["nanoseconds"]) / 1e9
    return {
        "exp_name": cfg.exp_name,
        "arm": cfg.arm,
        "setup_mode": cfg.setup_mode,
        "uses_gripper": cfg.uses_gripper,
        "target_control_hz": cfg.control_hz,
        "bag_duration_seconds": bag_duration_s,
        "teleoperation_type": meta_json.get("operation_info", {}).get("teleoperation_type"),
        "resolved_topics": resolved_topics,
        "conversion_strategy": {
            "images": "keep original image payloads in stage-1 conversion; resize later for model-specific training",
            "observations": {
                "image_primary": "from resolved head camera topic",
                "image_wrist": "from resolved wrist camera topic",
                "tcp_pose": "from end-effector pose topic",
                "joint_pos": "from arm JointState feedback",
                "joint_vel": "use JointState.velocity when present, otherwise estimate from consecutive joint_pos",
                "gripper_pose": "from gripper feedback if available, otherwise zeros",
                "tcp_force": "currently unavailable in inspected official teleop sample; fill zeros",
                "tcp_torque": "currently unavailable in inspected official teleop sample; fill zeros",
                "joint_effort": "use JointState.effort only if present and non-empty, otherwise fill zeros",
            },
            "actions": {
                "preferred": "reconstruct 7D action from consecutive EE pose deltas",
                "arm_target_aux": "joint target topic is available and may be used for diagnostics",
                "gripper": "use gripper target/feedback only when task uses gripper; otherwise keep open-compatible value",
            },
            "stage2": {
                "with_embeddings": "optional; when enabled, use the same Octo embedding post-processing path as record_demos_r1lite_octo.py",
            },
        },
    }


def _load_rosbags_reader():
    try:
        from rosbags.highlevel import AnyReader
    except ImportError as exc:
        raise RuntimeError(
            "rosbags is required for full conversion. Please install it in the conversion environment "
            "with: python -m pip install rosbags"
        ) from exc
    return AnyReader


def _decode_compressed_image(msg) -> np.ndarray:
    data = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode compressed image payload")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _extract_pose_stamped(msg) -> np.ndarray:
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


def _extract_joint_state(msg, default_len: int) -> Dict[str, np.ndarray]:
    position = np.asarray(list(getattr(msg, "position", [])), dtype=np.float32)
    velocity = np.asarray(list(getattr(msg, "velocity", [])), dtype=np.float32)
    effort = np.asarray(list(getattr(msg, "effort", [])), dtype=np.float32)

    def _fit(vec: np.ndarray, fill: float = 0.0) -> np.ndarray:
        if vec.size >= default_len:
            return vec[:default_len].astype(np.float32)
        out = np.full((default_len,), fill, dtype=np.float32)
        if vec.size > 0:
            out[: vec.size] = vec.astype(np.float32)
        return out

    return {
        "position": _fit(position),
        "velocity": _fit(velocity),
        "effort": _fit(effort),
        "has_velocity": bool(velocity.size >= default_len),
        "has_effort": bool(effort.size >= default_len),
    }


def _extract_gripper_joint_state(msg) -> Dict[str, float]:
    position = list(getattr(msg, "position", []))
    velocity = list(getattr(msg, "velocity", []))
    effort = list(getattr(msg, "effort", []))
    return {
        "position": float(position[0]) if position else 0.0,
        "velocity": float(velocity[0]) if velocity else 0.0,
        "effort": float(effort[0]) if effort else 0.0,
    }


def _parse_message(topic: str, msg) -> Any:
    if topic.endswith("/compressed"):
        return _decode_compressed_image(msg)
    if "pose_ee_arm_" in topic or "target_pose_arm_" in topic:
        return _extract_pose_stamped(msg)
    if "feedback_arm_" in topic or "target_joint_state_arm_" in topic:
        return _extract_joint_state(msg, default_len=6)
    if "gripper" in topic or "feedback_hand_" in topic:
        return _extract_gripper_joint_state(msg)
    return msg


def _read_topic_series(input_dir: Path, metadata: dict, topics: Sequence[str]) -> Dict[str, TopicSeries]:
    AnyReader = _load_rosbags_reader()
    topic_set = set(topics)
    buckets: Dict[str, List[Tuple[int, Any]]] = defaultdict(list)
    reader_input_dir, temp_dir = _prepare_rosbags_input_dir(input_dir, metadata)
    try:
        with AnyReader([reader_input_dir]) as reader:
            connections = [conn for conn in reader.connections if conn.topic in topic_set]
            for connection, timestamp, rawdata in reader.messages(connections=connections):
                msg = reader.deserialize(rawdata, connection.msgtype)
                buckets[connection.topic].append((int(timestamp), _parse_message(connection.topic, msg)))
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()

    result = {}
    for topic in topics:
        samples = buckets.get(topic, [])
        if not samples:
            continue
        timestamps = np.asarray([item[0] for item in samples], dtype=np.int64)
        values = [item[1] for item in samples]
        result[topic] = TopicSeries(timestamps=timestamps, values=values)
    return result


def _series_value_at(series: TopicSeries, timestamp_ns: int) -> Any:
    idx = int(np.searchsorted(series.timestamps, timestamp_ns, side="right") - 1)
    if idx < 0:
        idx = 0
    return series.values[idx]


def _effective_interval(series_map: Dict[str, TopicSeries], topic_names: Sequence[str]) -> Tuple[int, int]:
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


def _rotvec_delta(current_xyzw: np.ndarray, next_xyzw: np.ndarray) -> np.ndarray:
    current = Rotation.from_quat(current_xyzw)
    nxt = Rotation.from_quat(next_xyzw)
    return ((current.inv() * nxt).as_rotvec()).astype(np.float32)


def _euler_delta_xyz(current_xyzw: np.ndarray, next_xyzw: np.ndarray) -> np.ndarray:
    current = Rotation.from_quat(current_xyzw)
    nxt = Rotation.from_quat(next_xyzw)
    # env.step() 使用的是: delta_euler * current
    # 所以离线重建 action 时，要反推出这个 delta_euler 本身。
    delta = nxt * current.inv()
    return delta.as_euler("xyz").astype(np.float32)


def _estimate_tcp_velocity(current_pose: np.ndarray, next_pose: np.ndarray, dt: float) -> np.ndarray:
    if dt <= 0:
        return np.zeros((6,), dtype=np.float32)
    linear = (next_pose[:3] - current_pose[:3]) / dt
    angular = _rotvec_delta(current_pose[3:], next_pose[3:]) / dt
    return np.concatenate([linear, angular], axis=0).astype(np.float32)


def _estimate_joint_velocity(current_joint: np.ndarray, next_joint: np.ndarray, dt: float) -> np.ndarray:
    if dt <= 0:
        return np.zeros_like(current_joint, dtype=np.float32)
    return ((next_joint - current_joint) / dt).astype(np.float32)


def _build_dense_frames(
    cfg: ConversionConfig,
    resolved_topics: Dict[str, str],
    series_map: Dict[str, TopicSeries],
    timeline_ns: np.ndarray,
) -> List[Dict[str, Any]]:
    frames: List[Dict[str, Any]] = []
    for timestamp_ns in timeline_ns:
        pose = _series_value_at(series_map[resolved_topics["tcp_pose"]], int(timestamp_ns))
        joint_fb = _series_value_at(series_map[resolved_topics["joint_feedback"]], int(timestamp_ns))
        gripper_fb = None
        if "gripper_feedback" in resolved_topics:
            gripper_fb = _series_value_at(series_map[resolved_topics["gripper_feedback"]], int(timestamp_ns))
        image_primary = _series_value_at(series_map[resolved_topics["image_primary"]], int(timestamp_ns))
        image_wrist = _series_value_at(series_map[resolved_topics["image_wrist"]], int(timestamp_ns))
        gripper_action = 1.0
        if cfg.uses_gripper and "gripper_target" in resolved_topics:
            gripper_target = _series_value_at(series_map[resolved_topics["gripper_target"]], int(timestamp_ns))
            gripper_action = float(np.clip(gripper_target["position"] / 50.0 - 1.0, -1.0, 1.0))

        frames.append(
            {
                "timestamp_ns": int(timestamp_ns),
                "image_primary": image_primary,
                "image_wrist": image_wrist,
                "tcp_pose": np.asarray(pose, dtype=np.float32),
                "joint_pos": np.asarray(joint_fb["position"], dtype=np.float32),
                "joint_vel_raw": np.asarray(joint_fb["velocity"], dtype=np.float32),
                "joint_effort_raw": np.asarray(joint_fb["effort"], dtype=np.float32),
                "joint_has_velocity": bool(joint_fb["has_velocity"]),
                "joint_has_effort": bool(joint_fb["has_effort"]),
                "gripper_pose": np.asarray(
                    [gripper_fb["position"] if gripper_fb is not None else 0.0], dtype=np.float32
                ),
                "gripper_action": float(gripper_action),
            }
        )

    if not frames:
        return frames

    for idx in range(len(frames)):
        if idx < len(frames) - 1:
            dt = max(1e-6, (frames[idx + 1]["timestamp_ns"] - frames[idx]["timestamp_ns"]) / 1e9)
            next_frame = frames[idx + 1]
        else:
            dt = max(1e-6, (frames[idx]["timestamp_ns"] - frames[idx - 1]["timestamp_ns"]) / 1e9) if idx > 0 else 1.0 / cfg.control_hz
            next_frame = frames[idx]
        frames[idx]["tcp_vel"] = _estimate_tcp_velocity(frames[idx]["tcp_pose"], next_frame["tcp_pose"], dt)
        if frames[idx]["joint_has_velocity"]:
            frames[idx]["joint_vel"] = frames[idx]["joint_vel_raw"].astype(np.float32)
        else:
            frames[idx]["joint_vel"] = _estimate_joint_velocity(frames[idx]["joint_pos"], next_frame["joint_pos"], dt)
        if frames[idx]["joint_has_effort"]:
            frames[idx]["joint_effort"] = frames[idx]["joint_effort_raw"].astype(np.float32)
        else:
            frames[idx]["joint_effort"] = np.zeros((6,), dtype=np.float32)
        frames[idx]["tcp_force"] = np.zeros((3,), dtype=np.float32)
        frames[idx]["tcp_torque"] = np.zeros((3,), dtype=np.float32)
    return frames


def _compose_proprio_state(frame: Dict[str, Any], proprio_keys: Sequence[str]) -> np.ndarray:
    proprio_keys = tuple(sorted(proprio_keys))
    field_map = {
        "tcp_pose": frame["tcp_pose"],
        "tcp_vel": frame["tcp_vel"],
        "tcp_force": frame["tcp_force"],
        "tcp_torque": frame["tcp_torque"],
        "gripper_pose": frame["gripper_pose"],
        "joint_pos": frame["joint_pos"],
        "joint_vel": frame["joint_vel"],
        "joint_effort": frame["joint_effort"],
    }
    proprio_space = gym.spaces.Dict(
        {
            key: gym.spaces.Box(
                -np.inf,
                np.inf,
                shape=_PROPRIO_FIELD_SHAPES[key],
                dtype=np.float32,
            )
            for key in proprio_keys
        }
    )
    proprio_dict = {
        key: np.asarray(field_map[key], dtype=np.float32).reshape(_PROPRIO_FIELD_SHAPES[key])
        for key in proprio_keys
    }
    return gym.spaces.flatten(proprio_space, proprio_dict).astype(np.float32)


def _stack_history(sequence: Sequence[Any], end_idx: int, horizon: int) -> List[Any]:
    start_idx = max(0, end_idx - horizon + 1)
    window = list(sequence[start_idx : end_idx + 1])
    while len(window) < horizon:
        window.insert(0, window[0])
    return window


def _build_observation(frames: Sequence[Dict[str, Any]], index: int, cfg: ConversionConfig) -> Dict[str, np.ndarray]:
    history = _stack_history(frames, index, cfg.obs_horizon)
    state = np.stack([_compose_proprio_state(frame, cfg.proprio_keys) for frame in history], axis=0)
    image_primary = np.stack([np.asarray(frame["image_primary"], dtype=np.uint8) for frame in history], axis=0)
    image_wrist = np.stack([np.asarray(frame["image_wrist"], dtype=np.uint8) for frame in history], axis=0)
    return {
        "state": state.astype(np.float32),
        "image_primary": image_primary.astype(np.uint8),
        "image_wrist": image_wrist.astype(np.uint8),
    }


def _build_action(current_frame: Dict[str, Any], next_frame: Dict[str, Any], cfg: ConversionConfig) -> np.ndarray:
    action = np.zeros((7,), dtype=np.float32)
    action[:3] = (
        (next_frame["tcp_pose"][:3] - current_frame["tcp_pose"][:3]) / max(cfg.action_xyz_scale, 1e-6)
    )
    action[3:6] = _euler_delta_xyz(current_frame["tcp_pose"][3:], next_frame["tcp_pose"][3:]) / max(
        cfg.action_rot_scale, 1e-6
    )
    action[6] = float(next_frame["gripper_action"] if cfg.uses_gripper else 1.0)
    return np.clip(action, -1.0, 1.0).astype(np.float32)


def _compute_reward_and_success(next_frame: Dict[str, Any], cfg: ConversionConfig) -> Tuple[float, bool, Dict[str, Any]]:
    if cfg.task_config is None:
        return 0.0, False, {}

    tcp_pose = np.asarray(next_frame["tcp_pose"], dtype=np.float32)
    target_pose = cfg.task_config.target_pose.astype(np.float32)
    pos_error = float(np.linalg.norm(tcp_pose[:3] - target_pose[:3]))
    ori_error = quat_angle_error_rad(tcp_pose[3:], target_pose[3:])
    succeed = (
        pos_error <= cfg.task_config.position_tolerance_m
        and ori_error <= cfg.task_config.orientation_tolerance_rad
    )

    reward = cfg.task_config.reward_neg
    reward -= cfg.task_config.dense_position_weight * pos_error
    reward -= cfg.task_config.dense_orientation_weight * ori_error
    if succeed:
        reward = cfg.task_config.success_reward

    info = {
        "succeed": succeed,
        "target_pose": target_pose.copy(),
        "position_error_m": pos_error,
        "orientation_error_rad": ori_error,
        "reward_components": {
            "base": cfg.task_config.reward_neg,
            "position_penalty": -cfg.task_config.dense_position_weight * pos_error,
            "orientation_penalty": -cfg.task_config.dense_orientation_weight * ori_error,
            "success_bonus": cfg.task_config.success_reward if succeed else 0.0,
        },
    }
    return float(reward), bool(succeed), info


def _build_transitions(
    cfg: ConversionConfig,
    frames: Sequence[Dict[str, Any]],
    require_success: bool,
) -> List[Dict[str, Any]]:
    if len(frames) < 2:
        raise ValueError("Need at least two aligned frames to build transitions")

    transitions: List[Dict[str, Any]] = []
    success_seen = False
    for idx in range(len(frames) - 1):
        obs = _build_observation(frames, idx, cfg)
        next_obs = _build_observation(frames, idx + 1, cfg)
        action = _build_action(frames[idx], frames[idx + 1], cfg)
        reward, succeed, info = _compute_reward_and_success(frames[idx + 1], cfg)
        done = bool(succeed or idx == len(frames) - 2)
        transition = {
            "observations": obs,
            "actions": action,
            "next_observations": next_obs,
            "rewards": float(reward),
            "masks": float(1.0 - done),
            "dones": bool(done),
            "infos": {
                **info,
                "timestamp_ns": int(frames[idx]["timestamp_ns"]),
                "next_timestamp_ns": int(frames[idx + 1]["timestamp_ns"]),
                "conversion_source": "official_teleop_rosbags",
                "state_layout": "gym_sorted",
                "proprio_keys": tuple(cfg.proprio_keys),
                "action_type": "env_normalized_eef_delta_and_absolute_gripper_target",
                "xyz_scale": float(cfg.action_xyz_scale),
                "rot_scale": float(cfg.action_rot_scale),
            },
        }
        transitions.append(transition)
        if succeed:
            success_seen = True
            break

    if require_success and not success_seen:
        raise ValueError("Converted episode never reached task success under the current task definition")

    transitions = add_mc_returns_to_trajectory(
        transitions,
        gamma=cfg.discount,
        reward_scale=1.0,
        reward_bias=0.0,
        reward_neg=cfg.reward_neg,
        is_sparse_reward=True,
    )
    return transitions


def _load_octo_model():
    try:
        from octo.model.octo_model import OctoModel
    except ImportError as exc:
        raise RuntimeError(
            "Octo is required for --with_embeddings but could not be imported from the current environment."
        ) from exc
    return OctoModel


def _maybe_add_embeddings(
    transitions: List[Dict[str, Any]],
    cfg: ConversionConfig,
    with_embeddings: bool,
) -> List[Dict[str, Any]]:
    if not with_embeddings:
        return transitions
    if len(cfg.image_keys) < 2:
        raise ValueError("--with_embeddings requires at least two image_keys in the target experiment config")
    OctoModel = _load_octo_model()
    model = OctoModel.load_pretrained(cfg.octo_path)
    tasks = model.create_tasks(texts=[cfg.task_desc])
    transitions = add_embeddings_to_trajectory(
        transitions,
        model,
        tasks=tasks,
        image_keys=tuple(cfg.image_keys),
    )
    transitions = add_next_embeddings_to_trajectory(transitions)
    return transitions


def convert_official_teleop_to_transitions(
    exp_name: str,
    input_dir: Path,
    output_file: Path,
    meta_json: Optional[Path] = None,
    dry_run: bool = False,
    require_success: bool = False,
    with_embeddings: bool = False,
) -> dict | list:
    cfg = _load_task_config(exp_name)
    metadata_path, mcap_path, meta_json_path = _resolve_bag_paths(input_dir, meta_json)
    metadata, meta = _load_metadata(metadata_path, meta_json_path)
    counts = _topic_counts(metadata)
    resolved_topics = _validate_required_topics(_build_topic_mapping(cfg), counts)
    plan = _make_conversion_plan(cfg, metadata, meta, resolved_topics)
    plan["files"] = {
        "metadata_yaml": str(metadata_path),
        "meta_json": str(meta_json_path),
        "mcap": str(mcap_path),
        "output_file": str(output_file),
    }

    if dry_run:
        timeline_ns = _build_uniform_timeline(
            int(metadata["rosbag2_bagfile_information"]["starting_time"]["nanoseconds_since_epoch"]),
            int(metadata["rosbag2_bagfile_information"]["starting_time"]["nanoseconds_since_epoch"])
            + int(metadata["rosbag2_bagfile_information"]["duration"]["nanoseconds"]),
            cfg.control_hz,
        )
        plan["timeline"] = {
            "num_steps": int(len(timeline_ns)),
            "start_ns": int(timeline_ns[0]) if len(timeline_ns) else None,
            "end_ns": int(timeline_ns[-1]) if len(timeline_ns) else None,
        }
        return plan

    selected_topics = list(dict.fromkeys(resolved_topics.values()))
    series_map = _read_topic_series(input_dir, metadata, selected_topics)
    required_for_overlap = [
        resolved_topics["image_primary"],
        resolved_topics["image_wrist"],
        resolved_topics["tcp_pose"],
        resolved_topics["joint_feedback"],
    ]
    start_ns, end_ns = _effective_interval(series_map, required_for_overlap)
    timeline_ns = _build_uniform_timeline(start_ns, end_ns, cfg.control_hz)
    frames = _build_dense_frames(cfg, resolved_topics, series_map, timeline_ns)
    transitions = _build_transitions(cfg, frames, require_success=require_success)
    transitions = _maybe_add_embeddings(transitions, cfg, with_embeddings=with_embeddings)
    return transitions


def main():
    parser = argparse.ArgumentParser(description="Convert official teleop mcap into ConRFT RL transitions.")
    parser.add_argument("--exp_name", required=True, help="Target experiment name, e.g. r1lite_reach_target")
    parser.add_argument("--input_dir", required=True, help="Path to <episode>_RAW directory containing metadata.yaml and .mcap")
    parser.add_argument("--meta_json", default=None, help="Optional path to sibling <episode>_RAW.json")
    parser.add_argument("--output_file", required=True, help="Where to write the transition .pkl or plan .json")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only validate metadata/config/topics and emit a JSON conversion plan.",
    )
    parser.add_argument(
        "--require_success",
        action="store_true",
        help="Fail conversion if the task reward logic never reaches success within the bag.",
    )
    parser.add_argument(
        "--with_embeddings",
        action="store_true",
        help="Run Octo stage-2 post-processing and add embeddings/next_embeddings to the converted transitions.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    meta_json = Path(args.meta_json).expanduser().resolve() if args.meta_json else None
    output_file = Path(args.output_file).expanduser().resolve()

    plan_or_data = convert_official_teleop_to_transitions(
        exp_name=args.exp_name,
        input_dir=input_dir,
        output_file=output_file,
        meta_json=meta_json,
        dry_run=args.dry_run,
        require_success=args.require_success,
        with_embeddings=args.with_embeddings,
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        with output_file.open("w", encoding="utf-8") as f:
            json.dump(plan_or_data, f, indent=2, ensure_ascii=False)
        print(f"Wrote conversion plan to {output_file}")
        return

    with output_file.open("wb") as f:
        pkl.dump(plan_or_data, f)
    print(f"Wrote converted transitions to {output_file}")


if __name__ == "__main__":
    main()
