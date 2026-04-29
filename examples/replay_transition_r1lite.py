#!/usr/bin/env python3

import argparse
import csv
import json
import os
import pickle
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation


def _ensure_examples_on_path():
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

from experiments.mappings import CONFIG_MAPPING  # noqa: E402
from r1lite_env import DualR1LiteEnv, R1LiteArmEnv  # noqa: E402


_FIELD_SHAPES = {
    "tcp_pose": (7,),
    "tcp_vel": (6,),
    "tcp_force": (3,),
    "tcp_torque": (3,),
    "gripper_pose": (1,),
    "joint_pos": (6,),
    "joint_vel": (6,),
    "joint_effort": (6,),
    "left/tcp_pose": (7,),
    "left/tcp_vel": (6,),
    "left/tcp_force": (3,),
    "left/tcp_torque": (3,),
    "left/gripper_pose": (1,),
    "left/joint_pos": (6,),
    "left/joint_vel": (6,),
    "left/joint_effort": (6,),
    "right/tcp_pose": (7,),
    "right/tcp_vel": (6,),
    "right/tcp_force": (3,),
    "right/tcp_torque": (3,),
    "right/gripper_pose": (1,),
    "right/joint_pos": (6,),
    "right/joint_vel": (6,),
    "right/joint_effort": (6,),
    "torso": (9,),
    "torso_pos": (1,),
}


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


def _canonical_pose(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float32).copy()
    if pose.shape[0] >= 7 and pose[6] < 0:
        pose[3:7] *= -1.0
    return pose


def _quat_angle_error_rad(current_xyzw: np.ndarray, target_xyzw: np.ndarray) -> float:
    if float(np.linalg.norm(current_xyzw)) < 1e-8 or float(np.linalg.norm(target_xyzw)) < 1e-8:
        return float("nan")
    current = Rotation.from_quat(current_xyzw)
    target = Rotation.from_quat(target_xyzw)
    delta = current.inv() * target
    return float(np.linalg.norm(delta.as_rotvec()))


def _pose_error(actual_pose: Optional[np.ndarray], expected_pose: Optional[np.ndarray]) -> Dict[str, float]:
    if actual_pose is None or expected_pose is None:
        return {"position_error_m": float("nan"), "orientation_error_rad": float("nan")}
    return {
        "position_error_m": float(np.linalg.norm(actual_pose[:3] - expected_pose[:3])),
        "orientation_error_rad": _quat_angle_error_rad(actual_pose[3:], expected_pose[3:]),
    }


def _canonical_proprio_keys(proprio_keys: Sequence[str]) -> List[str]:
    return sorted(proprio_keys)


def _infer_flatten_slices(proprio_keys: Sequence[str]) -> Dict[str, slice]:
    proprio_keys = _canonical_proprio_keys(proprio_keys)
    missing = [key for key in proprio_keys if key not in _FIELD_SHAPES]
    if missing:
        raise ValueError(f"Unknown proprio key shape(s): {missing}")

    proprio_space = gym.spaces.Dict(
        {
            key: gym.spaces.Box(-np.inf, np.inf, shape=_FIELD_SHAPES[key], dtype=np.float32)
            for key in proprio_keys
        }
    )
    slices: Dict[str, slice] = {}
    for key in proprio_keys:
        sample = {
            item: np.zeros(_FIELD_SHAPES[item], dtype=np.float32)
            for item in proprio_keys
        }
        sample[key] = np.arange(
            1,
            int(np.prod(_FIELD_SHAPES[key])) + 1,
            dtype=np.float32,
        ).reshape(_FIELD_SHAPES[key])
        flat = gym.spaces.flatten(proprio_space, sample)
        nz = np.flatnonzero(flat)
        if nz.size == 0:
            raise RuntimeError(f"Failed to infer flattened slice for key {key}")
        slices[key] = slice(int(nz[0]), int(nz[-1]) + 1)
    return slices


def _last_state(obs: Dict) -> np.ndarray:
    state = np.asarray(obs["state"], dtype=np.float32)
    if state.ndim == 1:
        return state
    return state[-1]


def _extract_field(obs: Dict, field_slices: Dict[str, slice], key: str) -> np.ndarray:
    return np.asarray(_last_state(obs)[field_slices[key]], dtype=np.float32)


def _extract_pose_map(
    obs: Dict,
    field_slices: Dict[str, slice],
    task_mode: str,
    single_arm: Optional[str],
) -> Dict[str, np.ndarray]:
    if task_mode == "dual":
        return {
            "left": _canonical_pose(_extract_field(obs, field_slices, "left/tcp_pose")),
            "right": _canonical_pose(_extract_field(obs, field_slices, "right/tcp_pose")),
        }
    return {
        single_arm or "arm": _canonical_pose(_extract_field(obs, field_slices, "tcp_pose")),
    }


def _extract_gripper_map(
    obs: Dict,
    field_slices: Dict[str, slice],
    task_mode: str,
    single_arm: Optional[str],
) -> Dict[str, float]:
    if task_mode == "dual":
        result = {}
        for arm in ("left", "right"):
            key = f"{arm}/gripper_pose"
            if key in field_slices:
                result[arm] = float(_extract_field(obs, field_slices, key).reshape(-1)[0])
        return result
    if "gripper_pose" not in field_slices:
        return {}
    return {single_arm or "arm": float(_extract_field(obs, field_slices, "gripper_pose").reshape(-1)[0])}


def _action_map(action: np.ndarray, task_mode: str, single_arm: Optional[str]) -> Dict[str, np.ndarray]:
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if task_mode == "dual":
        if action.size < 14:
            raise ValueError(f"Dual replay expected at least 14 action dims, got {action.shape}")
        return {"left": action[:7], "right": action[7:14]}
    if action.size < 7:
        raise ValueError(f"Single-arm replay expected at least 7 action dims, got {action.shape}")
    return {single_arm or "arm": action[:7]}


def _gripper_target_from_action(env, arm: str, arm_action: np.ndarray) -> Optional[float]:
    if bool(env.config.FIX_GRIPPER_OPEN):
        return None
    return float(np.clip((float(arm_action[6]) + 1.0) * 50.0, 0.0, 100.0))


def _clip_pose(env, arm: str, pose: np.ndarray) -> np.ndarray:
    env_arm = arm if isinstance(env, DualR1LiteEnv) else env.arm
    return env._clip_pose_to_safety_box(env_arm, pose)


def _targets_from_action(
    env,
    reference_poses: Dict[str, np.ndarray],
    actions: Dict[str, np.ndarray],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, Optional[float]]]:
    unclipped_targets = {}
    clipped_targets = {}
    gripper_targets = {}
    for arm, arm_action in actions.items():
        target = env._target_pose_from_action(reference_poses[arm], arm_action[:6])
        clipped = _clip_pose(env, arm, target)
        unclipped_targets[arm] = target
        clipped_targets[arm] = clipped
        gripper_targets[arm] = _gripper_target_from_action(env, arm, arm_action)
    return unclipped_targets, clipped_targets, gripper_targets


def _layout_invalid_count(
    transitions: List[Dict],
    field_slices: Dict[str, slice],
    task_mode: str,
    single_arm: Optional[str],
    max_checks: int = 20,
) -> int:
    invalid = 0
    for transition in transitions[:max_checks]:
        for obs_key in ("observations", "next_observations"):
            try:
                poses = _extract_pose_map(transition[obs_key], field_slices, task_mode, single_arm)
            except Exception:
                invalid += 10
                continue
            for pose in poses.values():
                quat_norm = float(np.linalg.norm(pose[3:7]))
                xyz_abs_max = float(np.max(np.abs(pose[:3])))
                if not (0.5 <= quat_norm <= 1.5) or xyz_abs_max > 5.0:
                    invalid += 1
    return invalid


def _select_field_slices(
    transitions: List[Dict],
    proprio_keys: Sequence[str],
    task_mode: str,
    single_arm: Optional[str],
) -> Tuple[List[str], Dict[str, slice]]:
    selected_order = _canonical_proprio_keys(proprio_keys)
    field_slices = _infer_flatten_slices(selected_order)
    invalid = _layout_invalid_count(transitions, field_slices, task_mode, single_arm)
    if invalid:
        raise ValueError(
            "This PKL does not match the canonical gym-sorted state layout for the task. "
            f"Invalid pose extraction count={invalid}. Re-convert the dataset with the latest converter."
        )
    return selected_order, field_slices


def _make_env(exp_name: str):
    if exp_name not in CONFIG_MAPPING:
        raise ValueError(f"Unknown experiment: {exp_name}")
    cfg = CONFIG_MAPPING[exp_name]()
    wrapped_env = cfg.get_environment(fake_env=True, save_video=False, classifier=False, stack_obs_num=2)
    env = wrapped_env.unwrapped
    if isinstance(env, DualR1LiteEnv):
        task_mode = "dual"
        single_arm = None
    elif isinstance(env, R1LiteArmEnv):
        task_mode = "single"
        single_arm = env.arm
    else:
        raise TypeError(f"{exp_name} resolved to unsupported base env type: {type(env).__name__}")
    proprio_keys = _canonical_proprio_keys(getattr(cfg, "proprio_keys", []))
    return cfg, env, task_mode, single_arm, proprio_keys


def _trajectory_indices(trajectories: List[List[Dict]], trajectory_index: int, all_trajectories: bool) -> List[int]:
    if all_trajectories:
        return list(range(len(trajectories)))
    if trajectory_index < 0 or trajectory_index >= len(trajectories):
        raise IndexError(
            f"trajectory_index={trajectory_index} is out of range for {len(trajectories)} trajectories"
        )
    return [trajectory_index]


def _print_summary(trajectories: List[List[Dict]], max_items: int = 20):
    print(f"Found {len(trajectories)} trajectory(ies)")
    for idx, trajectory in enumerate(trajectories[:max_items]):
        first = trajectory[0]
        last = trajectory[-1]
        succeed = bool(last.get("infos", {}).get("succeed", False))
        reward_sum = float(sum(float(t.get("rewards", 0.0)) for t in trajectory))
        actions = np.stack([np.asarray(t["actions"], dtype=np.float32).reshape(-1) for t in trajectory], axis=0)
        action_norms = np.linalg.norm(actions[:, : min(6, actions.shape[1])], axis=1)
        nonzero_idx = np.flatnonzero(action_norms > 1e-6)
        first_nonzero = None if nonzero_idx.size == 0 else int(nonzero_idx[0])
        source = first.get("infos", {}).get("conversion_source", "unknown")
        print(
            f"  trajectory[{idx}]: steps={len(trajectory)}, "
            f"succeed={succeed}, return={reward_sum:.3f}, "
            f"first_nonzero_action_step={first_nonzero}, source={source}"
        )
    if len(trajectories) > max_items:
        print(f"  ... {len(trajectories) - max_items} more trajectory(ies)")


def _row_base(
    trajectory_index: int,
    step_index: int,
    exec_mode: str,
    replay_mode: str,
    offline_reference: str,
    transition: Dict,
) -> Dict[str, object]:
    infos = transition.get("infos", {})
    return {
        "trajectory_index": trajectory_index,
        "step_index": step_index,
        "exec_mode": exec_mode,
        "replay_mode": replay_mode,
        "offline_reference": offline_reference,
        "reward": float(transition.get("rewards", 0.0)),
        "done": bool(transition.get("dones", False)),
        "conversion_source": infos.get("conversion_source", ""),
        "action_type": infos.get("action_type", ""),
        "frame_index": infos.get("frame_index", ""),
        "next_frame_index": infos.get("next_frame_index", ""),
    }


def _pose_columns(prefix: str, pose: Optional[np.ndarray]) -> Dict[str, object]:
    if pose is None:
        return {f"{prefix}_{name}": "" for name in ("x", "y", "z", "qx", "qy", "qz", "qw")}
    names = ("x", "y", "z", "qx", "qy", "qz", "qw")
    return {f"{prefix}_{name}": float(value) for name, value in zip(names, pose[:7])}


def _append_arm_row(
    rows: List[Dict[str, object]],
    base: Dict[str, object],
    arm: str,
    action: Optional[np.ndarray],
    recorded_current_pose: Optional[np.ndarray],
    recorded_next_pose: Optional[np.ndarray],
    reference_pose: Optional[np.ndarray],
    unclipped_target_pose: Optional[np.ndarray],
    target_pose: Optional[np.ndarray],
    actual_pose: Optional[np.ndarray],
    recorded_current_gripper: Optional[float],
    recorded_next_gripper: Optional[float],
    target_gripper: Optional[float],
    actual_gripper: Optional[float],
):
    actual_to_target = _pose_error(actual_pose, target_pose)
    target_to_recorded = _pose_error(target_pose, recorded_next_pose)
    actual_to_recorded = _pose_error(actual_pose, recorded_next_pose)
    reference_to_recorded = _pose_error(reference_pose, recorded_current_pose)
    clip_error = _pose_error(target_pose, unclipped_target_pose)

    row = dict(base)
    row.update(
        {
            "arm": arm,
            "action_norm": float(np.linalg.norm(action[:6])) if action is not None else float("nan"),
            "action_gripper": float(action[6]) if action is not None and action.size >= 7 else "",
            "actual_to_target_pos_err_m": actual_to_target["position_error_m"],
            "actual_to_target_ori_err_rad": actual_to_target["orientation_error_rad"],
            "target_to_recorded_pos_err_m": target_to_recorded["position_error_m"],
            "target_to_recorded_ori_err_rad": target_to_recorded["orientation_error_rad"],
            "actual_to_recorded_pos_err_m": actual_to_recorded["position_error_m"],
            "actual_to_recorded_ori_err_rad": actual_to_recorded["orientation_error_rad"],
            "reference_to_recorded_current_pos_err_m": reference_to_recorded["position_error_m"],
            "reference_to_recorded_current_ori_err_rad": reference_to_recorded["orientation_error_rad"],
            "clip_pos_m": clip_error["position_error_m"],
            "clip_ori_rad": clip_error["orientation_error_rad"],
            "recorded_current_gripper": "" if recorded_current_gripper is None else recorded_current_gripper,
            "recorded_next_gripper": "" if recorded_next_gripper is None else recorded_next_gripper,
            "target_gripper": "" if target_gripper is None else target_gripper,
            "actual_gripper": "" if actual_gripper is None else actual_gripper,
            "target_to_recorded_gripper_err": ""
            if target_gripper is None or recorded_next_gripper is None
            else abs(float(target_gripper) - float(recorded_next_gripper)),
            "actual_to_target_gripper_err": ""
            if actual_gripper is None or target_gripper is None
            else abs(float(actual_gripper) - float(target_gripper)),
        }
    )
    row.update(_pose_columns("recorded_current", recorded_current_pose))
    row.update(_pose_columns("recorded_next", recorded_next_pose))
    row.update(_pose_columns("reference", reference_pose))
    row.update(_pose_columns("unclipped_target", unclipped_target_pose))
    row.update(_pose_columns("target", target_pose))
    row.update(_pose_columns("actual", actual_pose))
    rows.append(row)


def run_offline(
    env,
    trajectories: List[List[Dict]],
    selected_indices: Iterable[int],
    field_slices: Dict[str, slice],
    task_mode: str,
    single_arm: Optional[str],
    replay_mode: str,
    offline_reference: str,
    start_step: int,
    max_steps: Optional[int],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for traj_idx in selected_indices:
        trajectory = trajectories[traj_idx]
        rollout_poses = None
        rollout_grippers = None
        stop = len(trajectory) if max_steps is None else min(len(trajectory), start_step + max_steps)
        for step_idx in range(start_step, stop):
            transition = trajectory[step_idx]
            recorded_current = _extract_pose_map(transition["observations"], field_slices, task_mode, single_arm)
            recorded_next = _extract_pose_map(transition["next_observations"], field_slices, task_mode, single_arm)
            recorded_current_gripper = _extract_gripper_map(transition["observations"], field_slices, task_mode, single_arm)
            recorded_next_gripper = _extract_gripper_map(transition["next_observations"], field_slices, task_mode, single_arm)
            actions = _action_map(transition["actions"], task_mode, single_arm)

            if rollout_poses is None or offline_reference == "teacher_forced":
                reference = {arm: pose.copy() for arm, pose in recorded_current.items()}
                rollout_grippers = recorded_current_gripper.copy()
            else:
                reference = {arm: pose.copy() for arm, pose in rollout_poses.items()}

            if replay_mode == "action":
                unclipped_targets, targets, target_grippers = _targets_from_action(env, reference, actions)
            else:
                unclipped_targets = {arm: pose.copy() for arm, pose in recorded_next.items()}
                targets = {arm: _clip_pose(env, arm, pose) for arm, pose in recorded_next.items()}
                target_grippers = recorded_next_gripper.copy()

            rollout_poses = {arm: pose.copy() for arm, pose in targets.items()}
            if target_grippers:
                rollout_grippers = {arm: value for arm, value in target_grippers.items() if value is not None}

            base = _row_base(traj_idx, step_idx, "offline", replay_mode, offline_reference, transition)
            for arm in targets:
                _append_arm_row(
                    rows,
                    base,
                    arm,
                    actions.get(arm),
                    recorded_current.get(arm),
                    recorded_next.get(arm),
                    reference.get(arm),
                    unclipped_targets.get(arm),
                    targets.get(arm),
                    targets.get(arm),
                    recorded_current_gripper.get(arm),
                    recorded_next_gripper.get(arm),
                    target_grippers.get(arm),
                    None if rollout_grippers is None else rollout_grippers.get(arm),
                )
    return rows


def _current_pose_and_gripper_from_env(env, task_mode: str, single_arm: Optional[str]):
    raw = env.client.get_state()
    if task_mode == "dual":
        poses = {
            "left": _canonical_pose(raw["state"]["left"]["tcp_pose"]),
            "right": _canonical_pose(raw["state"]["right"]["tcp_pose"]),
        }
        grippers = {
            "left": float(np.asarray(raw["state"]["left"]["gripper_pose"], dtype=np.float32).reshape(-1)[0]),
            "right": float(np.asarray(raw["state"]["right"]["gripper_pose"], dtype=np.float32).reshape(-1)[0]),
        }
        return poses, grippers
    arm = single_arm or env.arm
    poses = {arm: _canonical_pose(raw["state"][arm]["tcp_pose"] if "state" in raw and arm in raw["state"] else raw["state"][env.arm]["tcp_pose"])}
    grippers = {arm: float(np.asarray(raw["state"][env.arm]["gripper_pose"], dtype=np.float32).reshape(-1)[0])}
    return poses, grippers


def _send_state_targets(env, task_mode: str, single_arm: Optional[str], targets: Dict[str, np.ndarray], grippers: Dict[str, float]):
    start_time = time.time()
    if task_mode == "dual":
        left_payload = {
            "pose_target": _clip_pose(env, "left", targets["left"]).tolist(),
            "preset": env.config.DEFAULT_PRESET,
        }
        right_payload = {
            "pose_target": _clip_pose(env, "right", targets["right"]).tolist(),
            "preset": env.config.DEFAULT_PRESET,
        }
        if not bool(env.config.FIX_GRIPPER_OPEN):
            if "left" in grippers:
                left_payload["gripper"] = float(grippers["left"])
            if "right" in grippers:
                right_payload["gripper"] = float(grippers["right"])
        env.client.post_action(
            {
                "mode": env.config.DEFAULT_MODE,
                "owner": "policy",
                "left": left_payload,
                "right": right_payload,
            }
        )
        env.commanded_pose["left"] = np.asarray(left_payload["pose_target"], dtype=np.float32)
        env.commanded_pose["right"] = np.asarray(right_payload["pose_target"], dtype=np.float32)
    else:
        arm = single_arm or env.arm
        payload = {
            "pose_target": _clip_pose(env, arm, targets[arm]).tolist(),
            "preset": env.config.DEFAULT_PRESET,
        }
        if not bool(env.config.FIX_GRIPPER_OPEN) and arm in grippers:
            payload["gripper"] = float(grippers[arm])
        env.client.post_action(
            {
                "mode": env.config.DEFAULT_MODE,
                "owner": "policy",
                env.arm: payload,
            }
        )
        env.commanded_pose = np.asarray(payload["pose_target"], dtype=np.float32)

    env._step_sleep(start_time)
    return _current_pose_and_gripper_from_env(env, task_mode, single_arm)


def _debug_info_to_maps(info: Dict, task_mode: str, single_arm: Optional[str]):
    if task_mode == "dual":
        reference = {
            "left": np.asarray(info["debug_left_reference_pose"], dtype=np.float32),
            "right": np.asarray(info["debug_right_reference_pose"], dtype=np.float32),
        }
        unclipped = {
            "left": np.asarray(info["debug_left_unclipped_target_pose"], dtype=np.float32),
            "right": np.asarray(info["debug_right_unclipped_target_pose"], dtype=np.float32),
        }
        target = {
            "left": np.asarray(info["debug_left_target_pose"], dtype=np.float32),
            "right": np.asarray(info["debug_right_target_pose"], dtype=np.float32),
        }
        actual = {
            "left": np.asarray(info["debug_left_next_pose"], dtype=np.float32),
            "right": np.asarray(info["debug_right_next_pose"], dtype=np.float32),
        }
        return reference, unclipped, target, actual
    arm = single_arm or "arm"
    reference = {arm: np.asarray(info["debug_reference_pose"], dtype=np.float32)}
    target = {arm: np.asarray(info["debug_target_pose"], dtype=np.float32)}
    actual = {arm: np.asarray(info["debug_next_pose"], dtype=np.float32)}
    return reference, target.copy(), target, actual


def run_online(
    env,
    trajectories: List[List[Dict]],
    selected_indices: Iterable[int],
    field_slices: Dict[str, slice],
    task_mode: str,
    single_arm: Optional[str],
    replay_mode: str,
    start_step: int,
    max_steps: Optional[int],
    reset_before: bool,
    reset_after: bool,
    reset_wait_sec: float,
    log_every: int,
) -> List[Dict[str, object]]:
    selected_indices = list(selected_indices)
    if len(selected_indices) != 1:
        raise ValueError("Online replay supports exactly one trajectory. Use --trajectory_index, not --all_trajectories.")
    traj_idx = selected_indices[0]
    trajectory = trajectories[traj_idx]

    if reset_before:
        print("Resetting robot before replay...")
        env.reset()
        time.sleep(max(0.0, reset_wait_sec))

    rows: List[Dict[str, object]] = []
    stop = len(trajectory) if max_steps is None else min(len(trajectory), start_step + max_steps)
    print(f"Online replay trajectory={traj_idx}, steps={stop - start_step}, mode={replay_mode}")
    for step_idx in range(start_step, stop):
        transition = trajectory[step_idx]
        recorded_current = _extract_pose_map(transition["observations"], field_slices, task_mode, single_arm)
        recorded_next = _extract_pose_map(transition["next_observations"], field_slices, task_mode, single_arm)
        recorded_current_gripper = _extract_gripper_map(transition["observations"], field_slices, task_mode, single_arm)
        recorded_next_gripper = _extract_gripper_map(transition["next_observations"], field_slices, task_mode, single_arm)
        actions = _action_map(transition["actions"], task_mode, single_arm)
        before_poses, before_grippers = _current_pose_and_gripper_from_env(env, task_mode, single_arm)

        if replay_mode == "action":
            action_array = np.asarray(transition["actions"], dtype=np.float32)
            _, _, _, _, info = env.step(action_array)
            reference, unclipped_targets, targets, actual_poses = _debug_info_to_maps(info, task_mode, single_arm)
            _, actual_grippers = _current_pose_and_gripper_from_env(env, task_mode, single_arm)
            target_grippers = {
                arm: _gripper_target_from_action(env, arm, action)
                for arm, action in actions.items()
            }
        else:
            reference = before_poses
            unclipped_targets = {arm: pose.copy() for arm, pose in recorded_next.items()}
            targets = {arm: _clip_pose(env, arm, pose) for arm, pose in recorded_next.items()}
            target_grippers = recorded_next_gripper.copy()
            actual_poses, actual_grippers = _send_state_targets(
                env,
                task_mode,
                single_arm,
                targets,
                target_grippers,
            )

        base = _row_base(traj_idx, step_idx, "online", replay_mode, "", transition)
        for arm in targets:
            _append_arm_row(
                rows,
                base,
                arm,
                actions.get(arm),
                recorded_current.get(arm),
                recorded_next.get(arm),
                reference.get(arm),
                unclipped_targets.get(arm),
                targets.get(arm),
                actual_poses.get(arm),
                recorded_current_gripper.get(arm),
                recorded_next_gripper.get(arm),
                target_grippers.get(arm),
                actual_grippers.get(arm),
            )

        if (step_idx - start_step) % max(1, log_every) == 0 or step_idx == stop - 1:
            latest = rows[-(2 if task_mode == "dual" else 1):]
            summary = " ".join(
                f"{row['arm']}:track={row['actual_to_target_pos_err_m']:.4f}m "
                f"recorded={row['actual_to_recorded_pos_err_m']:.4f}m"
                for row in latest
            )
            print(f"[online-replay] step={step_idx + 1}/{stop} {summary}")

    if reset_after:
        print("Resetting robot after replay...")
        env.reset()
        time.sleep(max(0.0, reset_wait_sec))
    return rows


def _write_csv(rows: List[Dict[str, object]], output_csv: Path):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote replay errors CSV to {output_csv}")


def _summary(rows: List[Dict[str, object]]) -> Dict[str, object]:
    metrics = defaultdict(list)
    for row in rows:
        arm = str(row["arm"])
        for key in (
            "actual_to_target_pos_err_m",
            "actual_to_target_ori_err_rad",
            "target_to_recorded_pos_err_m",
            "target_to_recorded_ori_err_rad",
            "actual_to_recorded_pos_err_m",
            "actual_to_recorded_ori_err_rad",
            "clip_pos_m",
            "clip_ori_rad",
        ):
            value = row.get(key)
            if value == "" or value is None:
                continue
            value = float(value)
            if np.isfinite(value):
                metrics[(arm, key)].append(value)

    result = {"num_rows": len(rows), "arms": {}}
    for (arm, key), values in metrics.items():
        arr = np.asarray(values, dtype=np.float64)
        result["arms"].setdefault(arm, {})[key] = {
            "mean": float(np.mean(arr)),
            "max": float(np.max(arr)),
            "q50": float(np.quantile(arr, 0.50)),
            "q90": float(np.quantile(arr, 0.90)),
            "q95": float(np.quantile(arr, 0.95)),
            "q99": float(np.quantile(arr, 0.99)),
        }
    return result


def _write_npz(rows: List[Dict[str, object]], output_npz: Path):
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    numeric = defaultdict(list)
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (int, float, bool, np.integer, np.floating, np.bool_)):
                numeric[key].append(float(value))
    np.savez_compressed(output_npz, **{key: np.asarray(values) for key, values in numeric.items()})
    print(f"Wrote replay numeric NPZ to {output_npz}")


def _write_summary_json(summary: Dict[str, object], output_json: Path):
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Wrote replay summary JSON to {output_json}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Validate or replay ConRFT transition actions/states for R1Lite tasks. "
            "The task layout is read from CONFIG_MAPPING[exp_name]."
        )
    )
    parser.add_argument("--exp_name", default="r1lite_reach_target", help="Target experiment name.")
    parser.add_argument("--input_file", required=True, help="Path to transition .pkl file.")
    parser.add_argument("--trajectory_index", type=int, default=0, help="Trajectory index after splitting by dones.")
    parser.add_argument("--all_trajectories", action="store_true", help="Offline only: process every trajectory.")
    parser.add_argument("--list_only", action="store_true", help="Only print trajectory summary.")
    parser.add_argument("--exec_mode", choices=("offline", "online"), default="offline")
    parser.add_argument("--replay_mode", choices=("action", "state"), default="action")
    parser.add_argument(
        "--offline_reference",
        choices=("teacher_forced", "rollout"),
        default="teacher_forced",
        help="Offline action replay reference: recorded current state each step, or integrated predicted state.",
    )
    parser.add_argument("--start_step", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--no_reset_before", action="store_true", help="Online only: skip reset before replay.")
    parser.add_argument("--no_reset_after", action="store_true", help="Online only: skip reset after replay.")
    parser.add_argument("--reset_wait_sec", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--output_csv", default=None, help="Optional path for per-step error CSV.")
    parser.add_argument("--output_npz", default=None, help="Optional path for numeric error arrays.")
    parser.add_argument("--output_summary_json", default=None, help="Optional path for summary JSON.")
    args = parser.parse_args()

    input_file = Path(args.input_file).expanduser().resolve()
    transitions = _load_transitions(input_file)
    trajectories = _split_trajectories(transitions)
    _print_summary(trajectories)
    if args.list_only:
        return

    if args.exec_mode == "online" and args.all_trajectories:
        raise ValueError("--all_trajectories is only supported for --exec_mode=offline")

    cfg, env, task_mode, single_arm, proprio_keys = _make_env(args.exp_name)
    selected_order, field_slices = _select_field_slices(
        transitions,
        proprio_keys,
        task_mode,
        single_arm,
    )
    selected = _trajectory_indices(trajectories, args.trajectory_index, args.all_trajectories)
    print(f"Task mode: {task_mode}, single_arm={single_arm}")
    print(f"State layout: gym_sorted, proprio_order={selected_order}")
    print(f"Replay mode: exec={args.exec_mode}, replay={args.replay_mode}")

    try:
        if args.exec_mode == "offline":
            rows = run_offline(
                env,
                trajectories,
                selected,
                field_slices,
                task_mode,
                single_arm,
                args.replay_mode,
                args.offline_reference,
                args.start_step,
                args.max_steps,
            )
        else:
            print(
                "[replay] online mode sends owner=policy commands. "
                "Make sure other controllers have released robot ownership."
            )
            rows = run_online(
                env,
                trajectories,
                selected,
                field_slices,
                task_mode,
                single_arm,
                args.replay_mode,
                args.start_step,
                args.max_steps,
                reset_before=not args.no_reset_before,
                reset_after=not args.no_reset_after,
                reset_wait_sec=args.reset_wait_sec,
                log_every=args.log_every,
            )
    finally:
        env.close()

    summary = _summary(rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.output_csv:
        _write_csv(rows, Path(args.output_csv).expanduser().resolve())
    if args.output_npz:
        _write_npz(rows, Path(args.output_npz).expanduser().resolve())
    if args.output_summary_json:
        _write_summary_json(summary, Path(args.output_summary_json).expanduser().resolve())


if __name__ == "__main__":
    main()
