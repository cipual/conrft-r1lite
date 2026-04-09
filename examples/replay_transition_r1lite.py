#!/usr/bin/env python3

import argparse
import os
import pickle
import time
from pathlib import Path
from typing import Dict, List, Tuple

import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation


def _ensure_examples_on_path():
    """允许独立脚本直接导入 experiments / r1lite_env。"""
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
from r1lite_env import R1LiteArmEnv  # noqa: E402


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


def _tcp_pose_error(actual_pose: np.ndarray, expected_pose: np.ndarray) -> Dict[str, float]:
    return {
        "position_error_m": float(np.linalg.norm(actual_pose[:3] - expected_pose[:3])),
        "orientation_error_rad": _quat_angle_error_rad(actual_pose[3:], expected_pose[3:]),
    }


def _infer_flatten_slices(proprio_keys: List[str]) -> Dict[str, slice]:
    proprio_space = gym.spaces.Dict(
        {
            key: gym.spaces.Box(-np.inf, np.inf, shape=_PROPRIO_FIELD_SHAPES[key], dtype=np.float32)
            for key in proprio_keys
        }
    )
    slices: Dict[str, slice] = {}
    for idx, key in enumerate(proprio_keys):
        sample = {
            k: np.zeros(_PROPRIO_FIELD_SHAPES[k], dtype=np.float32)
            for k in proprio_keys
        }
        # 给每个字段填独特的正数，反推出 flatten 后的切片位置。
        sample[key] = np.arange(1, np.prod(_PROPRIO_FIELD_SHAPES[key]) + 1, dtype=np.float32).reshape(_PROPRIO_FIELD_SHAPES[key])
        flat = gym.spaces.flatten(proprio_space, sample)
        nz = np.flatnonzero(flat)
        if nz.size == 0:
            raise RuntimeError(f"Failed to infer flattened slice for key {key}")
        slices[key] = slice(int(nz[0]), int(nz[-1]) + 1)
    return slices


def _extract_tcp_pose_from_flat_state(flat_state: np.ndarray, field_slices: Dict[str, slice]) -> np.ndarray:
    return np.asarray(flat_state[field_slices["tcp_pose"]], dtype=np.float32)


def _extract_expected_next_pose(transition: Dict, field_slices: Dict[str, slice]) -> np.ndarray:
    flat_state = np.asarray(transition["next_observations"]["state"][-1], dtype=np.float32)
    return _extract_tcp_pose_from_flat_state(flat_state, field_slices)


def _validate_supported_transition_format(transitions: List[Dict], field_slices: Dict[str, slice]):
    """
    replay 现在只支持统一后的正确格式：
    - `state` 必须与在线 demo 一样，使用 SERLObsWrapper / gym.flatten 的编码顺序
    旧版官方转换 pkl（手工拼接 state）需要先重新转换。
    """
    if not transitions:
        raise ValueError("Transition list is empty")

    sample = transitions[0]
    flat_state = np.asarray(sample["next_observations"]["state"][-1], dtype=np.float32)
    tcp_pose = _extract_tcp_pose_from_flat_state(flat_state, field_slices)
    quat = np.asarray(tcp_pose[3:], dtype=np.float32)
    if float(np.linalg.norm(quat)) < 1e-8:
        source = str(sample.get("infos", {}).get("conversion_source", "unknown"))
        raise ValueError(
            "Unsupported transition state layout for replay. "
            "This replay script now only supports the unified flattened-state format "
            f"(SERLObsWrapper-compatible). Detected source={source!r} with invalid tcp_pose slice. "
            "If this is an older official-teleop converted pkl, please re-run "
            "convert_official_teleop_to_conrft_demo.py with the latest version."
        )


def _make_base_env(exp_name: str) -> Tuple[R1LiteArmEnv, List[str], Dict[str, slice]]:
    if exp_name not in CONFIG_MAPPING:
        raise ValueError(f"Unknown experiment: {exp_name}")
    cfg = CONFIG_MAPPING[exp_name]()
    # fake_env=True 时不会挂 teleop wrapper，但仍能复用同一个任务配置来源。
    wrapped_env = cfg.get_environment(fake_env=True, save_video=False, classifier=False, stack_obs_num=2)
    base_env = wrapped_env.unwrapped
    if not isinstance(base_env, R1LiteArmEnv):
        raise TypeError(f"{exp_name} does not resolve to a single-arm R1LiteArmEnv")
    proprio_keys = list(getattr(cfg, "proprio_keys", []))
    field_slices = _infer_flatten_slices(proprio_keys)
    return base_env, proprio_keys, field_slices


def _print_summary(trajectories: List[List[Dict]]):
    print(f"Found {len(trajectories)} trajectory(ies)")
    for idx, trajectory in enumerate(trajectories):
        first = trajectory[0]
        last = trajectory[-1]
        succeed = bool(last.get("infos", {}).get("succeed", False))
        reward_sum = float(sum(float(t.get("rewards", 0.0)) for t in trajectory))
        action_norms = np.linalg.norm(
            np.stack([np.asarray(t["actions"], dtype=np.float32)[:6] for t in trajectory], axis=0),
            axis=1,
        )
        nonzero_idx = np.flatnonzero(action_norms > 1e-6)
        first_nonzero = None if nonzero_idx.size == 0 else int(nonzero_idx[0])
        print(
            f"  trajectory[{idx}]: steps={len(trajectory)}, "
            f"succeed={succeed}, return={reward_sum:.3f}, "
            f"first_nonzero_action_step={first_nonzero}"
        )


def replay_trajectory(
    env: R1LiteArmEnv,
    field_slices: Dict[str, slice],
    trajectory: List[Dict],
    replay_mode: str,
    reset_before: bool,
    reset_after: bool,
    reset_wait_sec: float,
    log_every: int,
):
    if reset_before:
        print("Resetting robot before replay...")
        env.reset()
        time.sleep(max(0.0, reset_wait_sec))

    print(f"Replaying {len(trajectory)} step(s)...")
    last_obs = None
    for idx, transition in enumerate(trajectory):
        expected_next_pose = _extract_expected_next_pose(transition, field_slices)
        action = np.asarray(transition["actions"], dtype=np.float32)

        if replay_mode == "action":
            obs, _, done, _, _ = env.step(action)
            last_obs = obs
        else:
            start_time = time.time()
            pose_target = env._clip_pose_to_safety_box(env.arm, expected_next_pose).tolist()
            side_payload = {
                "pose_target": pose_target,
                "preset": env.config.DEFAULT_PRESET,
            }
            if not bool(env.config.FIX_GRIPPER_OPEN):
                side_payload["gripper"] = float(np.clip((action[6] + 1.0) * 50.0, 0.0, 100.0))
            env.client.post_action(
                {
                    "mode": env.config.DEFAULT_MODE,
                    "owner": "policy",
                    env.arm: side_payload,
                }
            )
            env.commanded_pose = np.asarray(expected_next_pose, dtype=np.float32).copy()
            env._step_sleep(start_time)
            env._maybe_log_effective_hz(f"replay-pose:{env.arm}")
            raw = env.client.get_state()
            obs = env._extract_obs(raw)
            last_obs = obs
            done = bool(idx == len(trajectory) - 1)

        actual_pose = np.asarray(last_obs["state"]["tcp_pose"], dtype=np.float32)
        pose_error = _tcp_pose_error(actual_pose, expected_next_pose)
        if idx % max(1, log_every) == 0 or idx == len(trajectory) - 1:
            print(
                f"[replay] step={idx + 1}/{len(trajectory)} "
                f"mode={replay_mode} "
                f"action_norm={float(np.linalg.norm(action[:6])):.4f} "
                f"pos_err={pose_error['position_error_m']:.4f}m "
                f"ori_err={pose_error['orientation_error_rad']:.4f}rad "
                f"env_done={bool(done)}"
            )

    if last_obs is not None:
        final_pose = np.asarray(last_obs["state"]["tcp_pose"], dtype=np.float32)
        print(
            "[replay] final tcp_pose="
            f"{np.array2string(final_pose, precision=4, separator=', ')}"
        )

    if reset_after:
        print("Resetting robot after replay...")
        env.reset()
        time.sleep(max(0.0, reset_wait_sec))


def main():
    parser = argparse.ArgumentParser(description="Replay one trajectory from a ConRFT transition .pkl on R1Lite.")
    parser.add_argument("--exp_name", default="r1lite_reach_target", help="Target experiment name.")
    parser.add_argument("--input_file", required=True, help="Path to transition .pkl file.")
    parser.add_argument("--trajectory_index", type=int, default=0, help="Which trajectory to replay after splitting by done.")
    parser.add_argument("--list_only", action="store_true", help="Only print trajectory summary, do not move the robot.")
    parser.add_argument("--no_reset_before", action="store_true", help="Skip reset before replay.")
    parser.add_argument("--no_reset_after", action="store_true", help="Skip reset after replay.")
    parser.add_argument("--reset_wait_sec", type=float, default=1.0, help="Extra sleep after reset before continuing.")
    parser.add_argument("--log_every", type=int, default=10, help="Print replay status every N steps.")
    parser.add_argument(
        "--replay_mode",
        choices=("pose_target", "action"),
        default="action",
        help="Replay by recorded normalized action (default) or by absolute saved next tcp_pose.",
    )
    args = parser.parse_args()

    input_file = Path(args.input_file).expanduser().resolve()
    transitions = _load_transitions(input_file)
    trajectories = _split_trajectories(transitions)
    _print_summary(trajectories)

    if args.list_only:
        return

    if args.trajectory_index < 0 or args.trajectory_index >= len(trajectories):
        raise IndexError(
            f"trajectory_index={args.trajectory_index} is out of range for {len(trajectories)} trajectories"
        )

    env, proprio_keys, field_slices = _make_base_env(args.exp_name)
    _validate_supported_transition_format(transitions, field_slices)
    print(f"Replay proprio order: {proprio_keys}")
    print(
        "[replay] make sure official teleop / other controllers have released robot ownership; "
        "this script sends commands with owner=policy"
    )
    try:
        replay_trajectory(
            env,
            field_slices,
            trajectories[args.trajectory_index],
            replay_mode=args.replay_mode,
            reset_before=not args.no_reset_before,
            reset_after=not args.no_reset_after,
            reset_wait_sec=args.reset_wait_sec,
            log_every=args.log_every,
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
