#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import sys
import termios
import time
import tty
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import requests
from scipy.spatial.transform import Rotation


def ensure_repo_paths() -> Path:
    root = Path(__file__).resolve()
    repo_root = root.parents[3]
    paths = (
        repo_root / "examples",
        repo_root / "examples/debug/r1lite_reach_target/consistance",
        repo_root / "serl_robot_infra",
    )
    for path in paths:
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    return repo_root


REPO_ROOT = ensure_repo_paths()

from debug_rosbag_utils import build_debug_frames  # noqa: E402
from decode_raw_mcap import resolve_raw_episode_dirs  # noqa: E402


class R1LiteClient:
    def __init__(self, server_url: str, timeout: float = 2.0):
        self.server_url = server_url.rstrip("/") + "/"
        self.timeout = timeout
        self.session = requests.Session()
        self.session.trust_env = False

    def _url(self, path: str) -> str:
        return self.server_url + path.lstrip("/")

    def get_state(self) -> Dict[str, Any]:
        response = self.session.get(self._url("state"), timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def post_action(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        response = self.session.post(self._url("action"), json=payload, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def close(self) -> None:
        self.session.close()


ANSI_RESET = "\033[0m"
ANSI_GREEN = "\033[92m"
ANSI_YELLOW = "\033[93m"
ANSI_RED = "\033[91m"
ANSI_CYAN = "\033[96m"
ANSI_MAGENTA = "\033[95m"
ANSI_BOLD = "\033[1m"


def colorize(text: str, color: str) -> str:
    return f"{color}{text}{ANSI_RESET}"


def read_single_key(prompt: str) -> str:
    print(prompt, end="", flush=True)
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        key = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    print(key)
    return key.lower()


def quat_angle_error_rad(current_xyzw: np.ndarray, target_xyzw: np.ndarray) -> float:
    if float(np.linalg.norm(current_xyzw)) < 1e-8 or float(np.linalg.norm(target_xyzw)) < 1e-8:
        return float("nan")
    current = Rotation.from_quat(current_xyzw)
    target = Rotation.from_quat(target_xyzw)
    delta = current.inv() * target
    return float(np.linalg.norm(delta.as_rotvec()))


def rotvec_delta(current_xyzw: np.ndarray, next_xyzw: np.ndarray) -> np.ndarray:
    current = Rotation.from_quat(current_xyzw)
    nxt = Rotation.from_quat(next_xyzw)
    return (current.inv() * nxt).as_rotvec().astype(np.float32)


def target_pose_from_eef_delta(reference_pose: np.ndarray, eef_delta: np.ndarray) -> np.ndarray:
    target = np.asarray(reference_pose, dtype=np.float32).copy()
    target[:3] += np.asarray(eef_delta[:3], dtype=np.float32)
    target[3:] = (Rotation.from_quat(reference_pose[3:]) * Rotation.from_rotvec(eef_delta[3:6])).as_quat()
    return target.astype(np.float32)


def arms_for_replay(arm: str) -> List[str]:
    return ["left", "right"] if arm == "dual" else [arm]


def align_dual_frames(left_frames: List[Dict[str, Any]], right_frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    left_ts = np.asarray([int(frame["timestamp_ns"]) for frame in left_frames], dtype=np.int64)
    right_ts = np.asarray([int(frame["timestamp_ns"]) for frame in right_frames], dtype=np.int64)
    start_ns = max(int(left_ts[0]), int(right_ts[0]))
    end_ns = min(int(left_ts[-1]), int(right_ts[-1]))
    if end_ns <= start_ns:
        raise ValueError("No overlapping left/right time interval.")

    aligned: List[Dict[str, Any]] = []
    right_idx = 0
    for left_idx, ts in enumerate(left_ts):
        if ts < start_ns or ts > end_ns:
            continue
        while right_idx + 1 < len(right_ts) and abs(int(right_ts[right_idx + 1]) - int(ts)) <= abs(
            int(right_ts[right_idx]) - int(ts)
        ):
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
        raise ValueError("Need at least two aligned left/right samples.")
    return aligned


def load_raw_trajectories(input_dir: Path, arm: str, control_hz: float) -> List[List[Dict[str, Any]]]:
    episodes = resolve_raw_episode_dirs([str(input_dir)], recursive=False)
    trajectories: List[List[Dict[str, Any]]] = []
    for episode_dir in episodes:
        if arm == "dual":
            left = build_debug_frames(episode_dir, "left", control_hz, require_arm_target=False)
            right = build_debug_frames(episode_dir, "right", control_hz, require_arm_target=False)
            trajectories.append(align_dual_frames(left, right))
        else:
            trajectories.append(build_debug_frames(episode_dir, arm, control_hz, require_arm_target=False))
    return trajectories


def frame_for_arm(frame: Dict[str, Any], arm: str) -> Dict[str, Any]:
    if arm in frame and isinstance(frame[arm], dict):
        return frame[arm]
    return frame


def gripper_value(frame: Dict[str, Any]) -> float:
    return float(np.asarray(frame.get("gripper_pose", [0.0]), dtype=np.float32).reshape(-1)[0])


def current_robot_maps(client: R1LiteClient, arms: Iterable[str]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, float]]:
    raw = client.get_state()
    pose_map = {}
    joint_map = {}
    gripper_map = {}
    for arm in arms:
        arm_state = raw["state"][arm]
        pose_map[arm] = np.asarray(arm_state["tcp_pose"], dtype=np.float32).reshape(-1)
        joint_map[arm] = np.asarray(arm_state["joint_pos"], dtype=np.float32).reshape(-1)[:6]
        gripper_map[arm] = float(np.asarray(arm_state["gripper_pose"], dtype=np.float32).reshape(-1)[0])
    return pose_map, joint_map, gripper_map


def send_pose_targets(
    client: R1LiteClient,
    targets: Dict[str, np.ndarray],
    grippers: Dict[str, float],
    mode: str,
    owner: str,
    preset: str,
) -> None:
    payload: Dict[str, Any] = {"mode": mode, "owner": owner}
    for arm, pose in targets.items():
        payload[arm] = {"pose_target": np.asarray(pose, dtype=np.float32).reshape(-1)[:7].tolist(), "preset": preset}
        if arm in grippers:
            payload[arm]["gripper"] = float(grippers[arm])
    client.post_action(payload)


def send_joint_targets(
    client: R1LiteClient,
    targets: Dict[str, np.ndarray],
    grippers: Dict[str, float],
    mode: str,
    owner: str,
    preset: str,
) -> None:
    payload: Dict[str, Any] = {"mode": mode, "owner": owner}
    for arm, joint in targets.items():
        payload[arm] = {"joint_target": np.asarray(joint, dtype=np.float32).reshape(-1)[:6].tolist(), "preset": preset}
        if arm in grippers:
            payload[arm]["gripper"] = float(grippers[arm])
    client.post_action(payload)


def debug_error_color(pos_err_m: float, ori_err_rad: float, pos_thresh: float, ori_thresh: float) -> str:
    if pos_err_m <= pos_thresh and ori_err_rad <= ori_thresh:
        return ANSI_GREEN
    if pos_err_m <= 2.0 * pos_thresh and ori_err_rad <= 2.0 * ori_thresh:
        return ANSI_YELLOW
    return ANSI_RED


def print_eef_action_debug(step_idx: int, current: Dict[str, Any], nxt: Dict[str, Any], arms: List[str]) -> bool:
    print(colorize(f"\n[raw-eef-debug] step={step_idx}", ANSI_BOLD + ANSI_CYAN))
    for arm in arms:
        cur = frame_for_arm(current, arm)
        nxt_arm = frame_for_arm(nxt, arm)
        delta_xyz = np.asarray(nxt_arm["tcp_pose"][:3], dtype=np.float32) - np.asarray(cur["tcp_pose"][:3], dtype=np.float32)
        delta_rpy = rotvec_delta(np.asarray(cur["tcp_pose"][3:], dtype=np.float32), np.asarray(nxt_arm["tcp_pose"][3:], dtype=np.float32))
        grip = gripper_value(nxt_arm)
        header_color = ANSI_GREEN if arm == "left" else ANSI_MAGENTA
        print(
            colorize(f"  action:{arm:<5}", header_color)
            + f" xyz={np.array2string(delta_xyz, precision=4, suppress_small=True)} "
            + f"rpy={np.array2string(delta_rpy, precision=4, suppress_small=True)} "
            + f"grip={grip:.3f}"
        )
    while True:
        key = read_single_key("[raw-eef-debug] press 's' to execute one step, 'q' to quit: ")
        if key == "s":
            return True
        if key == "q":
            return False


def print_joint_action_debug(step_idx: int, current: Dict[str, Any], nxt: Dict[str, Any], arms: List[str]) -> bool:
    print(colorize(f"\n[raw-joint-debug] step={step_idx}", ANSI_BOLD + ANSI_CYAN))
    for arm in arms:
        cur = frame_for_arm(current, arm)
        nxt_arm = frame_for_arm(nxt, arm)
        delta = np.asarray(nxt_arm["joint_pos"], dtype=np.float32) - np.asarray(cur["joint_pos"], dtype=np.float32)
        grip = gripper_value(nxt_arm)
        header_color = ANSI_GREEN if arm == "left" else ANSI_MAGENTA
        print(
            colorize(f"  action:{arm:<5}", header_color)
            + f" joint_delta={np.array2string(delta, precision=4, suppress_small=True)} "
            + f"grip={grip:.3f}"
        )
    while True:
        key = read_single_key("[raw-joint-debug] press 's' to execute one step, 'q' to quit: ")
        if key == "s":
            return True
        if key == "q":
            return False


def print_exec_debug(
    prefix: str,
    arm: str,
    reference_pose: np.ndarray,
    target_pose: np.ndarray,
    actual_pose: np.ndarray,
    pos_thresh: float,
    ori_thresh: float,
) -> Dict[str, Any]:
    cmd_delta = target_pose[:3] - reference_pose[:3]
    actual_delta = actual_pose[:3] - reference_pose[:3]
    pos_err = float(np.linalg.norm(actual_pose[:3] - target_pose[:3]))
    ori_err = quat_angle_error_rad(actual_pose[3:7], target_pose[3:7])
    color = debug_error_color(pos_err, ori_err, pos_thresh, ori_thresh)
    header_color = ANSI_GREEN if arm == "left" else ANSI_MAGENTA
    print(colorize(f"  [{prefix}:{arm}-exec]", ANSI_BOLD + header_color))
    print(
        f"    ref_xyz    = {np.array2string(reference_pose[:3], precision=4, suppress_small=True)}\n"
        f"    target_xyz = {np.array2string(target_pose[:3], precision=4, suppress_small=True)}\n"
        f"    next_xyz   = {np.array2string(actual_pose[:3], precision=4, suppress_small=True)}"
    )
    print(
        f"    cmd_dxyz   = {np.array2string(cmd_delta, precision=4, suppress_small=True)}\n"
        f"    actual_dxyz= {np.array2string(actual_delta, precision=4, suppress_small=True)}\n"
        f"    clip_dxyz  = {np.array2string(np.zeros(3, dtype=np.float32), precision=4, suppress_small=True)}"
    )
    print(colorize(f"    target_err = pos={pos_err:.4f} m  ori={ori_err:.4f} rad", color))
    return {"label": arm, "target_pose": np.asarray(target_pose, dtype=np.float32), "pos_err_m": pos_err, "ori_err_rad": ori_err}


def wait_until_small_error(
    client: R1LiteClient,
    prefix: str,
    arms: List[str],
    metrics: List[Dict[str, Any]],
    pos_thresh: float,
    ori_thresh: float,
    poll_sec: float,
) -> bool:
    print(
        colorize(
            f"[{prefix}] blocking until execution error is small enough "
            f"(pos<={pos_thresh:.4f}m, ori<={ori_thresh:.4f}rad)",
            ANSI_BOLD + ANSI_YELLOW,
        )
    )
    while True:
        key = read_single_key(f"[{prefix}] press 'c' to check error, 's' to skip wait, 'q' to quit: ")
        if key == "q":
            return False
        if key == "s":
            return True
        if key != "c":
            continue
        pose_map, _, _ = current_robot_maps(client, arms)
        all_small = True
        for metric in metrics:
            arm = str(metric["label"])
            target = np.asarray(metric["target_pose"], dtype=np.float32)
            actual = pose_map[arm]
            pos_err = float(np.linalg.norm(actual[:3] - target[:3]))
            ori_err = quat_angle_error_rad(actual[3:7], target[3:7])
            color = debug_error_color(pos_err, ori_err, pos_thresh, ori_thresh)
            print(
                colorize(
                    (
                        f"  [{prefix}:{arm}-wait]\n"
                        f"    target_xyz = {np.array2string(target[:3], precision=4, suppress_small=True)}\n"
                        f"    actual_xyz = {np.array2string(actual[:3], precision=4, suppress_small=True)}\n"
                        f"    pos_err    = ||actual_xyz - target_xyz|| = {pos_err:.4f} m\n"
                        f"    ori_err    = {ori_err:.4f} rad"
                    ),
                    color,
                )
            )
            if pos_err > pos_thresh or ori_err > ori_thresh:
                all_small = False
        if all_small:
            print(colorize(f"[{prefix}] error gate passed", ANSI_BOLD + ANSI_GREEN))
            return True
        time.sleep(max(0.0, poll_sec))


def add_common_args(parser: argparse.ArgumentParser, default_input: Path) -> None:
    parser.add_argument("--input_dir", default=str(default_input), help="RAW episode dir or parent dir containing *_RAW episodes.")
    parser.add_argument("--trajectory_index", type=int, default=0)
    parser.add_argument("--arm", choices=("left", "right", "dual"), default="dual")
    parser.add_argument("--server_url", default="http://192.168.12.12:8001/")
    parser.add_argument("--exec_mode", choices=("offline", "online"), default="online")
    parser.add_argument("--replay_mode", choices=("action", "state"), default="action")
    parser.add_argument("--start_exec_step", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--online_start_mode", choices=("move_to_recorded", "current"), default="move_to_recorded")
    parser.add_argument("--control_hz", type=float, default=10.0)
    parser.add_argument("--mode", default="ee_pose_servo", choices=("ee_pose_servo",))
    parser.add_argument("--preset", default="free_space")
    parser.add_argument("--owner", default="policy")
    parser.add_argument("--reset_wait_sec", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug_block_until_error_small", action="store_true")
    parser.add_argument("--debug_block_pos_err_m", type=float, default=0.01)
    parser.add_argument("--debug_block_ori_err_rad", type=float, default=0.10)
    parser.add_argument("--debug_block_poll_sec", type=float, default=0.2)
    parser.add_argument("--output_csv", default=None)
    parser.add_argument("--output_summary_json", default=None)


def write_rows(rows: List[Dict[str, Any]], output_csv: Optional[str], output_json: Optional[str]) -> None:
    if output_csv:
        path = Path(output_csv).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames: List[str] = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote CSV to {path}")
    if output_json:
        path = Path(output_json).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump({"rows": rows}, f, ensure_ascii=False, indent=2)
        print(f"Wrote JSON to {path}")
