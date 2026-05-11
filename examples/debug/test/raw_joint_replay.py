#!/usr/bin/env python3

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from raw_replay_common import (
    REPO_ROOT,
    R1LiteClient,
    add_common_args,
    arms_for_replay,
    current_robot_maps,
    frame_for_arm,
    gripper_value,
    load_raw_trajectories,
    print_joint_action_debug,
    quat_angle_error_rad,
    send_joint_targets,
    wait_until_small_error,
    write_rows,
    print_exec_debug,
    colorize,
    ANSI_BOLD,
    ANSI_GREEN,
    ANSI_RED,
    ANSI_YELLOW,
    read_single_key,
)


def _targets_from_frames(
    current_frame: Dict[str, Any],
    next_frame: Dict[str, Any],
    arms: List[str],
    replay_mode: str,
    live_reference: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    targets = {}
    for arm in arms:
        cur = frame_for_arm(current_frame, arm)
        nxt = frame_for_arm(next_frame, arm)
        if replay_mode == "state":
            targets[arm] = np.asarray(nxt["joint_pos"], dtype=np.float32).copy()
        else:
            delta = np.asarray(nxt["joint_pos"], dtype=np.float32) - np.asarray(cur["joint_pos"], dtype=np.float32)
            targets[arm] = (live_reference[arm] + delta).astype(np.float32)
    return targets


def _grippers_from_frame(frame: Dict[str, Any], arms: List[str]) -> Dict[str, float]:
    return {arm: gripper_value(frame_for_arm(frame, arm)) for arm in arms}


def _wait_start_joint_state(
    client: R1LiteClient,
    arms: List[str],
    targets: Dict[str, np.ndarray],
    threshold: float,
    poll_sec: float,
) -> bool:
    print(colorize(f"[raw-joint-debug] waiting for recorded start joint state (joint_err<={threshold:.4f})", ANSI_BOLD + ANSI_YELLOW))
    while True:
        key = read_single_key("[raw-joint-debug] press 'c' to check start joint error, 's' to skip wait, 'q' to quit: ")
        if key == "q":
            return False
        if key == "s":
            return True
        if key != "c":
            continue
        _, actual_joint, _ = current_robot_maps(client, arms)
        all_small = True
        for arm in arms:
            target = np.asarray(targets[arm], dtype=np.float32)
            actual = actual_joint[arm]
            err = float(np.linalg.norm(actual - target))
            color = ANSI_GREEN if err <= threshold else ANSI_RED
            print(
                colorize(
                    (
                        f"  [raw-joint-debug:{arm}-start]\n"
                        f"    target_joint = {np.array2string(target, precision=4, suppress_small=True)}\n"
                        f"    actual_joint = {np.array2string(actual, precision=4, suppress_small=True)}\n"
                        f"    joint_err    = {err:.4f}"
                    ),
                    color,
                )
            )
            if err > threshold:
                all_small = False
        if all_small:
            print(colorize("[raw-joint-debug] start joint gate passed", ANSI_BOLD + ANSI_GREEN))
            return True
        time.sleep(max(0.0, poll_sec))


def _run_offline(trajectory: List[Dict[str, Any]], arms: List[str], start: int, stop: int, replay_mode: str) -> List[Dict[str, Any]]:
    rows = []
    for idx in range(start, stop):
        cur = trajectory[idx]
        nxt = trajectory[idx + 1]
        live_ref = {arm: np.asarray(frame_for_arm(cur, arm)["joint_pos"], dtype=np.float32) for arm in arms}
        targets = _targets_from_frames(cur, nxt, arms, replay_mode, live_ref)
        for arm in arms:
            recorded = np.asarray(frame_for_arm(nxt, arm)["joint_pos"], dtype=np.float32)
            rows.append(
                {
                    "step": idx,
                    "arm": arm,
                    "target_to_recorded_joint_err": float(np.linalg.norm(targets[arm] - recorded)),
                }
            )
    return rows


def _run_online(args: argparse.Namespace, trajectory: List[Dict[str, Any]], arms: List[str], start: int, stop: int) -> List[Dict[str, Any]]:
    client = R1LiteClient(args.server_url)
    rows: List[Dict[str, Any]] = []
    try:
        if args.online_start_mode == "move_to_recorded":
            start_targets = {arm: np.asarray(frame_for_arm(trajectory[start], arm)["joint_pos"], dtype=np.float32) for arm in arms}
            start_grippers = _grippers_from_frame(trajectory[start], arms)
            print(f"[raw-joint-replay] moving to recorded current joint state at step={start}")
            send_joint_targets(client, start_targets, start_grippers, args.mode, args.owner, args.preset)
            time.sleep(max(0.0, args.reset_wait_sec))
            if args.debug_block_until_error_small:
                if not _wait_start_joint_state(
                    client,
                    arms,
                    start_targets,
                    args.debug_block_pos_err_m,
                    args.debug_block_poll_sec,
                ):
                    return rows

        for idx in range(start, stop):
            cur = trajectory[idx]
            nxt = trajectory[idx + 1]
            if args.debug and not print_joint_action_debug(idx, cur, nxt, arms):
                break

            before_pose, before_joint, _ = current_robot_maps(client, arms)
            targets = _targets_from_frames(cur, nxt, arms, args.replay_mode, before_joint)
            grippers = _grippers_from_frame(nxt, arms)
            step_start = time.time()
            send_joint_targets(client, targets, grippers, args.mode, args.owner, args.preset)
            time.sleep(max(0.0, (1.0 / max(args.control_hz, 1e-6)) - (time.time() - step_start)))
            actual_pose, actual_joint, actual_grippers = current_robot_maps(client, arms)

            metrics = []
            for arm in arms:
                recorded_joint = np.asarray(frame_for_arm(nxt, arm)["joint_pos"], dtype=np.float32)
                recorded_pose = np.asarray(frame_for_arm(nxt, arm)["tcp_pose"], dtype=np.float32)
                rows.append(
                    {
                        "step": idx,
                        "arm": arm,
                        "replay_mode": args.replay_mode,
                        "actual_to_target_joint_err": float(np.linalg.norm(actual_joint[arm] - targets[arm])),
                        "actual_to_recorded_joint_err": float(np.linalg.norm(actual_joint[arm] - recorded_joint)),
                        "actual_to_recorded_pos_err_m": float(np.linalg.norm(actual_pose[arm][:3] - recorded_pose[:3])),
                        "actual_to_recorded_ori_err_rad": quat_angle_error_rad(actual_pose[arm][3:7], recorded_pose[3:7]),
                        "actual_gripper": actual_grippers[arm],
                        "recorded_gripper": gripper_value(frame_for_arm(nxt, arm)),
                    }
                )
                if args.debug:
                    # For joint replay, target TCP pose is unknown until the robot/IK executes; use recorded pose for the wait gate.
                    metrics.append(
                        print_exec_debug(
                            "raw-joint-debug",
                            arm,
                            before_pose[arm],
                            recorded_pose,
                            actual_pose[arm],
                            args.debug_block_pos_err_m,
                            args.debug_block_ori_err_rad,
                        )
                    )

            if args.debug and args.debug_block_until_error_small:
                if not wait_until_small_error(
                    client,
                    "raw-joint-debug",
                    arms,
                    metrics,
                    args.debug_block_pos_err_m,
                    args.debug_block_ori_err_rad,
                    args.debug_block_poll_sec,
                ):
                    break
            elif (idx - start) % max(1, args.log_every) == 0 or idx == stop - 1:
                print(f"[raw-joint-replay] step={idx}/{stop - 1}")
    finally:
        client.close()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay R1Lite RAW data directly as joint state/action commands.")
    add_common_args(parser, REPO_ROOT / "data/RAW/test")
    args = parser.parse_args()

    trajectories = load_raw_trajectories(Path(args.input_dir), args.arm, args.control_hz)
    if args.trajectory_index < 0 or args.trajectory_index >= len(trajectories):
        raise IndexError(f"trajectory_index={args.trajectory_index} out of range for {len(trajectories)} trajectories")
    trajectory = trajectories[args.trajectory_index]
    arms = arms_for_replay(args.arm)
    start = int(args.start_exec_step)
    if start < 0 or start >= len(trajectory) - 1:
        raise ValueError(f"start_exec_step={start} out of range for {len(trajectory)} raw frames")
    stop = len(trajectory) - 1 if args.max_steps is None else min(len(trajectory) - 1, start + int(args.max_steps))

    print(
        f"RAW joint replay: trajectory={args.trajectory_index}, frames={len(trajectory)}, "
        f"steps={stop - start}, exec={args.exec_mode}, replay={args.replay_mode}, start={start}"
    )
    if args.exec_mode == "offline":
        rows = _run_offline(trajectory, arms, start, stop, args.replay_mode)
    else:
        rows = _run_online(args, trajectory, arms, start, stop)
    write_rows(rows, args.output_csv, args.output_summary_json)


if __name__ == "__main__":
    main()
