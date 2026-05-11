#!/usr/bin/env python3

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from raw_replay_common import (
    REPO_ROOT,
    add_common_args,
    arms_for_replay,
    current_robot_maps,
    frame_for_arm,
    gripper_value,
    load_raw_trajectories,
    print_eef_action_debug,
    print_exec_debug,
    quat_angle_error_rad,
    rotvec_delta,
    send_pose_targets,
    target_pose_from_eef_delta,
    wait_until_small_error,
    write_rows,
    R1LiteClient,
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
            targets[arm] = np.asarray(nxt["tcp_pose"], dtype=np.float32).copy()
        else:
            delta = np.concatenate(
                [
                    np.asarray(nxt["tcp_pose"][:3], dtype=np.float32) - np.asarray(cur["tcp_pose"][:3], dtype=np.float32),
                    rotvec_delta(np.asarray(cur["tcp_pose"][3:], dtype=np.float32), np.asarray(nxt["tcp_pose"][3:], dtype=np.float32)),
                ],
                axis=0,
            )
            targets[arm] = target_pose_from_eef_delta(live_reference[arm], delta)
    return targets


def _grippers_from_frame(frame: Dict[str, Any], arms: List[str]) -> Dict[str, float]:
    return {arm: gripper_value(frame_for_arm(frame, arm)) for arm in arms}


def _run_offline(trajectory: List[Dict[str, Any]], arms: List[str], start: int, stop: int, replay_mode: str) -> List[Dict[str, Any]]:
    rows = []
    for idx in range(start, stop):
        cur = trajectory[idx]
        nxt = trajectory[idx + 1]
        live_ref = {arm: np.asarray(frame_for_arm(cur, arm)["tcp_pose"], dtype=np.float32) for arm in arms}
        targets = _targets_from_frames(cur, nxt, arms, replay_mode, live_ref)
        for arm in arms:
            recorded = np.asarray(frame_for_arm(nxt, arm)["tcp_pose"], dtype=np.float32)
            target = targets[arm]
            rows.append(
                {
                    "step": idx,
                    "arm": arm,
                    "target_to_recorded_pos_err_m": float(np.linalg.norm(target[:3] - recorded[:3])),
                    "target_to_recorded_ori_err_rad": quat_angle_error_rad(target[3:7], recorded[3:7]),
                }
            )
    return rows


def _run_online(args: argparse.Namespace, trajectory: List[Dict[str, Any]], arms: List[str], start: int, stop: int) -> List[Dict[str, Any]]:
    client = R1LiteClient(args.server_url)
    rows: List[Dict[str, Any]] = []
    try:
        if args.online_start_mode == "move_to_recorded":
            start_targets = {arm: np.asarray(frame_for_arm(trajectory[start], arm)["tcp_pose"], dtype=np.float32) for arm in arms}
            start_grippers = _grippers_from_frame(trajectory[start], arms)
            print(f"[raw-eef-replay] moving to recorded current pose at step={start}")
            send_pose_targets(client, start_targets, start_grippers, args.mode, args.owner, args.preset)
            time.sleep(max(0.0, args.reset_wait_sec))

        for idx in range(start, stop):
            cur = trajectory[idx]
            nxt = trajectory[idx + 1]
            if args.debug and not print_eef_action_debug(idx, cur, nxt, arms):
                break

            before_pose, _, _ = current_robot_maps(client, arms)
            targets = _targets_from_frames(cur, nxt, arms, args.replay_mode, before_pose)
            grippers = _grippers_from_frame(nxt, arms)
            step_start = time.time()
            send_pose_targets(client, targets, grippers, args.mode, args.owner, args.preset)
            time.sleep(max(0.0, (1.0 / max(args.control_hz, 1e-6)) - (time.time() - step_start)))
            actual_pose, _, actual_grippers = current_robot_maps(client, arms)

            metrics = []
            for arm in arms:
                recorded = np.asarray(frame_for_arm(nxt, arm)["tcp_pose"], dtype=np.float32)
                target = targets[arm]
                actual = actual_pose[arm]
                rows.append(
                    {
                        "step": idx,
                        "arm": arm,
                        "replay_mode": args.replay_mode,
                        "actual_to_target_pos_err_m": float(np.linalg.norm(actual[:3] - target[:3])),
                        "actual_to_target_ori_err_rad": quat_angle_error_rad(actual[3:7], target[3:7]),
                        "actual_to_recorded_pos_err_m": float(np.linalg.norm(actual[:3] - recorded[:3])),
                        "actual_to_recorded_ori_err_rad": quat_angle_error_rad(actual[3:7], recorded[3:7]),
                        "actual_gripper": actual_grippers[arm],
                        "recorded_gripper": gripper_value(frame_for_arm(nxt, arm)),
                    }
                )
                if args.debug:
                    metrics.append(
                        print_exec_debug(
                            "raw-eef-debug",
                            arm,
                            before_pose[arm],
                            target,
                            actual,
                            args.debug_block_pos_err_m,
                            args.debug_block_ori_err_rad,
                        )
                    )

            if args.debug and args.debug_block_until_error_small:
                if not wait_until_small_error(
                    client,
                    "raw-eef-debug",
                    arms,
                    metrics,
                    args.debug_block_pos_err_m,
                    args.debug_block_ori_err_rad,
                    args.debug_block_poll_sec,
                ):
                    break
            elif (idx - start) % max(1, args.log_every) == 0 or idx == stop - 1:
                print(f"[raw-eef-replay] step={idx}/{stop - 1}")
    finally:
        client.close()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay R1Lite RAW data directly as EEF state/action commands.")
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
        f"RAW EEF replay: trajectory={args.trajectory_index}, frames={len(trajectory)}, "
        f"steps={stop - start}, exec={args.exec_mode}, replay={args.replay_mode}, start={start}"
    )
    if args.exec_mode == "offline":
        rows = _run_offline(trajectory, arms, start, stop, args.replay_mode)
    else:
        rows = _run_online(args, trajectory, arms, start, stop)
    write_rows(rows, args.output_csv, args.output_summary_json)


if __name__ == "__main__":
    main()
