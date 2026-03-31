import argparse
import time
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from franka_env.spacemouse.spacemouse_expert import SpaceMouseExpert
from r1lite_env.client import R1LiteClient


def _pose_target_from_action(tcp_pose: np.ndarray, action: np.ndarray, xyz_scale: float, rot_scale: float) -> np.ndarray:
    pose = np.asarray(tcp_pose, dtype=np.float64).copy()
    delta = np.asarray(action[:6], dtype=np.float64)
    pose[:3] = pose[:3] + delta[:3] * xyz_scale
    pose[3:] = (Rotation.from_euler("xyz", delta[3:] * rot_scale) * Rotation.from_quat(pose[3:])).as_quat()
    return pose


def _single_arm_gripper(buttons) -> Optional[float]:
    if len(buttons) >= 1 and buttons[0]:
        return 0.0
    if len(buttons) >= 2 and buttons[-1]:
        return 100.0
    return None


def _dual_arm_grippers(buttons) -> Tuple[Optional[float], Optional[float]]:
    if len(buttons) != 4:
        return None, None
    left = 0.0 if buttons[0] else 100.0 if buttons[1] else None
    right = 0.0 if buttons[2] else 100.0 if buttons[3] else None
    return left, right


def _build_single_arm_payload(
    state: Dict,
    arm: str,
    action: np.ndarray,
    buttons,
    seq: int,
    mode: str,
    preset: str,
    xyz_scale: float,
    rot_scale: float,
) -> Dict:
    tcp_pose = np.asarray(state["state"][arm]["tcp_pose"], dtype=np.float64)
    arm_payload = {
        "pose_target": _pose_target_from_action(tcp_pose, action[:6], xyz_scale, rot_scale).tolist(),
        "preset": preset,
    }
    gripper = _single_arm_gripper(buttons)
    if gripper is not None:
        arm_payload["gripper"] = gripper
    return {
        "owner": "teleop",
        "teleop_source": "spacemouse",
        "mode": mode,
        "seq": seq,
        arm: arm_payload,
    }


def _build_dual_arm_payload(
    state: Dict,
    action: np.ndarray,
    buttons,
    seq: int,
    mode: str,
    preset: str,
    xyz_scale: float,
    rot_scale: float,
) -> Dict:
    left_pose = np.asarray(state["state"]["left"]["tcp_pose"], dtype=np.float64)
    right_pose = np.asarray(state["state"]["right"]["tcp_pose"], dtype=np.float64)
    left_gripper, right_gripper = _dual_arm_grippers(buttons)

    left_payload = {
        "pose_target": _pose_target_from_action(left_pose, action[:6], xyz_scale, rot_scale).tolist(),
        "preset": preset,
    }
    right_payload = {
        "pose_target": _pose_target_from_action(right_pose, action[6:12], xyz_scale, rot_scale).tolist(),
        "preset": preset,
    }
    if left_gripper is not None:
        left_payload["gripper"] = left_gripper
    if right_gripper is not None:
        right_payload["gripper"] = right_gripper

    return {
        "owner": "teleop",
        "teleop_source": "spacemouse",
        "mode": mode,
        "seq": seq,
        "left": left_payload,
        "right": right_payload,
    }


def run() -> None:
    parser = argparse.ArgumentParser(description="Control R1Lite via SpaceMouse teleop owner")
    parser.add_argument("--server-url", default="http://127.0.0.1:8001/")
    parser.add_argument("--arm", default="right", choices=["left", "right", "dual"])
    parser.add_argument("--hz", type=float, default=10.0)
    parser.add_argument("--xyz-scale", type=float, default=0.03)
    parser.add_argument("--rot-scale", type=float, default=0.20)
    parser.add_argument("--preset", default="free_space")
    parser.add_argument("--mode", default=None, help="Defaults to the current service active_mode")
    args = parser.parse_args()

    client = R1LiteClient(args.server_url)
    expert = SpaceMouseExpert()
    seq = 0
    last_log_time = 0.0
    step_dt = 1.0 / max(args.hz, 1e-6)

    try:
        while True:
            start_time = time.time()
            state = client.get_state()
            mode = args.mode or state["meta"]["mode"]
            action, buttons = expert.get_action()

            if args.arm == "dual":
                if len(action) < 12:
                    raise RuntimeError("dual arm teleop requires two SpaceMouse devices")
                payload = _build_dual_arm_payload(
                    state=state,
                    action=action,
                    buttons=buttons,
                    seq=seq,
                    mode=mode,
                    preset=args.preset,
                    xyz_scale=args.xyz_scale,
                    rot_scale=args.rot_scale,
                )
            else:
                if len(action) < 6:
                    raise RuntimeError("failed to read a valid SpaceMouse 6DoF action")
                payload = _build_single_arm_payload(
                    state=state,
                    arm=args.arm,
                    action=action,
                    buttons=buttons,
                    seq=seq,
                    mode=mode,
                    preset=args.preset,
                    xyz_scale=args.xyz_scale,
                    rot_scale=args.rot_scale,
                )

            response = client.post_action(payload)
            if time.time() - last_log_time > 1.0:
                print(
                    f"seq={seq} owner={response.get('owner')} teleop_source={response.get('teleop_source')} "
                    f"published={response.get('published')}"
                )
                last_log_time = time.time()
            seq += 1
            time.sleep(max(0.0, step_dt - (time.time() - start_time)))
    finally:
        expert.close()


if __name__ == "__main__":
    run()
