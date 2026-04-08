import copy
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import cv2
import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation

from r1lite_env.client import R1LiteClient, decode_image_base64


@dataclass
class R1LiteEnvConfig:
    SERVER_URL: str = "http://127.0.0.1:8001/"
    ACTION_SCALE: np.ndarray = field(default_factory=lambda: np.array([0.03, 0.20, 1.0], dtype=np.float32))
    CONTROL_HZ: float = 10.0
    LOG_EFFECTIVE_HZ: bool = False
    DISPLAY_IMAGE: bool = False
    MAX_EPISODE_LENGTH: int = 100
    DEFAULT_MODE: str = "ee_pose_servo"
    DEFAULT_PRESET: str = "free_space"
    RESET_LEFT_POSE: list = field(default_factory=lambda: [0.35, -0.25, 0.32, 0.0, 1.0, 0.0, 0.0])
    RESET_RIGHT_POSE: list = field(default_factory=lambda: [0.35, 0.25, 0.32, 0.0, 1.0, 0.0, 0.0])
    RESET_LEFT_JOINT: list = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    RESET_RIGHT_JOINT: list = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    RESET_TORSO: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    RANDOM_RESET: bool = False
    RANDOM_XY_RANGE: float = 0.0
    RANDOM_RZ_RANGE: float = 0.0
    # reset 不是瞬时完成的，给机器人留一段收敛时间后再采当前位姿，
    # 避免下一步继续沿用 reset 前的目标点。
    RESET_SETTLE_SEC: float = 1.5
    RESET_WAIT_TIMEOUT_SEC: float = 6.0
    # reset 超时后默认直接报错停住，避免未完成 reset 就进入下一条 episode。
    RESET_FAIL_ON_TIMEOUT: bool = True
    RESET_POLL_SEC: float = 0.1
    RESET_STABLE_COUNT: int = 4
    RESET_DEBUG: bool = False
    RESET_POSITION_TOLERANCE_M: float = 0.03
    RESET_ORIENTATION_TOLERANCE_RAD: float = 0.35
    RESET_JOINT_TOLERANCE_RAD: float = 0.08
    RESET_STABLE_POS_EPS_M: float = 0.005
    RESET_STABLE_ORI_EPS_RAD: float = 0.08
    RESET_STABLE_JOINT_EPS_RAD: float = 0.03
    ABS_POSE_LIMIT_LOW: Dict[str, list] = field(
        default_factory=lambda: {
            "left": [-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf],
            "right": [-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf],
        }
    )
    ABS_POSE_LIMIT_HIGH: Dict[str, list] = field(
        default_factory=lambda: {
            "left": [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
            "right": [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
        }
    )
    IMAGE_RESOLUTION: Dict[str, tuple] = field(
        default_factory=lambda: {
            "head": (256, 256),
            "left_wrist": (128, 128),
            "right_wrist": (128, 128),
        }
    )
    IMAGE_KEYS: tuple = ("head", "left_wrist", "right_wrist")
    ARM_IMAGE_KEYS: Dict[str, tuple] = field(default_factory=lambda: {"left": ("head", "left_wrist"), "right": ("head", "right_wrist")})
    # 参考 Franka 的 fixed-gripper 语义：任务不需要夹爪时，reset 后固定打开，
    # 后续 step 不再发送 gripper 指令，避免零动作把夹爪拉到中间值。
    FIX_GRIPPER_OPEN: bool = False
    FIXED_GRIPPER_VALUE: float = 100.0


class _BaseR1LiteEnv(gym.Env):
    def __init__(self, config: Optional[R1LiteEnvConfig] = None, hz: int = 10, client: Optional[R1LiteClient] = None):
        self.config = config or R1LiteEnvConfig()
        self.client = client or R1LiteClient(self.config.SERVER_URL)
        self.hz = float(getattr(self.config, "CONTROL_HZ", hz))
        self.max_episode_length = self.config.MAX_EPISODE_LENGTH
        self.curr_path_length = 0
        self.last_obs = None
        self.random_reset = bool(self.config.RANDOM_RESET)
        self.random_xy_range = float(self.config.RANDOM_XY_RANGE)
        self.random_rz_range = float(self.config.RANDOM_RZ_RANGE)
        self._hz_counter = 0
        self._hz_log_start_time = time.time()

    def _step_sleep(self, start_time: float):
        time.sleep(max(0.0, (1.0 / self.hz) - (time.time() - start_time)))

    def _maybe_log_effective_hz(self, tag: str):
        if not bool(getattr(self.config, "LOG_EFFECTIVE_HZ", False)):
            return
        self._hz_counter += 1
        now = time.time()
        dt = now - self._hz_log_start_time
        if dt >= 1.0:
            hz = self._hz_counter / max(dt, 1e-6)
            print(f"[{tag}] effective_control_hz={hz:.2f}")
            self._hz_counter = 0
            self._hz_log_start_time = now

    def _decode_images(self, image_dict: Dict[str, Optional[str]], keys) -> Dict[str, np.ndarray]:
        decoded = {}
        for key in keys:
            image = decode_image_base64(image_dict.get(key))
            target_hw = self.config.IMAGE_RESOLUTION.get(key, (256, 256))
            if image is None:
                image = np.zeros((target_hw[0], target_hw[1], 3), dtype=np.uint8)
            elif image.shape[:2] != tuple(target_hw):
                image = cv2.resize(image, (int(target_hw[1]), int(target_hw[0])), interpolation=cv2.INTER_AREA)
            decoded[key] = image
        return decoded

    def _maybe_set_fixed_gripper_open(self, arm: str):
        if not bool(self.config.FIX_GRIPPER_OPEN):
            return
        self.client.post_action(
            {
                "mode": self.config.DEFAULT_MODE,
                "owner": "policy",
                arm: {
                    "gripper": float(self.config.FIXED_GRIPPER_VALUE),
                    "preset": self.config.DEFAULT_PRESET,
                },
            }
        )

    def _target_pose_from_action(self, tcp_pose: np.ndarray, action: np.ndarray) -> np.ndarray:
        pose = tcp_pose.copy()
        pose[:3] = pose[:3] + action[:3] * float(self.config.ACTION_SCALE[0])
        pose[3:] = (
            Rotation.from_euler("xyz", action[3:6] * float(self.config.ACTION_SCALE[1]))
            * Rotation.from_quat(tcp_pose[3:])
        ).as_quat()
        return pose

    def _quat_angle_error_rad(self, current_xyzw: np.ndarray, target_xyzw: np.ndarray) -> float:
        current = Rotation.from_quat(current_xyzw)
        target = Rotation.from_quat(target_xyzw)
        delta = current.inv() * target
        return float(np.linalg.norm(delta.as_rotvec()))

    def _normalize_quat_or_identity(self, quat: np.ndarray) -> np.ndarray:
        quat = np.asarray(quat, dtype=np.float32).copy()
        norm = float(np.linalg.norm(quat))
        # 配置里如果误填了零四元数，兜底成单位四元数，避免 reset 直接崩掉。
        if norm < 1e-8:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        return quat / norm

    def _sample_reset_pose(self, arm: str) -> Optional[list]:
        if arm == "left":
            pose = np.asarray(self.config.RESET_LEFT_POSE, dtype=np.float32).copy()
        else:
            pose = np.asarray(self.config.RESET_RIGHT_POSE, dtype=np.float32).copy()

        # 兼容 R1Lite 当前配置习惯：全 0 代表“不要走末端 pose reset，而是回落到服务端默认关节 reset”。
        if pose.shape[0] >= 7 and np.allclose(pose, 0.0):
            return None
        pose[3:] = self._normalize_quat_or_identity(pose[3:])

        if self.random_reset:
            pose[:2] += np.random.uniform(-self.random_xy_range, self.random_xy_range, size=2).astype(np.float32)
            euler = Rotation.from_quat(pose[3:]).as_euler("xyz")
            euler[-1] += float(np.random.uniform(-self.random_rz_range, self.random_rz_range))
            pose[3:] = Rotation.from_euler("xyz", euler).as_quat().astype(np.float32)

        return self._clip_pose_to_safety_box(arm, pose).tolist()

    def _clip_pose_to_safety_box(self, arm: str, pose: np.ndarray) -> np.ndarray:
        clipped = np.asarray(pose, dtype=np.float32).copy()
        lower = np.asarray(self.config.ABS_POSE_LIMIT_LOW[arm], dtype=np.float32)
        upper = np.asarray(self.config.ABS_POSE_LIMIT_HIGH[arm], dtype=np.float32)

        clipped[:3] = np.clip(clipped[:3], lower[:3], upper[:3])
        clipped[3:] = self._normalize_quat_or_identity(clipped[3:])
        euler = Rotation.from_quat(clipped[3:]).as_euler("xyz")

        sign = np.sign(euler[0]) if euler[0] != 0 else 1.0
        euler[0] = sign * np.clip(np.abs(euler[0]), lower[3], upper[3])
        euler[1:] = np.clip(euler[1:], lower[4:], upper[4:])
        clipped[3:] = Rotation.from_euler("xyz", euler).as_quat().astype(np.float32)
        return clipped

    def _wait_until_reset_ready(self, arm: str, target_pose: Optional[np.ndarray], target_joint: Optional[np.ndarray]) -> Dict:
        deadline = time.time() + max(0.0, float(self.config.RESET_WAIT_TIMEOUT_SEC))
        prev_pose = None
        prev_joint = None
        stable_count = 0
        latest_raw = None
        poll_index = 0

        if bool(self.config.RESET_DEBUG):
            target_kind = "pose" if target_pose is not None else "joint"
            print(
                f"[reset-debug:{arm}] waiting for reset completion "
                f"(target={target_kind}, settle={self.config.RESET_SETTLE_SEC:.2f}s, "
                f"timeout={self.config.RESET_WAIT_TIMEOUT_SEC:.2f}s, "
                f"stable_count_required={self.config.RESET_STABLE_COUNT})"
            )

        while time.time() < deadline:
            poll_index += 1
            latest_raw = self.client.get_state()
            arm_state = latest_raw["state"][arm]
            current_pose = np.asarray(arm_state["tcp_pose"], dtype=np.float32)
            current_joint = np.asarray(arm_state["joint_pos"], dtype=np.float32)

            reached = True
            pos_error = None
            ori_error = None
            joint_error = None
            if target_pose is not None:
                pos_error = float(np.linalg.norm(current_pose[:3] - target_pose[:3]))
                ori_error = self._quat_angle_error_rad(current_pose[3:], target_pose[3:])
                reached = (
                    pos_error <= float(self.config.RESET_POSITION_TOLERANCE_M)
                    and ori_error <= float(self.config.RESET_ORIENTATION_TOLERANCE_RAD)
                )
            elif target_joint is not None:
                joint_error = float(np.max(np.abs(current_joint[:6] - target_joint[:6])))
                reached = joint_error <= float(self.config.RESET_JOINT_TOLERANCE_RAD)

            stable = False
            pos_delta = None
            ori_delta = None
            joint_delta = None
            if prev_pose is not None and prev_joint is not None:
                pos_delta = float(np.linalg.norm(current_pose[:3] - prev_pose[:3]))
                ori_delta = self._quat_angle_error_rad(current_pose[3:], prev_pose[3:])
                joint_delta = float(np.max(np.abs(current_joint[:6] - prev_joint[:6])))
                stable = (
                    pos_delta <= float(self.config.RESET_STABLE_POS_EPS_M)
                    and ori_delta <= float(self.config.RESET_STABLE_ORI_EPS_RAD)
                    and joint_delta <= float(self.config.RESET_STABLE_JOINT_EPS_RAD)
                )

            stable_count = stable_count + 1 if (reached and stable) else 0

            if bool(self.config.RESET_DEBUG):
                if target_pose is not None:
                    print(
                        f"[reset-debug:{arm}] poll={poll_index} "
                        f"reached={reached} stable={stable} stable_count={stable_count}/{self.config.RESET_STABLE_COUNT} "
                        f"pos_error={pos_error:.4f} ori_error={ori_error:.4f} "
                        f"pos_delta={0.0 if pos_delta is None else pos_delta:.4f} "
                        f"ori_delta={0.0 if ori_delta is None else ori_delta:.4f}"
                    )
                else:
                    print(
                        f"[reset-debug:{arm}] poll={poll_index} "
                        f"reached={reached} stable={stable} stable_count={stable_count}/{self.config.RESET_STABLE_COUNT} "
                        f"joint_error={joint_error:.4f} "
                        f"joint_delta={0.0 if joint_delta is None else joint_delta:.4f}"
                    )
            prev_pose = current_pose.copy()
            prev_joint = current_joint.copy()

            if stable_count >= int(self.config.RESET_STABLE_COUNT):
                if bool(self.config.RESET_DEBUG):
                    print(f"[reset-debug:{arm}] reset accepted after {poll_index} polls")
                return latest_raw

            time.sleep(max(0.0, float(self.config.RESET_POLL_SEC)))

        if bool(self.config.RESET_DEBUG):
            print(f"[reset-debug:{arm}] reset wait timed out after {poll_index} polls")
        if bool(self.config.RESET_FAIL_ON_TIMEOUT):
            if target_pose is not None:
                raise TimeoutError(
                    f"{arm} reset timed out: pose reset did not become reached+stable within "
                    f"{self.config.RESET_WAIT_TIMEOUT_SEC:.2f}s"
                )
            raise TimeoutError(
                f"{arm} reset timed out: joint reset did not become reached+stable within "
                f"{self.config.RESET_WAIT_TIMEOUT_SEC:.2f}s"
            )
        # 仅在显式关闭 fail-on-timeout 时才回退到旧行为。
        return latest_raw if latest_raw is not None else self.client.get_state()


class R1LiteArmEnv(_BaseR1LiteEnv):
    def __init__(self, arm: str = "left", config: Optional[R1LiteEnvConfig] = None, hz: int = 10, client: Optional[R1LiteClient] = None):
        assert arm in ("left", "right")
        self.arm = arm
        self.commanded_pose: Optional[np.ndarray] = None
        super().__init__(config=config, hz=hz, client=client)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "tcp_pose": gym.spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32),
                        "tcp_vel": gym.spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
                        "tcp_force": gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                        "tcp_torque": gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                        "gripper_pose": gym.spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
                        "joint_pos": gym.spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
                        "joint_vel": gym.spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
                        "joint_effort": gym.spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
                    }
                ),
                "images": gym.spaces.Dict(
                    {
                        key: gym.spaces.Box(
                            0,
                            255,
                            shape=(
                                int(self.config.IMAGE_RESOLUTION[key][0]),
                                int(self.config.IMAGE_RESOLUTION[key][1]),
                                3,
                            ),
                            dtype=np.uint8,
                        )
                        for key in self.config.ARM_IMAGE_KEYS[self.arm]
                    }
                ),
            }
        )

    def _extract_obs(self, raw: Dict) -> Dict:
        arm_state = raw["state"][self.arm]
        return {
            "state": {
                "tcp_pose": np.asarray(arm_state["tcp_pose"], dtype=np.float32),
                "tcp_vel": np.asarray(arm_state["tcp_vel"], dtype=np.float32),
                "tcp_force": np.asarray(arm_state["tcp_force"], dtype=np.float32),
                "tcp_torque": np.asarray(arm_state["tcp_torque"], dtype=np.float32),
                "gripper_pose": np.asarray(arm_state["gripper_pose"], dtype=np.float32),
                "joint_pos": np.asarray(arm_state["joint_pos"], dtype=np.float32),
                "joint_vel": np.asarray(arm_state["joint_vel"], dtype=np.float32),
                "joint_effort": np.asarray(arm_state["joint_effort"], dtype=np.float32),
            },
            "images": self._decode_images(raw["images"], self.config.ARM_IMAGE_KEYS[self.arm]),
        }

    def reset(self, **kwargs):
        self.curr_path_length = 0
        self.commanded_pose = None
        reset_pose = self._sample_reset_pose(self.arm)
        reset_joint = None
        if self.arm == "left":
            if reset_pose is None:
                reset_joint = np.asarray(self.config.RESET_LEFT_JOINT[:6], dtype=np.float32)
            self.client.reset(left_pose=reset_pose, torso=self.config.RESET_TORSO)
        else:
            if reset_pose is None:
                reset_joint = np.asarray(self.config.RESET_RIGHT_JOINT[:6], dtype=np.float32)
            self.client.reset(right_pose=reset_pose, torso=self.config.RESET_TORSO)
        # fixed-gripper 任务在 reset 后显式张开一次夹爪，后续 step 将不再驱动它。
        self._maybe_set_fixed_gripper_open(self.arm)
        time.sleep(max(0.0, float(self.config.RESET_SETTLE_SEC)))
        raw = self._wait_until_reset_ready(
            self.arm,
            None if reset_pose is None else np.asarray(reset_pose, dtype=np.float32),
            reset_joint,
        )
        obs = self._extract_obs(raw)
        # reset 后把“最后一次已发送目标”同步到当前末端位姿。
        # 之后零动作会保持这个目标，而不是每步追着 noisy tcp_pose 跑。
        self.commanded_pose = np.asarray(raw["state"][self.arm]["tcp_pose"], dtype=np.float32).copy()
        self.last_obs = obs
        return obs, {"succeed": False}

    def step(self, action: np.ndarray):
        start_time = time.time()
        action = np.clip(np.asarray(action, dtype=np.float32), self.action_space.low, self.action_space.high)
        raw = self.client.get_state()
        tcp_pose = np.asarray(raw["state"][self.arm]["tcp_pose"], dtype=np.float32)
        # 对齐独立 teleop 的手感：有明确人工输入时，用当前观测 tcp_pose 做参考；
        # 只有零输入保持时，才继续沿用 commanded_pose 来 hold 住目标。
        if float(np.linalg.norm(action[:6])) > 1e-6:
            reference_pose = tcp_pose.copy()
        else:
            reference_pose = self.commanded_pose.copy() if self.commanded_pose is not None else tcp_pose.copy()
        pose_target = self._clip_pose_to_safety_box(self.arm, self._target_pose_from_action(reference_pose, action[:6]))
        side_payload = {
            "pose_target": pose_target.tolist(),
            "preset": self.config.DEFAULT_PRESET,
        }
        if not bool(self.config.FIX_GRIPPER_OPEN):
            side_payload["gripper"] = float(np.clip((action[6] + 1.0) * 50.0, 0.0, 100.0))
        payload = {
            "mode": self.config.DEFAULT_MODE,
            "owner": "policy",
            "left" if self.arm == "left" else "right": side_payload,
        }
        self.client.post_action(payload)
        self.commanded_pose = pose_target.copy()
        self._step_sleep(start_time)
        self._maybe_log_effective_hz(f"env-step:{self.arm}")
        next_raw = self.client.get_state()
        obs = self._extract_obs(next_raw)
        self.curr_path_length += 1
        done = self.curr_path_length >= self.max_episode_length
        self.last_obs = obs
        return obs, 0.0, done, False, {"succeed": False}


class DualR1LiteEnv(_BaseR1LiteEnv):
    def __init__(self, config: Optional[R1LiteEnvConfig] = None, hz: int = 10, client: Optional[R1LiteClient] = None):
        self.commanded_pose = {"left": None, "right": None}
        super().__init__(config=config, hz=hz, client=client)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(14,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "left": gym.spaces.Dict(
                            {
                                "tcp_pose": gym.spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32),
                                "tcp_vel": gym.spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
                                "tcp_force": gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                                "tcp_torque": gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                                "gripper_pose": gym.spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
                                "joint_pos": gym.spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
                                "joint_vel": gym.spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
                                "joint_effort": gym.spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
                            }
                        ),
                        "right": gym.spaces.Dict(
                            {
                                "tcp_pose": gym.spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32),
                                "tcp_vel": gym.spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
                                "tcp_force": gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                                "tcp_torque": gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                                "gripper_pose": gym.spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
                                "joint_pos": gym.spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
                                "joint_vel": gym.spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
                                "joint_effort": gym.spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
                            }
                        ),
                        "torso": gym.spaces.Box(-np.inf, np.inf, shape=(9,), dtype=np.float32),
                    }
                ),
                "images": gym.spaces.Dict(
                    {
                        key: gym.spaces.Box(
                            0,
                            255,
                            shape=(
                                int(self.config.IMAGE_RESOLUTION[key][0]),
                                int(self.config.IMAGE_RESOLUTION[key][1]),
                                3,
                            ),
                            dtype=np.uint8,
                        )
                        for key in self.config.IMAGE_KEYS
                    }
                ),
                "meta": gym.spaces.Dict(
                    {
                        "mode": gym.spaces.Text(max_length=64),
                        "health": gym.spaces.Dict({}),
                        "validity": gym.spaces.Dict({}),
                    }
                ),
            }
        )

    def _extract_obs(self, raw: Dict) -> Dict:
        return {
            "state": {
                "left": {key: np.asarray(value, dtype=np.float32) for key, value in raw["state"]["left"].items()},
                "right": {key: np.asarray(value, dtype=np.float32) for key, value in raw["state"]["right"].items()},
                "torso": np.concatenate(
                    [
                        np.asarray(raw["state"]["torso"]["joint_pos"], dtype=np.float32),
                        np.asarray(raw["state"]["torso"]["joint_vel"], dtype=np.float32),
                        np.asarray(raw["state"]["torso"]["joint_effort"], dtype=np.float32),
                    ],
                    axis=0,
                ),
            },
            "images": self._decode_images(raw["images"], self.config.IMAGE_KEYS),
            "meta": copy.deepcopy(raw["meta"]),
        }

    def reset(self, **kwargs):
        self.curr_path_length = 0
        self.commanded_pose = {"left": None, "right": None}
        self.client.reset(
            left_pose=self._sample_reset_pose("left"),
            right_pose=self._sample_reset_pose("right"),
            torso=self.config.RESET_TORSO,
        )
        if bool(self.config.FIX_GRIPPER_OPEN):
            self._maybe_set_fixed_gripper_open("left")
            self._maybe_set_fixed_gripper_open("right")
        time.sleep(max(0.0, float(self.config.RESET_SETTLE_SEC)))
        raw = self.client.get_state()
        obs = self._extract_obs(raw)
        self.commanded_pose["left"] = np.asarray(raw["state"]["left"]["tcp_pose"], dtype=np.float32).copy()
        self.commanded_pose["right"] = np.asarray(raw["state"]["right"]["tcp_pose"], dtype=np.float32).copy()
        self.last_obs = obs
        return obs, {"succeed": False}

    def step(self, action: np.ndarray):
        start_time = time.time()
        action = np.clip(np.asarray(action, dtype=np.float32), self.action_space.low, self.action_space.high)
        raw = self.client.get_state()
        left_action = action[:7]
        right_action = action[7:]
        left_reference_pose = (
            self.commanded_pose["left"].copy()
            if self.commanded_pose["left"] is not None
            else np.asarray(raw["state"]["left"]["tcp_pose"], dtype=np.float32).copy()
        )
        right_reference_pose = (
            self.commanded_pose["right"].copy()
            if self.commanded_pose["right"] is not None
            else np.asarray(raw["state"]["right"]["tcp_pose"], dtype=np.float32).copy()
        )
        left_pose = self._clip_pose_to_safety_box(
            "left",
            self._target_pose_from_action(left_reference_pose, left_action[:6]),
        )
        right_pose = self._clip_pose_to_safety_box(
            "right",
            self._target_pose_from_action(right_reference_pose, right_action[:6]),
        )
        left_payload = {
            "pose_target": left_pose.tolist(),
            "preset": self.config.DEFAULT_PRESET,
        }
        right_payload = {
            "pose_target": right_pose.tolist(),
            "preset": self.config.DEFAULT_PRESET,
        }
        if not bool(self.config.FIX_GRIPPER_OPEN):
            left_payload["gripper"] = float(np.clip((left_action[6] + 1.0) * 50.0, 0.0, 100.0))
            right_payload["gripper"] = float(np.clip((right_action[6] + 1.0) * 50.0, 0.0, 100.0))
        payload = {
            "mode": self.config.DEFAULT_MODE,
            "owner": "policy",
            "left": left_payload,
            "right": right_payload,
        }
        self.client.post_action(payload)
        self.commanded_pose["left"] = left_pose.copy()
        self.commanded_pose["right"] = right_pose.copy()
        self._step_sleep(start_time)
        next_raw = self.client.get_state()
        obs = self._extract_obs(next_raw)
        self.curr_path_length += 1
        done = self.curr_path_length >= self.max_episode_length
        self.last_obs = obs
        return obs, 0.0, done, False, {"succeed": False}
