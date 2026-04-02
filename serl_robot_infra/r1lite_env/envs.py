import copy
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation

from r1lite_env.client import R1LiteClient, decode_image_base64


@dataclass
class R1LiteEnvConfig:
    SERVER_URL: str = "http://127.0.0.1:8001/"
    ACTION_SCALE: np.ndarray = field(default_factory=lambda: np.array([0.03, 0.20, 1.0], dtype=np.float32))
    DISPLAY_IMAGE: bool = False
    MAX_EPISODE_LENGTH: int = 100
    DEFAULT_MODE: str = "ee_pose_servo"
    DEFAULT_PRESET: str = "free_space"
    RESET_LEFT_POSE: list = field(default_factory=lambda: [0.35, -0.25, 0.32, 0.0, 1.0, 0.0, 0.0])
    RESET_RIGHT_POSE: list = field(default_factory=lambda: [0.35, 0.25, 0.32, 0.0, 1.0, 0.0, 0.0])
    RESET_TORSO: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    RANDOM_RESET: bool = False
    RANDOM_XY_RANGE: float = 0.0
    RANDOM_RZ_RANGE: float = 0.0
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
    IMAGE_KEYS: tuple = ("head", "left_wrist", "right_wrist")
    ARM_IMAGE_KEYS: Dict[str, tuple] = field(default_factory=lambda: {"left": ("head", "left_wrist"), "right": ("head", "right_wrist")})


class _BaseR1LiteEnv(gym.Env):
    def __init__(self, config: Optional[R1LiteEnvConfig] = None, hz: int = 10, client: Optional[R1LiteClient] = None):
        self.config = config or R1LiteEnvConfig()
        self.client = client or R1LiteClient(self.config.SERVER_URL)
        self.hz = hz
        self.max_episode_length = self.config.MAX_EPISODE_LENGTH
        self.curr_path_length = 0
        self.last_obs = None
        self.random_reset = bool(self.config.RANDOM_RESET)
        self.random_xy_range = float(self.config.RANDOM_XY_RANGE)
        self.random_rz_range = float(self.config.RANDOM_RZ_RANGE)

    def _step_sleep(self, start_time: float):
        time.sleep(max(0.0, (1.0 / self.hz) - (time.time() - start_time)))

    def _decode_images(self, image_dict: Dict[str, Optional[str]], keys) -> Dict[str, np.ndarray]:
        decoded = {}
        for key in keys:
            image = decode_image_base64(image_dict.get(key))
            if image is None:
                image = np.zeros((256, 256, 3), dtype=np.uint8)
            decoded[key] = image
        return decoded

    def _target_pose_from_action(self, tcp_pose: np.ndarray, action: np.ndarray) -> np.ndarray:
        pose = tcp_pose.copy()
        pose[:3] = pose[:3] + action[:3] * float(self.config.ACTION_SCALE[0])
        pose[3:] = (
            Rotation.from_euler("xyz", action[3:6] * float(self.config.ACTION_SCALE[1]))
            * Rotation.from_quat(tcp_pose[3:])
        ).as_quat()
        return pose

    def _sample_reset_pose(self, arm: str) -> list:
        if arm == "left":
            pose = np.asarray(self.config.RESET_LEFT_POSE, dtype=np.float32).copy()
        else:
            pose = np.asarray(self.config.RESET_RIGHT_POSE, dtype=np.float32).copy()

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
        euler = Rotation.from_quat(clipped[3:]).as_euler("xyz")

        sign = np.sign(euler[0]) if euler[0] != 0 else 1.0
        euler[0] = sign * np.clip(np.abs(euler[0]), lower[3], upper[3])
        euler[1:] = np.clip(euler[1:], lower[4:], upper[4:])
        clipped[3:] = Rotation.from_euler("xyz", euler).as_quat().astype(np.float32)
        return clipped


class R1LiteArmEnv(_BaseR1LiteEnv):
    def __init__(self, arm: str = "left", config: Optional[R1LiteEnvConfig] = None, hz: int = 10, client: Optional[R1LiteClient] = None):
        assert arm in ("left", "right")
        self.arm = arm
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
                        key: gym.spaces.Box(0, 255, shape=(256, 256, 3), dtype=np.uint8)
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
        if self.arm == "left":
            self.client.reset(left_pose=self._sample_reset_pose("left"), torso=self.config.RESET_TORSO)
        else:
            self.client.reset(right_pose=self._sample_reset_pose("right"), torso=self.config.RESET_TORSO)
        raw = self.client.get_state()
        obs = self._extract_obs(raw)
        self.last_obs = obs
        return obs, {"succeed": False}

    def step(self, action: np.ndarray):
        start_time = time.time()
        action = np.clip(np.asarray(action, dtype=np.float32), self.action_space.low, self.action_space.high)
        raw = self.client.get_state()
        tcp_pose = np.asarray(raw["state"][self.arm]["tcp_pose"], dtype=np.float32)
        pose_target = self._clip_pose_to_safety_box(self.arm, self._target_pose_from_action(tcp_pose, action[:6]))
        payload = {
            "mode": self.config.DEFAULT_MODE,
            "owner": "policy",
            "left" if self.arm == "left" else "right": {
                "pose_target": pose_target.tolist(),
                "gripper": float(np.clip((action[6] + 1.0) * 50.0, 0.0, 100.0)),
                "preset": self.config.DEFAULT_PRESET,
            },
        }
        self.client.post_action(payload)
        self._step_sleep(start_time)
        next_raw = self.client.get_state()
        obs = self._extract_obs(next_raw)
        self.curr_path_length += 1
        done = self.curr_path_length >= self.max_episode_length
        self.last_obs = obs
        return obs, 0.0, done, False, {"succeed": False}


class DualR1LiteEnv(_BaseR1LiteEnv):
    def __init__(self, config: Optional[R1LiteEnvConfig] = None, hz: int = 10, client: Optional[R1LiteClient] = None):
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
                        key: gym.spaces.Box(0, 255, shape=(256, 256, 3), dtype=np.uint8)
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
        self.client.reset(
            left_pose=self._sample_reset_pose("left"),
            right_pose=self._sample_reset_pose("right"),
            torso=self.config.RESET_TORSO,
        )
        raw = self.client.get_state()
        obs = self._extract_obs(raw)
        self.last_obs = obs
        return obs, {"succeed": False}

    def step(self, action: np.ndarray):
        start_time = time.time()
        action = np.clip(np.asarray(action, dtype=np.float32), self.action_space.low, self.action_space.high)
        raw = self.client.get_state()
        left_action = action[:7]
        right_action = action[7:]
        left_pose = self._clip_pose_to_safety_box(
            "left",
            self._target_pose_from_action(np.asarray(raw["state"]["left"]["tcp_pose"], dtype=np.float32), left_action[:6]),
        )
        right_pose = self._clip_pose_to_safety_box(
            "right",
            self._target_pose_from_action(np.asarray(raw["state"]["right"]["tcp_pose"], dtype=np.float32), right_action[:6]),
        )
        payload = {
            "mode": self.config.DEFAULT_MODE,
            "owner": "policy",
            "left": {
                "pose_target": left_pose.tolist(),
                "gripper": float(np.clip((left_action[6] + 1.0) * 50.0, 0.0, 100.0)),
                "preset": self.config.DEFAULT_PRESET,
            },
            "right": {
                "pose_target": right_pose.tolist(),
                "gripper": float(np.clip((right_action[6] + 1.0) * 50.0, 0.0, 100.0)),
                "preset": self.config.DEFAULT_PRESET,
            },
        }
        self.client.post_action(payload)
        self._step_sleep(start_time)
        next_raw = self.client.get_state()
        obs = self._extract_obs(next_raw)
        self.curr_path_length += 1
        done = self.curr_path_length >= self.max_episode_length
        self.last_obs = obs
        return obs, 0.0, done, False, {"succeed": False}
