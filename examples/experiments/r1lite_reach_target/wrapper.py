from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class ReachTargetTaskConfig:
    arm: str = "right"
    target_left_pose: tuple = (0.43, -0.20, 0.28, 0.0, 1.0, 0.0, 0.0)
    target_right_pose: tuple = (0.43, 0.20, 0.28, 0.0, 1.0, 0.0, 0.0)
    position_tolerance_m: float = 0.03
    orientation_tolerance_rad: float = 0.35
    success_reward: float = 10.0
    dense_position_weight: float = 1.0
    dense_orientation_weight: float = 0.1
    reward_neg: float = -0.05

    @property
    def target_pose(self):
        if self.arm == "left":
            return np.asarray(self.target_left_pose, dtype=np.float32)
        return np.asarray(self.target_right_pose, dtype=np.float32)


def quat_angle_error_rad(current_xyzw: np.ndarray, target_xyzw: np.ndarray) -> float:
    current = Rotation.from_quat(current_xyzw)
    target = Rotation.from_quat(target_xyzw)
    delta = current.inv() * target
    return float(np.linalg.norm(delta.as_rotvec()))


class ReachTargetRewardWrapper(gym.Wrapper):
    """
    Dense reaching reward with a sparse success bonus.
    Success is defined by both position and orientation thresholds.
    """

    def __init__(self, env, task_config: ReachTargetTaskConfig):
        super().__init__(env)
        self.task_config = task_config
        self.target_pose = task_config.target_pose.copy()

    def _compute_metrics(self, observation):
        tcp_pose = np.asarray(observation["state"]["tcp_pose"], dtype=np.float32)
        pos_error = float(np.linalg.norm(tcp_pose[:3] - self.target_pose[:3]))
        ori_error = quat_angle_error_rad(tcp_pose[3:], self.target_pose[3:])
        return pos_error, ori_error

    def _success(self, pos_error: float, ori_error: float) -> bool:
        return (
            pos_error <= self.task_config.position_tolerance_m
            and ori_error <= self.task_config.orientation_tolerance_rad
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info = dict(info)
        info["target_pose"] = self.target_pose.copy()
        info["task_name"] = "reach_target_pose"
        return obs, info

    def step(self, action):
        observation, _, done, truncated, info = self.env.step(action)
        info = dict(info)

        pos_error, ori_error = self._compute_metrics(observation)
        succeed = self._success(pos_error, ori_error)

        reward = self.task_config.reward_neg
        reward -= self.task_config.dense_position_weight * pos_error
        reward -= self.task_config.dense_orientation_weight * ori_error
        if succeed:
            reward = self.task_config.success_reward

        done = bool(done or truncated or succeed)
        info["succeed"] = succeed
        info["target_pose"] = self.target_pose.copy()
        info["position_error_m"] = pos_error
        info["orientation_error_rad"] = ori_error
        info["reward_components"] = {
            "base": self.task_config.reward_neg,
            "position_penalty": -self.task_config.dense_position_weight * pos_error,
            "orientation_penalty": -self.task_config.dense_orientation_weight * ori_error,
            "success_bonus": self.task_config.success_reward if succeed else 0.0,
        }
        return observation, float(reward), done, truncated, info
