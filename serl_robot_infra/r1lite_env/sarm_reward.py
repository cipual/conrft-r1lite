import base64
from dataclasses import dataclass
from typing import Any, Dict, Optional

import cv2
import gymnasium as gym
import numpy as np
import requests


@dataclass
class SARMRewardConfig:
    enabled: bool = False
    log_only: bool = True
    endpoint_url: Optional[str] = None
    checkpoint_path: Optional[str] = None
    head_mode: str = "sparse"
    image_key: str = "head"
    success_threshold: float = 0.95
    success_reward: float = 10.0
    reward_scale: float = 1.0
    reward_bias: float = 0.0
    reward_clip_low: float = -1.0
    reward_clip_high: float = 1.0
    timeout: float = 2.0
    task_desc: str = ""


def _encode_image_rgb(image: np.ndarray) -> str:
    image = np.asarray(image)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    bgr = image[..., ::-1]
    ok, buf = cv2.imencode(".jpg", bgr)
    if not ok:
        raise ValueError("Failed to JPEG-encode image for SARM progress request")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _state_vector_from_nested_obs(obs: Dict[str, Any]) -> np.ndarray:
    state = obs["state"]
    left = state["left"]
    right = state["right"]
    torso = np.asarray(state.get("torso", [0.0]), dtype=np.float32).reshape(-1)
    torso_value = torso[:1] if torso.size else np.zeros((1,), dtype=np.float32)
    return np.concatenate(
        [
            np.asarray(left["tcp_pose"], dtype=np.float32).reshape(-1),
            np.asarray(left["tcp_vel"], dtype=np.float32).reshape(-1),
            np.asarray(left["joint_pos"], dtype=np.float32).reshape(-1),
            np.asarray(left["joint_vel"], dtype=np.float32).reshape(-1),
            np.asarray(left["gripper_pose"], dtype=np.float32).reshape(-1),
            np.asarray(right["tcp_pose"], dtype=np.float32).reshape(-1),
            np.asarray(right["tcp_vel"], dtype=np.float32).reshape(-1),
            np.asarray(right["joint_pos"], dtype=np.float32).reshape(-1),
            np.asarray(right["joint_vel"], dtype=np.float32).reshape(-1),
            np.asarray(right["gripper_pose"], dtype=np.float32).reshape(-1),
            torso_value,
        ],
        axis=0,
    ).astype(np.float32)


class SARMProgressClient:
    def __init__(self, config: SARMRewardConfig):
        self.config = config
        if not config.endpoint_url:
            raise ValueError(
                "SARM online reward requires reward_model.endpoint_url. "
                "Run a LeRobot SARM inference sidecar and expose a /predict_progress endpoint."
            )
        self.endpoint_url = config.endpoint_url.rstrip("/") + "/predict_progress"
        self.session = requests.Session()
        self.session.trust_env = False

    def predict(self, obs: Dict[str, Any]) -> float:
        image = obs["images"][self.config.image_key]
        payload = {
            "image_key": self.config.image_key,
            "head_mode": self.config.head_mode,
            "task": self.config.task_desc,
            "image_jpeg_base64": _encode_image_rgb(image),
            "state": _state_vector_from_nested_obs(obs).tolist(),
        }
        response = self.session.post(self.endpoint_url, json=payload, timeout=self.config.timeout)
        response.raise_for_status()
        data = response.json()
        if "progress" not in data:
            raise ValueError(f"SARM progress endpoint did not return `progress`: {data}")
        return float(data["progress"])

    def close(self):
        self.session.close()


class SARMProgressRewardWrapper(gym.Wrapper):
    def __init__(self, env, config: SARMRewardConfig):
        super().__init__(env)
        self.config = config
        self.client = SARMProgressClient(config)
        self.prev_progress = 0.0

    def _progress(self, obs):
        return float(np.clip(self.client.predict(obs), 0.0, 1.0))

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_progress = self._progress(obs)
        info = dict(info)
        info["sarm_progress"] = self.prev_progress
        info["sarm_reward_delta"] = 0.0
        info["succeed"] = False
        return obs, info

    def step(self, action):
        obs, env_reward, done, truncated, info = self.env.step(action)
        progress = self._progress(obs)
        delta = float(np.clip(progress - self.prev_progress, self.config.reward_clip_low, self.config.reward_clip_high))
        sarm_reward = self.config.reward_scale * delta + self.config.reward_bias
        succeed = bool(progress >= self.config.success_threshold)
        if succeed:
            sarm_reward += self.config.success_reward
        info = dict(info)
        info.update(
            {
                "sarm_progress": progress,
                "sarm_prev_progress": self.prev_progress,
                "sarm_reward_delta": delta,
                "sarm_reward": float(sarm_reward),
                "sarm_succeed": succeed,
            }
        )
        self.prev_progress = progress
        if self.config.log_only:
            info["sarm_log_only"] = True
            return obs, env_reward, done, truncated, info
        done = bool(done or succeed)
        info["succeed"] = bool(succeed)
        return obs, float(sarm_reward), done, truncated, info

    def close(self):
        close = getattr(self.client, "close", None)
        if callable(close):
            close()
        return super().close()
