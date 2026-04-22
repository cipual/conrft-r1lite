from dataclasses import dataclass

import gymnasium as gym


@dataclass
class DualMangoBoxTaskConfig:
    task_name: str = "r1lite_dual_mango_box"
    task_desc: str = "左臂抓住白色的框放在右臂的周围，右臂抓住发红的芒果，把它放入框内。"
    reward_neg: float = -0.05
    success_reward: float = 10.0
    success_on_episode_end: bool = True


class DualMangoBoxTaskWrapper(gym.Wrapper):
    """
    Bootstrap task wrapper for dual-arm demo collection.

    The physical task success is object-level and currently has no detector in
    this repo. For demo collection, we mark an episode as successful when it
    ends if `success_on_episode_end` is enabled. Replace this with a visual or
    manual success signal before using sparse online RL rewards seriously.
    """

    def __init__(self, env, task_config: DualMangoBoxTaskConfig):
        super().__init__(env)
        self.task_config = task_config

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info = dict(info)
        info["task_name"] = self.task_config.task_name
        info["task_desc"] = self.task_config.task_desc
        info["succeed"] = False
        return obs, info

    def step(self, action):
        obs, _, done, truncated, info = self.env.step(action)
        info = dict(info)
        episode_finished = bool(done or truncated)
        succeed = bool(self.task_config.success_on_episode_end and episode_finished)
        reward = self.task_config.success_reward if succeed else self.task_config.reward_neg
        info["task_name"] = self.task_config.task_name
        info["task_desc"] = self.task_config.task_desc
        info["succeed"] = succeed
        info["reward_components"] = {
            "base": self.task_config.reward_neg,
            "success_bonus": self.task_config.success_reward if succeed else 0.0,
        }
        return obs, float(reward), done, truncated, info
