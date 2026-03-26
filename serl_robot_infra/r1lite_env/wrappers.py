import copy

import gymnasium as gym
import numpy as np

from franka_env.spacemouse.spacemouse_expert import SpaceMouseExpert


class R1LiteObsWrapper(gym.ObservationWrapper):
    """
    Convert nested dual-arm R1 Lite observations into the flat state dict shape
    expected by existing SERL wrappers.
    """

    def __init__(self, env):
        super().__init__(env)
        state_space = {
            "left/tcp_pose": self.env.observation_space["state"]["left"]["tcp_pose"],
            "left/tcp_vel": self.env.observation_space["state"]["left"]["tcp_vel"],
            "left/tcp_force": self.env.observation_space["state"]["left"]["tcp_force"],
            "left/tcp_torque": self.env.observation_space["state"]["left"]["tcp_torque"],
            "left/gripper_pose": self.env.observation_space["state"]["left"]["gripper_pose"],
            "left/joint_pos": self.env.observation_space["state"]["left"]["joint_pos"],
            "left/joint_vel": self.env.observation_space["state"]["left"]["joint_vel"],
            "left/joint_effort": self.env.observation_space["state"]["left"]["joint_effort"],
            "right/tcp_pose": self.env.observation_space["state"]["right"]["tcp_pose"],
            "right/tcp_vel": self.env.observation_space["state"]["right"]["tcp_vel"],
            "right/tcp_force": self.env.observation_space["state"]["right"]["tcp_force"],
            "right/tcp_torque": self.env.observation_space["state"]["right"]["tcp_torque"],
            "right/gripper_pose": self.env.observation_space["state"]["right"]["gripper_pose"],
            "right/joint_pos": self.env.observation_space["state"]["right"]["joint_pos"],
            "right/joint_vel": self.env.observation_space["state"]["right"]["joint_vel"],
            "right/joint_effort": self.env.observation_space["state"]["right"]["joint_effort"],
            "torso": self.env.observation_space["state"]["torso"],
        }
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(state_space),
                "images": self.env.observation_space["images"],
                "meta": self.env.observation_space["meta"],
            }
        )

    def observation(self, observation):
        return {
            "state": {
                "left/tcp_pose": observation["state"]["left"]["tcp_pose"],
                "left/tcp_vel": observation["state"]["left"]["tcp_vel"],
                "left/tcp_force": observation["state"]["left"]["tcp_force"],
                "left/tcp_torque": observation["state"]["left"]["tcp_torque"],
                "left/gripper_pose": observation["state"]["left"]["gripper_pose"],
                "left/joint_pos": observation["state"]["left"]["joint_pos"],
                "left/joint_vel": observation["state"]["left"]["joint_vel"],
                "left/joint_effort": observation["state"]["left"]["joint_effort"],
                "right/tcp_pose": observation["state"]["right"]["tcp_pose"],
                "right/tcp_vel": observation["state"]["right"]["tcp_vel"],
                "right/tcp_force": observation["state"]["right"]["tcp_force"],
                "right/tcp_torque": observation["state"]["right"]["tcp_torque"],
                "right/gripper_pose": observation["state"]["right"]["gripper_pose"],
                "right/joint_pos": observation["state"]["right"]["joint_pos"],
                "right/joint_vel": observation["state"]["right"]["joint_vel"],
                "right/joint_effort": observation["state"]["right"]["joint_effort"],
                "torso": observation["state"]["torso"],
            },
            "images": observation["images"],
            "meta": observation["meta"],
        }

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info


class R1LiteTeleopInterventionWrapper(gym.ActionWrapper):
    """
    Map SpaceMouse teleop onto the dual-arm high-level action interface.
    One device controls one arm, two devices control both arms.
    """

    def __init__(self, env, action_indices=None):
        super().__init__(env)
        self.expert = SpaceMouseExpert()
        self.action_indices = action_indices
        self.left1 = self.left2 = self.right1 = self.right2 = False

    def action(self, action: np.ndarray) -> np.ndarray:
        expert_a, buttons = self.expert.get_action()
        intervened = False
        self.left1, self.left2, self.right1, self.right2 = tuple(buttons) if len(buttons) == 4 else (False, False, False, False)
        mapped = np.zeros_like(action)
        if len(expert_a) >= 12 and mapped.shape[0] >= 14:
            mapped[:6] = expert_a[:6]
            mapped[7:13] = expert_a[6:12]
            if self.left1:
                mapped[6] = -1.0
            elif self.left2:
                mapped[6] = 1.0
            if self.right1:
                mapped[13] = -1.0
            elif self.right2:
                mapped[13] = 1.0
            intervened = np.linalg.norm(mapped) > 1e-3
        elif len(expert_a) >= 6 and mapped.shape[0] >= 7:
            mapped[:6] = expert_a[:6]
            if self.left1:
                mapped[6] = -1.0
            elif self.left2:
                mapped[6] = 1.0
            intervened = np.linalg.norm(mapped) > 1e-3

        if self.action_indices is not None:
            filtered = np.zeros_like(mapped)
            filtered[self.action_indices] = mapped[self.action_indices]
            mapped = filtered

        return (mapped, True) if intervened else (action, False)

    def step(self, action):
        new_action, replaced = self.action(action)
        obs, rew, done, truncated, info = self.env.step(new_action)
        info = copy.deepcopy(info)
        if replaced:
            info["intervene_action"] = new_action
        info["left1"] = self.left1
        info["left2"] = self.left2
        info["right1"] = self.right1
        info["right2"] = self.right2
        return obs, rew, done, truncated, info
