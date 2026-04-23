import copy
import time

import gymnasium as gym
import numpy as np

from franka_env.spacemouse.spacemouse_expert import SpaceMouseExpert
from r1lite_env.spacemouse_teleop import _apply_deadzone


def _estimate_idle_bias_robust(expert: SpaceMouseExpert, duration_sec: float) -> np.ndarray:
    if duration_sec <= 0.0:
        action, _ = expert.get_action()
        return np.zeros_like(np.asarray(action, dtype=np.float64))

    deadline = time.time() + duration_sec
    samples = []
    while time.time() < deadline:
        action, _ = expert.get_action()
        samples.append(np.asarray(action, dtype=np.float64))
        time.sleep(0.01)
    if not samples:
        action, _ = expert.get_action()
        return np.zeros_like(np.asarray(action, dtype=np.float64))
    # 用中位数而不是均值做零点估计，对偶发尖峰和启动抖动更稳。
    return np.median(np.stack(samples, axis=0), axis=0)


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
            "torso_pos": gym.spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
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
                "torso_pos": np.asarray(observation["state"]["torso"], dtype=np.float32).reshape(-1)[:1],
            },
            "images": observation["images"],
            "meta": observation["meta"],
        }

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info


class R1LiteSingleArmConRFTObsWrapper(gym.ObservationWrapper):
    """
    Adapt single-arm R1 Lite observations to the state/images structure expected
    by SERLObsWrapper and ConRFT.
    """

    def __init__(self, env, image_aliases=None):
        super().__init__(env)
        default_aliases = {
            "head": "image_primary",
            "left_wrist": "image_wrist",
            "right_wrist": "image_wrist",
        }
        self.image_aliases = image_aliases or default_aliases

        state_space = self.env.observation_space["state"]
        image_space = self.env.observation_space["images"]
        remapped_images = {}
        for source_key, target_key in self.image_aliases.items():
            if source_key in image_space.spaces and target_key not in remapped_images:
                remapped_images[target_key] = image_space[source_key]

        self.observation_space = gym.spaces.Dict(
            {
                "state": state_space,
                "images": gym.spaces.Dict(remapped_images),
            }
        )

    def observation(self, observation):
        images = {}
        for source_key, target_key in self.image_aliases.items():
            if source_key in observation["images"] and target_key not in images:
                images[target_key] = observation["images"][source_key]
        return {
            "state": observation["state"],
            "images": images,
        }

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info


class R1LiteTeleopInterventionWrapper(gym.ActionWrapper):
    """
    Map SpaceMouse teleop onto the dual-arm high-level action interface.
    One device controls a selected arm, two devices control both arms.
    """

    def __init__(
        self,
        env,
        action_indices=None,
        arm="right",
        gripper_enabled=True,
        calibrate_seconds=1.0,
        trans_deadzone=0.12,
        rot_deadzone=0.12,
        activate_threshold=0.15,
        release_threshold=0.05,
    ):
        super().__init__(env)
        assert arm in ("left", "right", "dual")
        self.expert = SpaceMouseExpert()
        self.action_indices = action_indices
        self.arm = arm
        self.gripper_enabled = gripper_enabled
        self.trans_deadzone = trans_deadzone
        self.rot_deadzone = rot_deadzone
        self.activate_threshold = activate_threshold
        self.release_threshold = release_threshold
        self.intervention_active = False
        self.left1 = self.left2 = self.right1 = self.right2 = False
        # demo 录制更怕零飘，这里使用更稳的中位数校准，并适当拉长校准时间。
        print(
            f"[teleop-wrapper] calibrating SpaceMouse for {calibrate_seconds:.2f}s, keep it untouched..."
        )
        self.bias = _estimate_idle_bias_robust(self.expert, calibrate_seconds)
        print("[teleop-wrapper] calibration complete")
        print(
            "[teleop-wrapper] bias="
            f"{np.array2string(self.bias, precision=4, suppress_small=True)} "
            f"deadzone(trans={self.trans_deadzone:.3f}, rot={self.rot_deadzone:.3f}) "
            f"hysteresis(activate={self.activate_threshold:.3f}, release={self.release_threshold:.3f})"
        )

    def action(self, action: np.ndarray) -> np.ndarray:
        expert_a, buttons = self.expert.get_action()
        # 先去掉静置偏置，再套用 deadzone，和 spacemouse_teleop 的处理逻辑一致。
        expert_a = _apply_deadzone(
            np.asarray(expert_a, dtype=np.float64) - self.bias,
            trans_deadzone=self.trans_deadzone,
            rot_deadzone=self.rot_deadzone,
        )
        mapped = np.zeros_like(action)
        self.left1 = self.left2 = self.right1 = self.right2 = False

        if self.arm == "dual":
            if len(expert_a) < 12 or mapped.shape[0] < 14:
                return action, False
            mapped[:6] = expert_a[:6]
            mapped[7:13] = expert_a[6:12]
            if len(buttons) == 4:
                self.left1, self.left2, self.right1, self.right2 = tuple(buttons)
            if self.gripper_enabled:
                if self.left1:
                    mapped[6] = -1.0
                elif self.left2:
                    mapped[6] = 1.0
                if self.right1:
                    mapped[13] = -1.0
                elif self.right2:
                    mapped[13] = 1.0
        else:
            if len(expert_a) < 6:
                return action, False
            if mapped.shape[0] >= 14:
                if self.arm == "left":
                    mapped[:6] = expert_a[:6]
                else:
                    mapped[7:13] = expert_a[:6]
            elif mapped.shape[0] >= 7:
                mapped[:6] = expert_a[:6]
            else:
                return action, False

            close_pressed = len(buttons) >= 1 and buttons[0]
            open_pressed = len(buttons) >= 2 and buttons[-1]
            if self.arm == "left":
                self.left1, self.left2 = close_pressed, open_pressed
                if self.gripper_enabled:
                    if mapped.shape[0] >= 14:
                        if self.left1:
                            mapped[6] = -1.0
                        elif self.left2:
                            mapped[6] = 1.0
                    elif mapped.shape[0] >= 7:
                        if self.left1:
                            mapped[6] = -1.0
                        elif self.left2:
                            mapped[6] = 1.0
            else:
                self.right1, self.right2 = close_pressed, open_pressed
                if self.gripper_enabled:
                    if mapped.shape[0] >= 14:
                        if self.right1:
                            mapped[13] = -1.0
                        elif self.right2:
                            mapped[13] = 1.0
                    elif mapped.shape[0] >= 7:
                        if self.right1:
                            mapped[6] = -1.0
                        elif self.right2:
                            mapped[6] = 1.0

        motion_norm = float(np.linalg.norm(mapped))
        # fixed-gripper 任务下忽略 SpaceMouse 按钮，避免无意触发 intervention。
        button_pressed = self.gripper_enabled and any(bool(v) for v in buttons)
        # 使用滞回阈值抑制零飘：只有明显输入才接管，释放时则更早回到 policy。
        if self.intervention_active:
            intervened = button_pressed or motion_norm > self.release_threshold
        else:
            intervened = button_pressed or motion_norm > self.activate_threshold
        self.intervention_active = intervened

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

    def close(self):
        self.expert.close()
        return super().close()
