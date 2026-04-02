from dataclasses import dataclass

import numpy as np

from r1lite_env import (
    R1LiteArmEnv,
    R1LiteEnvConfig,
    R1LiteSingleArmConRFTObsWrapper,
    R1LiteTeleopInterventionWrapper,
)
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper

from experiments.config import DefaultTrainingConfig
from experiments.r1lite_reach_target.wrapper import ReachTargetRewardWrapper, ReachTargetTaskConfig


@dataclass
class EnvConfig(R1LiteEnvConfig):
    SERVER_URL: str = "http://127.0.0.1:8001/"
    MAX_EPISODE_LENGTH: int = 80
    DEFAULT_MODE: str = "ee_pose_servo"
    DEFAULT_PRESET: str = "free_space"
    RANDOM_RESET: bool = False
    RANDOM_XY_RANGE: float = 0.015
    RANDOM_RZ_RANGE: float = 0.15
    RESET_RIGHT_POSE: list = [0.35, 0.25, 0.32, 0.0, 1.0, 0.0, 0.0]
    RESET_LEFT_POSE: list = [0.35, -0.25, 0.32, 0.0, 1.0, 0.0, 0.0]
    ABS_POSE_LIMIT_LOW = {
        "left": [0.22, -0.42, 0.18, 0.0, -1.2, -1.2],
        "right": [0.22, 0.05, 0.18, 0.0, -1.2, -1.2],
    }
    ABS_POSE_LIMIT_HIGH = {
        "left": [0.55, -0.05, 0.45, np.pi, 1.2, 1.2],
        "right": [0.55, 0.42, 0.45, np.pi, 1.2, 1.2],
    }


class TrainConfig(DefaultTrainingConfig):
    arm = "right"
    image_keys = ["image_primary", "image_wrist"]
    proprio_keys = ["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose"]
    checkpoint_period = 2000
    cta_ratio = 2
    random_steps = 0
    discount = 0.98
    buffer_period = 1000
    encoder_type = "resnet-pretrained"
    setup_mode = "single-arm-learned-gripper"
    reward_neg = -0.05
    task_desc = "Move the R1Lite end effector to the target pose"
    octo_path = "/root/online_rl/octo_model/octo-small"

    def get_environment(self, fake_env=False, save_video=False, classifier=False, stack_obs_num=2):
        env = R1LiteArmEnv(arm=self.arm, config=EnvConfig())
        if not fake_env:
            env = R1LiteTeleopInterventionWrapper(env, arm=self.arm)
        env = ReachTargetRewardWrapper(
            env,
            ReachTargetTaskConfig(
                arm=self.arm,
                reward_neg=self.reward_neg,
            ),
        )
        env = R1LiteSingleArmConRFTObsWrapper(env)
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=stack_obs_num, act_exec_horizon=None)
        return env
