from dataclasses import dataclass

from r1lite_env import R1LiteArmEnv, R1LiteEnvConfig, R1LiteSingleArmConRFTObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper

from experiments.config import DefaultTrainingConfig


@dataclass
class EnvConfig(R1LiteEnvConfig):
    SERVER_URL: str = "http://127.0.0.1:8001/"
    MAX_EPISODE_LENGTH: int = 100
    DEFAULT_MODE: str = "ee_pose_servo"
    DEFAULT_PRESET: str = "free_space"


class TrainConfig(DefaultTrainingConfig):
    arm = "right"
    image_keys = ["image_primary", "image_wrist"]
    proprio_keys = ["gripper_pose", "tcp_force", "tcp_pose", "tcp_torque", "tcp_vel"]
    checkpoint_period = 2000
    cta_ratio = 2
    random_steps = 0
    discount = 0.98
    buffer_period = 1000
    encoder_type = "resnet-pretrained"
    setup_mode = "single-arm-learned-gripper"
    reward_neg = 0.0
    task_desc = "Control the R1Lite arm to complete the target manipulation task"
    octo_path = "/root/online_rl/octo_model/octo-small"

    def get_environment(self, fake_env=False, save_video=False, classifier=False, stack_obs_num=2):
        env = R1LiteArmEnv(arm=self.arm, config=EnvConfig())
        env = R1LiteSingleArmConRFTObsWrapper(env)
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=stack_obs_num, act_exec_horizon=None)
        return env
