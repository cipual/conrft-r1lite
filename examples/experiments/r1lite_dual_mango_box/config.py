import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from r1lite_env import (
    DualR1LiteEnv,
    R1LiteEnvConfig,
    R1LiteObsWrapper,
    R1LiteTeleopInterventionWrapper,
)
from r1lite_env.sarm_reward import SARMProgressRewardWrapper, SARMRewardConfig
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper

from experiments.config import DefaultTrainingConfig
from experiments.r1lite_dual_mango_box.wrapper import (
    DualMangoBoxTaskConfig,
    DualMangoBoxTaskWrapper,
)


_CONFIG_DIR = Path(__file__).resolve().parent
_CONFIG_PATH = Path(os.environ.get("R1LITE_DUAL_MANGO_BOX_CONFIG", _CONFIG_DIR / "config.yaml"))
_TRAIN_STAGE = os.environ.get("R1LITE_TRAIN_STAGE", "offline").strip().lower()


def _load_user_config() -> dict:
    if not _CONFIG_PATH.exists():
        return {}
    with _CONFIG_PATH.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid config file format: {_CONFIG_PATH}")
    return data


_USER_CONFIG = _load_user_config()


def _cfg(path: str, default):
    current = _USER_CONFIG
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _train_cfg(path: str, default):
    if _TRAIN_STAGE == "online":
        offline_default = _cfg(f"offline_training.{path}", default)
        return _cfg(f"online_training.{path}", offline_default)
    return _cfg(f"offline_training.{path}", default)


def _runtime_defaults() -> dict:
    return {
        "checkpoint_path": _cfg("offline_training.checkpoint_path", "./conrft"),
        "demo_path": _cfg("offline_training.demo_path", "./demo_data/replace_me.pkl"),
        "pretrain_steps": int(_cfg("offline_training.pretrain_steps", 20000)),
        "pretrain_q_weight": float(_cfg("offline_training.pretrain.q_weight", 0.1)),
        "pretrain_bc_weight": float(_cfg("offline_training.pretrain.bc_weight", 1.0)),
        "learner_q_weight": float(_cfg("offline_training.learner.q_weight", 1.0)),
        "learner_bc_weight": float(_cfg("offline_training.learner.bc_weight", 0.1)),
        "debug": bool(_cfg("offline_training.debug", False)),
        "xla_mem_fraction_pretrain": float(_cfg("offline_training.pretrain.xla_mem_fraction", 0.85)),
        "xla_mem_fraction_learner": float(_cfg("offline_training.learner.xla_mem_fraction", 0.5)),
        "online_checkpoint_path": _cfg(
            "online_training.checkpoint_path",
            _cfg("offline_training.checkpoint_path", "./conrft"),
        ),
        "online_demo_path": _cfg(
            "online_training.demo_path",
            _cfg("offline_training.demo_path", "./demo_data/replace_me.pkl"),
        ),
        "online_pretrain_steps": int(
            _cfg("online_training.pretrain_steps", _cfg("offline_training.pretrain_steps", 20000))
        ),
        "online_q_weight": float(
            _cfg("online_training.learner.q_weight", _cfg("offline_training.learner.q_weight", 1.0))
        ),
        "online_bc_weight": float(
            _cfg("online_training.learner.bc_weight", _cfg("offline_training.learner.bc_weight", 0.1))
        ),
        "online_debug": bool(_cfg("online_training.debug", _cfg("offline_training.debug", False))),
        "xla_mem_fraction_online_learner": float(
            _cfg("online_training.learner.xla_mem_fraction", _cfg("offline_training.learner.xla_mem_fraction", 0.5))
        ),
        "xla_mem_fraction_actor": float(_cfg("online_training.actor.xla_mem_fraction", 0.2)),
    }


def get_runtime_default(key: str, default=None):
    return _runtime_defaults().get(key, default)


@dataclass
class EnvConfig(R1LiteEnvConfig):
    SERVER_URL: str = os.environ.get("ROBOT", _cfg("env.server_url", "http://127.0.0.1:8001/"))
    ACTION_SCALE: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                float(_cfg("control.xyz_scale", 0.03)),
                float(_cfg("control.rot_scale", 0.20)),
                1.0,
            ],
            dtype=np.float32,
        )
    )
    CONTROL_HZ: float = float(_cfg("control.hz", 10.0))
    LOG_EFFECTIVE_HZ: bool = bool(_cfg("control.debug_effective_hz", False))
    MAX_EPISODE_LENGTH: int = int(_cfg("env.max_episode_length", 1000))
    DEFAULT_MODE: str = _cfg("env.default_mode", "ee_pose_servo")
    DEFAULT_PRESET: str = _cfg("env.default_preset", "free_space")
    RANDOM_RESET: bool = bool(_cfg("env.random_reset", False))
    RANDOM_XY_RANGE: float = float(_cfg("env.random_xy_range", 0.015))
    RANDOM_RZ_RANGE: float = float(_cfg("env.random_rz_range", 0.15))
    RESET_SETTLE_SEC: float = float(_cfg("env.reset_settle_sec", 0.5))
    RESET_WAIT_TIMEOUT_SEC: float = float(_cfg("env.reset_wait_timeout_sec", 15.0))
    RESET_FAIL_ON_TIMEOUT: bool = bool(_cfg("env.reset_fail_on_timeout", True))
    RESET_POLL_SEC: float = float(_cfg("env.reset_poll_sec", 0.1))
    RESET_STABLE_COUNT: int = int(_cfg("env.reset_stable_count", 4))
    RESET_DEBUG: bool = bool(_cfg("env.reset_debug", False))
    RESET_POSITION_TOLERANCE_M: float = float(_cfg("env.reset_position_tolerance_m", 0.03))
    RESET_ORIENTATION_TOLERANCE_RAD: float = float(_cfg("env.reset_orientation_tolerance_rad", 0.35))
    RESET_JOINT_TOLERANCE_RAD: float = float(_cfg("env.reset_joint_tolerance_rad", 0.08))
    RESET_STABLE_POS_EPS_M: float = float(_cfg("env.reset_stable_pos_eps_m", 0.005))
    RESET_STABLE_ORI_EPS_RAD: float = float(_cfg("env.reset_stable_ori_eps_rad", 0.08))
    RESET_STABLE_JOINT_EPS_RAD: float = float(_cfg("env.reset_stable_joint_eps_rad", 0.03))
    RESET_RIGHT_POSE: list = field(default_factory=lambda: list(_cfg("env.reset_right_pose", [0.0] * 7)))
    RESET_LEFT_POSE: list = field(default_factory=lambda: list(_cfg("env.reset_left_pose", [0.0] * 7)))
    RESET_RIGHT_JOINT: list = field(default_factory=lambda: list(_cfg("env.reset_right_joint", [0.0] * 6)))
    RESET_LEFT_JOINT: list = field(default_factory=lambda: list(_cfg("env.reset_left_joint", [0.0] * 6)))
    ABS_POSE_LIMIT_LOW: dict = field(
        default_factory=lambda: {
            "left": list(_cfg("env.abs_pose_limit_low.left", [0.22, -0.42, 0.18, -np.pi, -np.pi, -np.pi])),
            "right": list(_cfg("env.abs_pose_limit_low.right", [-0.10, -0.85, 0.27, -np.pi, -np.pi, -np.pi])),
        }
    )
    ABS_POSE_LIMIT_HIGH: dict = field(
        default_factory=lambda: {
            "left": list(_cfg("env.abs_pose_limit_high.left", [0.55, -0.05, 0.45, np.pi, np.pi, np.pi])),
            "right": list(_cfg("env.abs_pose_limit_high.right", [0.60, -0.19, 0.70, np.pi, np.pi, np.pi])),
        }
    )
    FIX_GRIPPER_OPEN: bool = bool(_cfg("gripper.fixed_open", False))
    FIXED_GRIPPER_VALUE: float = float(_cfg("gripper.open_value", 100.0))


class TrainConfig(DefaultTrainingConfig):
    arm = _cfg("train.arm", "dual")
    image_keys = list(_cfg("train.image_keys", ["head", "left_wrist", "right_wrist"]))
    proprio_keys = [
        "left/tcp_pose",
        "left/tcp_vel",
        "left/joint_pos",
        "left/joint_vel",
        "left/gripper_pose",
        "right/tcp_pose",
        "right/tcp_vel",
        "right/joint_pos",
        "right/joint_vel",
        "right/gripper_pose",
        "torso_pos",
    ]
    batch_size = int(_train_cfg("batch_size", 256))
    replay_buffer_capacity = int(_train_cfg("replay_buffer_capacity", 200000))
    checkpoint_period = int(_train_cfg("checkpoint_period", 2000))
    cta_ratio = int(_train_cfg("cta_ratio", 2))
    random_steps = 0
    discount = float(_train_cfg("discount", 0.98))
    buffer_period = int(_train_cfg("buffer_period", 1000))
    training_starts = int(_train_cfg("training_starts", DefaultTrainingConfig.training_starts))
    steps_per_update = int(_train_cfg("steps_per_update", DefaultTrainingConfig.steps_per_update))
    log_period = int(_train_cfg("log_period", DefaultTrainingConfig.log_period))
    eval_period = int(_train_cfg("eval_period", DefaultTrainingConfig.eval_period))
    encoder_type = "resnet-pretrained"
    setup_mode = _cfg("train.setup_mode", "dual-arm-learned-gripper")
    reward_neg = float(_cfg("task.reward_neg", -0.05))
    task_desc = _cfg(
        "train.task_desc",
        "左臂抓住白色的框放在右臂的周围，右臂抓住发红的芒果，把它放入框内，然后左右机械臂复位。",
    )
    octo_path = os.environ.get("OCTO_PATH", _cfg("train.octo_path", "hf://rail-berkeley/octo-small-1.5"))
    task_config = DualMangoBoxTaskConfig(
        task_name=_cfg("task.task_name", "r1lite_dual_mango_box"),
        task_desc=_cfg("task.task_desc", task_desc),
        reward_neg=float(_cfg("task.reward_neg", -0.05)),
        success_reward=float(_cfg("task.success_reward", 10.0)),
        success_on_episode_end=bool(_cfg("task.success_on_episode_end", True)),
    )
    reward_model_config = SARMRewardConfig(
        enabled=bool(_cfg("reward_model.enabled", False)),
        log_only=bool(_cfg("reward_model.log_only", True)),
        endpoint_url=_cfg("reward_model.endpoint_url", None),
        checkpoint_path=_cfg("reward_model.checkpoint_path", None),
        head_mode=_cfg("reward_model.head_mode", "sparse"),
        image_key=_cfg("reward_model.image_key", "head"),
        success_threshold=float(_cfg("reward_model.success_threshold", 0.95)),
        success_reward=float(_cfg("reward_model.success_reward", 10.0)),
        reward_scale=float(_cfg("reward_model.reward_scale", 1.0)),
        reward_bias=float(_cfg("reward_model.reward_bias", 0.0)),
        reward_clip_low=float(_cfg("reward_model.reward_clip_low", -1.0)),
        reward_clip_high=float(_cfg("reward_model.reward_clip_high", 1.0)),
        timeout=float(_cfg("reward_model.timeout", 2.0)),
        task_desc=task_desc,
    )
    teleop_config = {
        "calibrate_seconds": float(_cfg("teleop.calibrate_seconds", 0.5)),
        "trans_deadzone": float(_cfg("teleop.trans_deadzone", 0.08)),
        "rot_deadzone": float(_cfg("teleop.rot_deadzone", 0.08)),
        "activate_threshold": float(_cfg("teleop.activate_threshold", 1e-3)),
        "release_threshold": float(_cfg("teleop.release_threshold", 1e-3)),
    }

    def get_environment(self, fake_env=False, save_video=False, classifier=False, stack_obs_num=2):
        env_config = EnvConfig()
        env = DualR1LiteEnv(config=env_config)
        if not fake_env:
            env = R1LiteTeleopInterventionWrapper(
                env,
                arm="dual",
                gripper_enabled=not env_config.FIX_GRIPPER_OPEN,
                **self.teleop_config,
            )
        env = DualMangoBoxTaskWrapper(env, self.task_config)
        if self.reward_model_config.enabled:
            env = SARMProgressRewardWrapper(env, self.reward_model_config)
        env = R1LiteObsWrapper(env)
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=stack_obs_num, act_exec_horizon=None)
        return env
