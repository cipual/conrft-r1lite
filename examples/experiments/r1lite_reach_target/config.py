import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

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


_CONFIG_DIR = Path(__file__).resolve().parent
_CONFIG_PATH = Path(os.environ.get("R1LITE_REACH_CONFIG", _CONFIG_DIR / "config.yaml"))
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
    """按训练阶段读取参数，online 缺省时自动回落到 offline。"""
    if _TRAIN_STAGE == "online":
        offline_default = _cfg(f"offline_training.{path}", default)
        return _cfg(f"online_training.{path}", offline_default)
    return _cfg(f"offline_training.{path}", default)


def _runtime_defaults() -> dict:
    """集中返回脚本运行时默认值，供 bash 启动脚本和 Python 侧共用。"""
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
        # 在线阶段默认允许显式覆盖 offline 阶段的 checkpoint/demo 配置。
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
    """给 shell 启动脚本读取 YAML 默认值用，避免在多个脚本里写死。"""
    return _runtime_defaults().get(key, default)


@dataclass
class EnvConfig(R1LiteEnvConfig):
    # 优先从环境变量 ROBOT 读取机器人服务地址，便于在不同机器之间切换。
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
    MAX_EPISODE_LENGTH: int = int(_cfg("env.max_episode_length", 80))
    DEFAULT_MODE: str = _cfg("env.default_mode", "ee_pose_servo")
    DEFAULT_PRESET: str = _cfg("env.default_preset", "free_space")
    RANDOM_RESET: bool = bool(_cfg("env.random_reset", False))
    RANDOM_XY_RANGE: float = float(_cfg("env.random_xy_range", 0.015))
    RANDOM_RZ_RANGE: float = float(_cfg("env.random_rz_range", 0.15))
    RESET_SETTLE_SEC: float = float(_cfg("env.reset_settle_sec", 1.5))
    RESET_WAIT_TIMEOUT_SEC: float = float(_cfg("env.reset_wait_timeout_sec", 6.0))
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
    # 这里约定“全 0”表示不要发末端 pose reset，而是回落到服务端默认关节 reset。
    RESET_RIGHT_POSE: list = field(default_factory=lambda: list(_cfg("env.reset_right_pose", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])))
    RESET_LEFT_POSE: list = field(default_factory=lambda: list(_cfg("env.reset_left_pose", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])))
    RESET_RIGHT_JOINT: list = field(default_factory=lambda: list(_cfg("env.reset_right_joint", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])))
    RESET_LEFT_JOINT: list = field(default_factory=lambda: list(_cfg("env.reset_left_joint", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])))
    ABS_POSE_LIMIT_LOW: dict = field(
        default_factory=lambda: {
            "left": list(_cfg("env.abs_pose_limit_low.left", [0.22, -0.42, 0.18, 0.0, -1.2, -1.2])),
            "right": list(_cfg("env.abs_pose_limit_low.right", [-0.10, -0.85, 0.27, 0.0, 0.0, -1.2])),
        }
    )
    ABS_POSE_LIMIT_HIGH: dict = field(
        default_factory=lambda: {
            "left": list(_cfg("env.abs_pose_limit_high.left", [0.55, -0.05, 0.45, float(np.pi), 1.2, 1.2])),
            "right": list(_cfg("env.abs_pose_limit_high.right", [0.60, -0.19, 0.70, float(np.pi), 1.4, 1.2])),
        }
    )
    FIX_GRIPPER_OPEN: bool = bool(_cfg("gripper.fixed_open", True))
    FIXED_GRIPPER_VALUE: float = float(_cfg("gripper.open_value", 100.0))


class TrainConfig(DefaultTrainingConfig):
    arm = _cfg("train.arm", "right")
    image_keys = list(_cfg("train.image_keys", ["image_primary", "image_wrist"])) # 头部相机 + 腕部相机
    proprio_keys = ["gripper_pose", "tcp_force", "tcp_pose", "tcp_torque", "tcp_vel"]
    # 这些参数直接进入训练循环，按 offline / online 阶段分别读取。
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
    # 参考 Franka reach / pregrasp 场景：当前任务不学习夹爪，策略只关心 6DoF 末端动作。
    setup_mode = _cfg("train.setup_mode", "single-arm-fixed-gripper")
    reward_neg = -0.05
    task_desc = _cfg("train.task_desc", "Move the R1Lite end effector to the target pose") # 输入 vla 的 instruction
    # 优先使用外部显式指定的 Octo checkpoint；未指定时走 Octo 官方推荐的 HF 路径。
    # `OctoModel.load_pretrained()` 会自动下载并缓存到当前用户可读目录。
    octo_path = os.environ.get("OCTO_PATH", _cfg("train.octo_path", "hf://rail-berkeley/octo-small-1.5"))
    task_config = ReachTargetTaskConfig(
        arm=_cfg("train.arm", "right"),
        target_left_pose=tuple(_cfg("task.target_left_pose", [0.43, -0.20, 0.28, 0.0, 1.0, 0.0, 0.0])),
        target_right_pose=tuple(_cfg("task.target_right_pose", [0.332, -0.357, 0.280, 0.011, 0.656, -0.019, 0.754])),
        position_tolerance_m=float(_cfg("task.position_tolerance_m", 0.03)),
        orientation_tolerance_rad=float(_cfg("task.orientation_tolerance_rad", 0.35)),
        success_reward=float(_cfg("task.success_reward", 10.0)),
        dense_position_weight=float(_cfg("task.dense_position_weight", 1.0)),
        dense_orientation_weight=float(_cfg("task.dense_orientation_weight", 0.1)),
        reward_neg=float(_cfg("task.reward_neg", -0.05)),
    )
    teleop_config = {
        "calibrate_seconds": float(_cfg("teleop.calibrate_seconds", 0.5)),
        "trans_deadzone": float(_cfg("teleop.trans_deadzone", 0.08)),
        "rot_deadzone": float(_cfg("teleop.rot_deadzone", 0.08)),
        # 对齐独立 teleop：默认几乎不做额外 intervention 滞回。
        "activate_threshold": float(_cfg("teleop.activate_threshold", 1e-3)),
        "release_threshold": float(_cfg("teleop.release_threshold", 1e-3)),
    }

    def get_environment(self, fake_env=False, save_video=False, classifier=False, stack_obs_num=2):
        env = R1LiteArmEnv(arm=self.arm, config=EnvConfig())
        if not fake_env:
            env = R1LiteTeleopInterventionWrapper(
                env,
                arm=self.arm,
                gripper_enabled=not EnvConfig().FIX_GRIPPER_OPEN,
                **self.teleop_config,
            )
        env = ReachTargetRewardWrapper(
            env,
            self.task_config,
        )
        env = R1LiteSingleArmConRFTObsWrapper(env)
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=stack_obs_num, act_exec_horizon=None)
        return env
