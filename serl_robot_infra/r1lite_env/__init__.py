from r1lite_env.client import R1LiteClient

__all__ = ["R1LiteClient"]

try:
    from r1lite_env.envs import DualR1LiteEnv, R1LiteArmEnv, R1LiteEnvConfig
    from r1lite_env.spacemouse_teleop import run as run_spacemouse_teleop
    from r1lite_env.wrappers import R1LiteObsWrapper, R1LiteTeleopInterventionWrapper

    __all__ += [
        "R1LiteArmEnv",
        "DualR1LiteEnv",
        "R1LiteEnvConfig",
        "run_spacemouse_teleop",
        "R1LiteObsWrapper",
        "R1LiteTeleopInterventionWrapper",
    ]
except ModuleNotFoundError:
    # Robot-side acceptance checks only need the HTTP client.
    # Full env imports stay optional until the inference environment is installed.
    pass
