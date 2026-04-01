from experiments.task1_pick_banana.config import TrainConfig as PickBananaTrainConfig
from experiments.r1lite_reach_target.config import TrainConfig as R1LiteReachTargetTrainConfig
from experiments.r1lite_single_arm.config import TrainConfig as R1LiteSingleArmTrainConfig

CONFIG_MAPPING = {
    "task1_pick_banana": PickBananaTrainConfig,
    "r1lite_reach_target": R1LiteReachTargetTrainConfig,
    "r1lite_single_arm": R1LiteSingleArmTrainConfig,
}
