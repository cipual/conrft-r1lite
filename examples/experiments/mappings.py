from typing import Callable, Dict


def _load_pick_banana():
    # 仅在真正访问 Franka 任务时才导入，避免把 RealSense 依赖提前带进来。
    from experiments.task1_pick_banana.config import TrainConfig as PickBananaTrainConfig

    return PickBananaTrainConfig


def _load_r1lite_reach_target():
    # R1Lite 任务保持独立导入，避免被其他实验的硬件依赖误伤。
    from experiments.r1lite_reach_target.config import TrainConfig as R1LiteReachTargetTrainConfig

    return R1LiteReachTargetTrainConfig


def _load_r1lite_single_arm():
    from experiments.r1lite_single_arm.config import TrainConfig as R1LiteSingleArmTrainConfig

    return R1LiteSingleArmTrainConfig


def _load_r1lite_dual_mango_box():
    from experiments.r1lite_dual_mango_box.config import TrainConfig as R1LiteDualMangoBoxTrainConfig

    return R1LiteDualMangoBoxTrainConfig


_CONFIG_LOADERS: Dict[str, Callable[[], type]] = {
    "task1_pick_banana": _load_pick_banana,
    "r1lite_reach_target": _load_r1lite_reach_target,
    "r1lite_single_arm": _load_r1lite_single_arm,
    "r1lite_dual_mango_box": _load_r1lite_dual_mango_box,
}


class LazyConfigMapping(dict):
    """
    Resolve experiment configs on first access so unrelated experiments do not
    pull in optional hardware dependencies during import.
    """

    def __contains__(self, key):
        return key in _CONFIG_LOADERS or super().__contains__(key)

    def __getitem__(self, key):
        if key in _CONFIG_LOADERS:
            # 首次访问时再真正解析配置类，后续直接复用缓存结果。
            value = _CONFIG_LOADERS[key]()
            super().__setitem__(key, value)
            return value
        return super().__getitem__(key)

    def keys(self):
        return _CONFIG_LOADERS.keys()

    def items(self):
        for key in self.keys():
            yield key, self[key]

    def values(self):
        for _, value in self.items():
            yield value

    def get(self, key, default=None):
        if key in self:
            return self[key]
        return default


CONFIG_MAPPING = LazyConfigMapping()
