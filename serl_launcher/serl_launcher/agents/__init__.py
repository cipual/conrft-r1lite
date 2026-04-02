from .continuous.bc import BCAgent
from .continuous.ddpm_bc import DDPMBCAgent
from .continuous.sac import SACAgent
from .continuous.sac_single import SACAgentSingleArm

# 这里只暴露仓库中真实存在的 agent，避免导入阶段被失效路径拖死。
agents = {
    "bc": BCAgent,
    "ddpm_bc": DDPMBCAgent,
    "sac": SACAgent,
    "sac_single": SACAgentSingleArm,
}
