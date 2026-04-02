SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)

CHECKPOINT_PATH=${CHECKPOINT_PATH:-./conrft}
DEMO_PATH=${DEMO_PATH:-./demo_data/replace_me.pkl}

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.5}
# 预训练阶段同样依赖本地 serl 包，避免用户手动导出 PYTHONPATH。
export PYTHONPATH="${REPO_ROOT}/serl_robot_infra:${REPO_ROOT}/serl_launcher:${PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

python ../../train_conrft_octo.py "$@" \
    --exp_name=r1lite_reach_target \
    --checkpoint_path="${CHECKPOINT_PATH}" \
    --q_weight=0.1 \
    --bc_weight=1.0 \
    --demo_path="${DEMO_PATH}" \
    --pretrain_steps=20000 \
    --debug=False \
    --learner
