SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)

CHECKPOINT_PATH=${CHECKPOINT_PATH:-./conrft}

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.2}
# 保证从实验子目录启动时，本地包解析路径正确。
export PYTHONPATH="${REPO_ROOT}/serl_robot_infra:${REPO_ROOT}/serl_launcher:${PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

python ../../train_conrft_octo.py "$@" \
    --exp_name=r1lite_reach_target \
    --checkpoint_path="${CHECKPOINT_PATH}" \
    --actor
