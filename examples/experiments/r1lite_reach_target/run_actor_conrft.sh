SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
export PYTHONPATH="${REPO_ROOT}/examples:${REPO_ROOT}/serl_robot_infra:${REPO_ROOT}/serl_launcher:${PYTHONPATH}"
export R1LITE_TRAIN_STAGE=online

CFG_MODULE=experiments.r1lite_reach_target.config
cfg_value() {
    python -c "from ${CFG_MODULE} import get_runtime_default; value = get_runtime_default('${1}'); print(value if value is not None else '')"
}

CHECKPOINT_PATH=${CHECKPOINT_PATH:-$(cfg_value online_checkpoint_path)}

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-$(cfg_value xla_mem_fraction_actor)}
# 保证从实验子目录启动时，本地包解析路径正确。
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

python ../../train_conrft_octo.py "$@" \
    --exp_name=r1lite_reach_target \
    --checkpoint_path="${CHECKPOINT_PATH}" \
    --actor
