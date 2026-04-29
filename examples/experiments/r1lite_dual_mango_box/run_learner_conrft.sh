SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
export PYTHONPATH="${REPO_ROOT}/examples:${REPO_ROOT}/serl_robot_infra:${REPO_ROOT}/serl_launcher:${PYTHONPATH}"
export R1LITE_TRAIN_STAGE=online

CFG_MODULE=experiments.r1lite_dual_mango_box.config
cfg_value() {
    python -c "from ${CFG_MODULE} import get_runtime_default; value = get_runtime_default('${1}'); print(value if value is not None else '')"
}

CHECKPOINT_PATH=${CHECKPOINT_PATH:-$(cfg_value online_checkpoint_path)}
DEMO_PATH=${DEMO_PATH:-$(cfg_value online_demo_path)}
PRETRAIN_STEPS=${PRETRAIN_STEPS:-$(cfg_value online_pretrain_steps)}
Q_WEIGHT=${Q_WEIGHT:-$(cfg_value online_q_weight)}
BC_WEIGHT=${BC_WEIGHT:-$(cfg_value online_bc_weight)}
TRAIN_DEBUG=${TRAIN_DEBUG:-$(cfg_value online_debug)}

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-$(cfg_value xla_mem_fraction_online_learner)}
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

python "${REPO_ROOT}/examples/train_conrft_octo.py" "$@" \
    --exp_name=r1lite_dual_mango_box \
    --checkpoint_path="${CHECKPOINT_PATH}" \
    --q_weight="${Q_WEIGHT}" \
    --bc_weight="${BC_WEIGHT}" \
    --demo_path="${DEMO_PATH}" \
    --pretrain_steps="${PRETRAIN_STEPS}" \
    --debug="${TRAIN_DEBUG}" \
    --learner
