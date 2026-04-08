SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)

SUCCESS_COUNT=${SUCCESS_COUNT:-1}
OUTPUT_DIR=${OUTPUT_DIR:-./demo_data}
# 直接从实验目录执行时，也能找到本地 serl_robot_infra / serl_launcher 包。
export PYTHONPATH="${REPO_ROOT}/serl_robot_infra:${REPO_ROOT}/serl_launcher:${PYTHONPATH}"
# matplotlib 默认缓存目录在某些机器上不可写，这里显式切到 /tmp。
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

python ../../record_demos_r1lite_octo.py "$@" \
    --exp_name=r1lite_reach_target \
    --successes_needed="${SUCCESS_COUNT}" \
    --output_dir="${OUTPUT_DIR}"
