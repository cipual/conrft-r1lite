SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)

SUCCESS_COUNT=${SUCCESS_COUNT:-1}
OUTPUT_DIR=${OUTPUT_DIR:-./demo_data}
export PYTHONPATH="${REPO_ROOT}/serl_robot_infra:${REPO_ROOT}/serl_launcher:${PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

python ../../record_demos_r1lite_octo.py "$@" \
    --exp_name=r1lite_dual_mango_box \
    --successes_needed="${SUCCESS_COUNT}" \
    --output_dir="${OUTPUT_DIR}"
