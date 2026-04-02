SUCCESS_COUNT=${SUCCESS_COUNT:-20}
OUTPUT_DIR=${OUTPUT_DIR:-./demo_data}

python ../../record_demos_r1lite_octo.py "$@" \
    --exp_name=r1lite_reach_target \
    --successes_needed="${SUCCESS_COUNT}" \
    --output_dir="${OUTPUT_DIR}"
