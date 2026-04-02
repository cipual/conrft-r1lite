CHECKPOINT_PATH=${CHECKPOINT_PATH:-./conrft}

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.2}

python ../../train_conrft_octo.py "$@" \
    --exp_name=r1lite_reach_target \
    --checkpoint_path="${CHECKPOINT_PATH}" \
    --actor
