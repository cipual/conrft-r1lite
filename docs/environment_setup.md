# Quick Start: Environment Setup

This is the shortest path to bring up the R1Lite ConRFT + SARM workflow on a
new machine.

Assumptions:

- Ubuntu 22.04
- NVIDIA driver is installed and `nvidia-smi` works
- Miniforge is installed at `/home/robot/Applications/miniforge3`
- workspace root is `/home/robot/VLA-RL`

## 1. Clone Repositories

Use the patched LeRobot fork. It contains the SARM fixes needed by this project.

```bash
mkdir -p /home/robot/VLA-RL
cd /home/robot/VLA-RL

git clone git@github.com:cipual/conrft-r1lite.git
git clone https://github.com/cipual/lerobot.git
```

## 2. Install System Packages

```bash
sudo apt update
sudo apt install -y \
  git curl wget build-essential pkg-config \
  ffmpeg libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
  libhidapi-dev libusb-1.0-0-dev
```

## 3. Create `RWRL`

Use `RWRL` for ConRFT, Octo embeddings, robot envs, replay, offline pretraining,
and online RL.

```bash
source /home/robot/Applications/miniforge3/etc/profile.d/conda.sh
conda create -n RWRL python=3.10 -y
conda activate RWRL
python -m pip install --upgrade pip setuptools wheel

cd /home/robot/VLA-RL/conrft-r1lite
python -m pip install -e ./serl_launcher
python -m pip install -e ./serl_robot_infra
python -m pip install -e ./octo

python -m pip install --upgrade \
  numpy==1.26.4 \
  protobuf==4.25.3 \
  transformers==4.38.2

python -m pip install --upgrade --upgrade-strategy eager \
  "jax[cuda12]==0.4.38" \
  flax==0.10.2 \
  optax==0.2.4 \
  chex==0.1.88 \
  orbax-checkpoint==0.10.2

python -m pip install --upgrade \
  tensorflow==2.15.0 \
  tensorflow-probability==0.23.0 \
  tensorflow-hub==0.16.1 \
  tensorflow-text==2.15.0 \
  tensorflow-datasets==4.9.2 \
  tensorflow-graphics==2021.12.3 \
  tf-keras \
  gymnasium==0.29.1 \
  opencv-python==4.13.0.92 \
  scipy==1.15.3 \
  wandb==0.25.1
```

Sanity check:

```bash
python - <<'PY'
import jax, flax, optax, gymnasium, transformers
print("JAX devices:", jax.devices())
print("flax:", flax.__version__)
print("optax:", optax.__version__)
print("gymnasium:", gymnasium.__version__)
print("transformers:", transformers.__version__)
PY
```

## 4. Create `lerobot`

Use `lerobot` for RAW rosbag export, LeRobot datasets, SARM annotation, SARM
training, and SARM progress computation.

```bash
source /home/robot/Applications/miniforge3/etc/profile.d/conda.sh
conda create -n lerobot python=3.12 -y
conda activate lerobot
python -m pip install --upgrade pip setuptools wheel

cd /home/robot/VLA-RL/lerobot
python -m pip install -e .

python -m pip install --upgrade \
  torch==2.10.0 \
  torchvision==0.25.0 \
  transformers==5.3.0 \
  accelerate==1.13.0 \
  datasets==4.8.4 \
  pandas==2.3.3 \
  pyarrow==23.0.1 \
  av==15.1.0 \
  opencv-python-headless==4.13.0.92 \
  scipy==1.17.1
```

Sanity check:

```bash
python - <<'PY'
import torch, transformers, datasets, pandas, pyarrow, av
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
print("transformers:", transformers.__version__)
print("datasets:", datasets.__version__)
print("pandas:", pandas.__version__)
print("pyarrow:", pyarrow.__version__)
print("av:", av.__version__)
PY
```

## 5. Common Environment Variables

For robot / RL work in `RWRL`:

```bash
export ROBOT=http://192.168.12.12:8001
export MPLCONFIGDIR=/tmp/matplotlib
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

For offline training with local Hugging Face cache only:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
```

For SARM / LeRobot work:

```bash
export HF_LEROBOT_HOME=/home/robot/VLA-RL/conrft-r1lite/data/lerobot
export PYTHONPATH=/home/robot/VLA-RL/lerobot/src:$PYTHONPATH
```

## 6. Minimal Workflow Checks

Check ConRFT env shape:

```bash
conda activate RWRL
cd /home/robot/VLA-RL/conrft-r1lite
python - <<'PY'
import sys
sys.path[:0] = [
    "/home/robot/VLA-RL/conrft-r1lite/examples",
    "/home/robot/VLA-RL/conrft-r1lite/serl_robot_infra",
    "/home/robot/VLA-RL/conrft-r1lite/serl_launcher",
]
from experiments.r1lite_dual_mango_box.config import TrainConfig
env = TrainConfig().get_environment(fake_env=True)
print(env.observation_space["state"].shape, env.action_space.shape)
env.close()
PY
```

Expected:

```text
(2, 53) (14,)
```

Check SARM scripts:

```bash
conda activate lerobot
cd /home/robot/VLA-RL/conrft-r1lite
python examples/sarm/export_rosbag_to_lerobot_sarm.py --help
python examples/sarm/relabel_rosbag_or_conrft_with_sarm_reward.py --help

cd /home/robot/VLA-RL/lerobot
python -m lerobot.policies.sarm.compute_rabc_weights --help
```

## 7. Notes

- `cipual/lerobot` includes the local SARM patches: CLIP output compatibility,
  headless Matplotlib, CJK font handling, and local-only progress export by
  default.
- `RWRL` and `lerobot` should stay separate. Mixing their Torch/JAX/Transformers
  stacks is the fastest way to make the environment spicy in all the wrong ways.
- Data, checkpoints, videos, and generated pkl files live under
  `/home/robot/VLA-RL/conrft-r1lite/data` or `examples/*/outputs`; they are not
  tracked by git.
