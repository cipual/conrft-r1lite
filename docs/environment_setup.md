# Environment Setup

This document records the current known-good setup for this project on a fresh
machine.

It targets the following split workflow:

- `RWRL` conda env: ConRFT / Octo / robot env / offline+online RL
- `lerobot` conda env: LeRobot dataset export, SARM annotation, SARM training,
  progress computation

It is based on the current local workspace:

- `conrft-r1lite`: commit `059f175`
- `lerobot`: upstream commit `a07f22e`, plus local SARM compatibility patches

## 1. Scope And Assumptions

This guide assumes:

- Ubuntu 22.04
- NVIDIA GPU is available
- Miniforge is installed at `/home/robot/Applications/miniforge3`
- code will live under `/home/robot/VLA-RL`

Current reference machine:

- OS: Ubuntu 22.04.5 LTS
- GPU: NVIDIA GeForce RTX 5080
- Driver: 570.172.08

This guide does not try to install the full CUDA driver stack from scratch. It
assumes `nvidia-smi` already works.

## 2. Repository Layout

Recommended layout:

```text
/home/robot/VLA-RL/
├── conrft-r1lite/
└── lerobot/
```

Clone:

```bash
mkdir -p /home/robot/VLA-RL
cd /home/robot/VLA-RL

git clone git@github.com:cipual/conrft-r1lite.git
git clone https://github.com/huggingface/lerobot.git
```

## 3. Base System Packages

Install common build/runtime dependencies first:

```bash
sudo apt update
sudo apt install -y \
  git curl wget build-essential pkg-config \
  ffmpeg libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
  libhidapi-dev libusb-1.0-0-dev
```

Notes:

- `ffmpeg` is needed for LeRobot video export / preview.
- `hidapi` / `libusb` are needed for `pyspacemouse`.
- OpenCV usually works with the libraries above.

## 4. Conda Initialization

```bash
source /home/robot/Applications/miniforge3/etc/profile.d/conda.sh
conda config --set auto_activate_base false
```

## 5. Create The `RWRL` Environment

Create the base env:

```bash
source /home/robot/Applications/miniforge3/etc/profile.d/conda.sh
conda create -n RWRL python=3.10 -y
conda activate RWRL
python -m pip install --upgrade pip setuptools wheel
```

Install the local project packages in editable mode:

```bash
cd /home/robot/VLA-RL/conrft-r1lite

python -m pip install -e ./serl_launcher
python -m pip install -e ./serl_robot_infra
python -m pip install -e ./octo
```

Install the known-good core versions used in the current environment:

```bash
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

Current key package versions in `RWRL`:

```text
Python 3.10.20
jax==0.4.38
jaxlib==0.4.38
flax==0.10.2
optax==0.2.4
chex==0.1.88
orbax-checkpoint==0.10.2
tensorflow==2.15.0
transformers==4.38.2
gymnasium==0.29.1
opencv-python==4.13.0.92
```

### `RWRL` sanity check

```bash
conda activate RWRL
python - <<'PY'
import jax, flax, optax, gymnasium, transformers
print("JAX devices:", jax.devices())
print("flax:", flax.__version__)
print("optax:", optax.__version__)
print("gymnasium:", gymnasium.__version__)
print("transformers:", transformers.__version__)
PY
```

## 6. Create The `lerobot` Environment

Create the base env:

```bash
source /home/robot/Applications/miniforge3/etc/profile.d/conda.sh
conda create -n lerobot python=3.12 -y
conda activate lerobot
python -m pip install --upgrade pip setuptools wheel
```

Install LeRobot editable:

```bash
cd /home/robot/VLA-RL/lerobot
python -m pip install -e .
```

Install the known-good package set used in the current environment:

```bash
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

Current key package versions in `lerobot`:

```text
Python 3.12.13
lerobot==0.5.2
torch==2.10.0
torchvision==0.25.0
transformers==5.3.0
accelerate==1.13.0
datasets==4.8.4
pandas==2.3.3
pyarrow==23.0.1
av==15.1.0
```

### `lerobot` sanity check

```bash
conda activate lerobot
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

## 7. Environment Variables

### For `RWRL`

Typical variables:

```bash
export ROBOT=http://192.168.12.12:8001
export MPLCONFIGDIR=/tmp/matplotlib
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

For offline training without external network dependency:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
```

### For `lerobot`

Use a local LeRobot dataset cache root:

```bash
export HF_LEROBOT_HOME=/home/robot/VLA-RL/conrft-r1lite/data/lerobot
export PYTHONPATH=/home/robot/VLA-RL/lerobot/src
```

## 8. What Each Environment Is Used For

Use `RWRL` for:

- robot-side replay / teleop / monitor tools
- `examples/train_conrft_octo.py`
- ConRFT offline pretrain / online RL
- adding Octo embeddings into ConRFT pkl

Use `lerobot` for:

- RAW rosbag -> LeRobotDataset export
- manual / VLM SARM annotation
- SARM reward model training
- `compute_rabc_weights.py`
- LeRobotDataset + `sarm_progress.parquet` -> ConRFT pkl conversion

## 9. Required Local `lerobot` Patches

The current local `lerobot` checkout has two modified files:

- `src/lerobot/policies/sarm/processor_sarm.py`
- `src/lerobot/policies/sarm/compute_rabc_weights.py`

### Patch 1: `processor_sarm.py`

This patch is effectively required for the current local setup.

Why:

- with `transformers==5.3.0`, CLIP `get_image_features()` / `get_text_features()`
  may return a model-output object instead of a raw tensor
- upstream code assumed the return value always had `.detach()`
- without this patch, SARM training crashes with:

```text
AttributeError: 'BaseModelOutputWithPooling' object has no attribute 'detach'
```

So this patch is not just convenience; it is needed for this version
combination.

### Patch 2: `compute_rabc_weights.py`

This patch is recommended, but only part of it is strictly necessary.

What it changes:

- uses `matplotlib.use("Agg")` for headless servers
- configures CJK-capable fonts for Chinese labels
- changes `--push-to-hub` to default `False`

Interpretation:

- `Agg` + font setup are very useful on headless machines and avoid Tk / font
  issues during local visualization
- `--push-to-hub` default `False` is safer for local workflows and matches how
  this project is using LeRobot
- model correctness does not depend on this patch, but the local workflow does
  benefit from it

## 10. Should `lerobot` Be Maintained In A Remote Repository?

Yes, I recommend it.

Right now the local `lerobot` clone points directly at upstream:

```text
origin https://github.com/huggingface/lerobot.git
```

but it also contains project-specific local patches. That is fragile.

Recommended maintenance model:

1. Fork `huggingface/lerobot` to your own GitHub account.
2. Change local `origin` to your fork.
3. Keep upstream as `upstream`.
4. Maintain a branch such as:

```text
r1lite-sarm-patches
```

Suggested remote layout:

```bash
cd /home/robot/VLA-RL/lerobot
git remote rename origin upstream
git remote add origin git@github.com:<your-name>/lerobot.git
git checkout -b r1lite-sarm-patches
```

This is the best option if you want another machine to reproduce the exact same
SARM workflow without rediscovering local fixes.

## 11. Recommended Bring-Up Order On A New Machine

1. Install Miniforge.
2. Verify `nvidia-smi`.
3. Clone `conrft-r1lite`.
4. Clone your patched `lerobot` fork.
5. Create `RWRL`.
6. Install editable packages and the pinned `RWRL` dependency set.
7. Create `lerobot`.
8. Install editable `lerobot` and the pinned torch / transformers stack.
9. Apply or pull the required `lerobot` SARM patches.
10. Verify both envs with the sanity-check commands above.

## 12. Minimal Functional Checks

### Check robot-side Python path in `RWRL`

```bash
conda activate RWRL
cd /home/robot/VLA-RL/conrft-r1lite
python - <<'PY'
import sys
sys.path[:0] = [
    '/home/robot/VLA-RL/conrft-r1lite/examples',
    '/home/robot/VLA-RL/conrft-r1lite/serl_robot_infra',
    '/home/robot/VLA-RL/conrft-r1lite/serl_launcher',
]
from experiments.r1lite_dual_mango_box.config import TrainConfig
cfg = TrainConfig()
env = cfg.get_environment(fake_env=True)
print(env.observation_space["state"].shape, env.action_space.shape)
env.close()
PY
```

Expected:

```text
(2, 53) (14,)
```

### Check LeRobot dataset export tool

```bash
conda activate lerobot
cd /home/robot/VLA-RL/conrft-r1lite
python examples/sarm/export_rosbag_to_lerobot_sarm.py --help
python examples/sarm/relabel_rosbag_or_conrft_with_sarm_reward.py --help
```

### Check SARM progress compute path

```bash
conda activate lerobot
cd /home/robot/VLA-RL/lerobot
python -m lerobot.policies.sarm.compute_rabc_weights --help
```

## 13. Notes

- `conda env export --from-history` is not enough to reproduce these envs
  because most important dependencies were installed through `pip`.
- Treat the package versions in this document as the reproducible source of
  truth for now.
- If you upgrade `transformers` in either env, re-check both SARM training and
  Octo-related code paths.
