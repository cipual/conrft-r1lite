# R1Lite SARM Reward Modeling

This folder keeps the SARM pipeline decoupled from the ConRFT/RL training code.
The data path is:

```text
RAW rosbag -> LeRobotDataset -> LeRobot SARM annotation/training -> sarm_progress.parquet -> ConRFT pkl / online reward
```

The task includes robot reset as part of completion:

```text
左臂抓住白色的框放在右臂的周围，右臂抓住发红的芒果，把它放入框内，然后左右机械臂复位。
```

## 0. Environment

Run LeRobot export, annotation, SARM training, and progress computation in the
`lerobot` conda environment:

```bash
source /home/robot/Applications/miniforge3/etc/profile.d/conda.sh
conda activate lerobot
cd /home/robot/VLA-RL/conrft-r1lite
export HF_LEROBOT_HOME=/home/robot/VLA-RL/conrft-r1lite/data/lerobot
```

The manual annotation UI writes LeRobot episode parquet metadata directly, so
the environment must include LeRobot dataset dependencies such as `pandas`,
`pyarrow`, and `datasets`. If they are missing:

```bash
python -m pip install "lerobot[dataset]" pandas pyarrow datasets
```

## 1. Export RAW Rosbags To LeRobotDataset

Run this from the `lerobot` environment.

By default, the exporter creates a video-backed LeRobotDataset: camera features
are stored as `dtype=video` and MP4 files are written under `videos/...`. This is
required by both LeRobot VLM SARM annotation and the manual annotation UI.
Do not pass `--no_videos` for SARM annotation datasets. If an older dataset was
exported as image-backed data, re-run the export command below with
`--overwrite`.

The exporter defaults to `--vcodec=h264` because it is the most reliable choice
for browser-based manual annotation. If a dataset was exported with AV1 and the
browser video stays stuck at `0s`, re-export it with `--overwrite --vcodec=h264`.

```bash
cd /home/robot/VLA-RL/conrft-r1lite

python examples/sarm/export_rosbag_to_lerobot_sarm.py \
  --input_dirs=/home/robot/VLA-RL/conrft-r1lite/data/RAW/r1lite_dual_mango_box \
  --task_name=r1lite_dual_mango_box \
  --task_desc="左臂抓住白色的框放在右臂的周围，右臂抓住发红的芒果，把它放入框内，然后左右机械臂复位。" \
  --fps=10 \
  --action_space=eef \
  --output_repo_id=r1lite_dual_mango_box \
  --output_dir=/home/robot/VLA-RL/conrft-r1lite/data/lerobot/r1lite_dual_mango_box \
  --overwrite
```
```bash
export HF_LEROBOT_HOME=/home/robot/VLA-RL/conrft-r1lite/data/lerobot
```

You can also pass a parent directory and export every `*_RAW` episode inside it:

```bash
python examples/sarm/export_rosbag_to_lerobot_sarm.py \
  --input_dirs=/home/robot/VLA-RL/conrft-r1lite/data/RAW/r1lite_dual_mango_box \
  --task_name=r1lite_dual_mango_box \
  --task_desc="左臂抓住白色的框放在右臂的周围，右臂抓住发红的芒果，把它放入框内，然后左右机械臂复位。" \
  --fps=10 \
  --action_space=eef \
  --output_repo_id=r1lite_dual_mango_box \
  --output_dir=/home/robot/VLA-RL/conrft-r1lite/data/lerobot/r1lite_dual_mango_box \
  --overwrite
```

If the RAW directories are nested deeper, add `--recursive`. The default scan
pattern is `--raw_dir_glob='*_RAW'`.

`--output_repo_id` is the LeRobot dataset id. With
`HF_LEROBOT_HOME=/home/robot/VLA-RL/conrft-r1lite/data/lerobot` and
`--output_repo_id=r1lite_dual_mango_box`, LeRobot-compatible tools will look under
`data/lerobot/r1lite_dual_mango_box`. `--output_dir` controls the actual local
directory written on disk and should match that path for this local layout. If
the target directory already exists, pass `--overwrite` to delete and recreate
it, or choose a new `--output_dir` / `--output_repo_id`.

`--no_videos` is only for dependency-light debugging where annotation is not
needed; datasets exported with `--no_videos` cannot be used by the VLM or manual
SARM annotation tools.

For a dependency-light sanity check that does not require LeRobot:

```bash
python examples/sarm/export_rosbag_to_lerobot_sarm.py \
  --input_dirs=/home/robot/VLA-RL/conrft-r1lite/data/RAW/r1lite_reach_target/RB251106041_20260409152555451_RAW \
  --fps=10 \
  --action_space=eef \
  --output_repo_id=r1lite_reach_target \
  --output_dir=/home/robot/VLA-RL/conrft-r1lite/data/lerobot/r1lite_reach_target_dryrun \
  --dry_run_manifest=/tmp/r1lite_sarm_export_manifest.json
```

## 2. VLM Dual Annotation

Run from the LeRobot repository:

```bash
cd /home/robot/VLA-RL/lerobot

python src/lerobot/data_processing/sarm_annotations/subtask_annotation.py \
  --repo-id r1lite_dual_mango_box \
  --video-key observation.images.head \
  --sparse-subtasks "left arm grasps the white box,left arm positions the white box around the right arm,right arm grasps the red mango,right arm places the red mango into the white box,both robot arms return to the reset pose" \
  --dense-subtasks "left arm approaches the white box,left gripper closes on the white box,left arm moves the white box,the white box is positioned around the right arm,right arm approaches the red mango,right gripper closes on the red mango,right arm moves the mango above the white box,the mango is released inside the white box,both arms move away from the objects,both arms reach the reset pose" \
  --num-workers=1 \
  --num-visualizations=5
```

In the current LeRobot SARM script, annotation mode is inferred from the
arguments: passing both `--sparse-subtasks` and `--dense-subtasks` creates dual
annotations. Do not pass `--annotation-mode` to `subtask_annotation.py`; reserve
`--policy.annotation_mode=dual` for SARM training.

### Optional Manual Annotation UI

If you want to annotate or correct episodes by hand, use the local LeRobot
preview UI. It loads an already-exported LeRobotDataset, not RAW rosbag
directories, so the manual and VLM annotation paths share the same data format.
Saving writes:

```text
<dataset_root>/meta/sarm_manual_annotations.json
<dataset_root>/meta/temporal_proportions_sparse.json
<dataset_root>/meta/temporal_proportions_dense.json
<dataset_root>/meta/episodes/**/*.parquet sparse_* / dense_* columns
```

```bash
cd /home/robot/VLA-RL/conrft-r1lite

python examples/sarm/manual_annotate_sarm.py \
  --dataset_root=/home/robot/VLA-RL/conrft-r1lite/data/lerobot/r1lite_dual_mango_box \
  --video_key=observation.images.head \
  --port=8020
```

Open the printed URL, preview each episode video, mark start/end frames for
each sparse and dense subtask, and press `Save to LeRobot`.

To continue editing an existing annotation file:

```bash
python examples/sarm/manual_annotate_sarm.py \
  --dataset_root=/home/robot/VLA-RL/conrft-r1lite/data/lerobot/r1lite_dual_mango_box \
  --annotations_file=/home/robot/VLA-RL/conrft-r1lite/data/lerobot/r1lite_dual_mango_box/meta/sarm_manual_annotations.json \
  --video_key=observation.images.head \
  --port=8020
```

For sidecar preparation/checking without starting the browser server:

```bash
python examples/sarm/manual_annotate_sarm.py \
  --dataset_root=/home/robot/VLA-RL/conrft-r1lite/data/lerobot/r1lite_dual_mango_box \
  --video_key=observation.images.head \
  --prepare_only
```

By default, the first save in a server session creates `.bak-<timestamp>`
backups of modified LeRobot parquet/proportion files. Pass `--no_backup` only
if you intentionally do not want that safety net.

## 3. Train SARM

```bash
export HF_LEROBOT_HOME=/home/robot/VLA-RL/conrft-r1lite/data/lerobot

lerobot-train \
  --dataset.repo_id=r1lite_dual_mango_box \
  --policy.type=sarm \
  --policy.annotation_mode=dense_only \
  --policy.image_key=observation.images.head \
  --policy.state_key=observation.state \
  --policy.frame_gap=10 \
  --policy.push_to_hub=false \
  --output_dir=/home/robot/VLA-RL/conrft-r1lite/examples/sarm/outputs/train/r1lite_dual_mango_box_sarm_$(date +%Y%m%d_%H%M%S) \
  --batch_size=4 \
  --steps=5000 \
  --num_workers=6 \
  --prefetch_factor=4 \
  --persistent_workers=true
```

Compute progress values:

```bash
cd /home/robot/VLA-RL/lerobot
export HF_LEROBOT_HOME=/home/robot/VLA-RL/conrft-r1lite/data/lerobot
export PYTHONPATH=/home/robot/VLA-RL/lerobot/src

python -m lerobot.policies.sarm.compute_rabc_weights \
  --dataset-repo-id r1lite_dual_mango_box \
  --reward-model-path /home/robot/VLA-RL/conrft-r1lite/examples/sarm/outputs/train/r1lite_dual_mango_box_sarm_20260422_122234/checkpoints/005000/pretrained_model \
  --head-mode dense \
  --output-path /home/robot/VLA-RL/conrft-r1lite/data/lerobot/r1lite_dual_mango_box/sarm_progress.parquet \
  --output-dir /home/robot/VLA-RL/conrft-r1lite/examples/sarm/outputs/rabc_viz \
  --num-visualizations 5 \
  --stride 1
```

Use `--head-mode dense` when the model was trained with
`--policy.annotation_mode=dense_only`. Use `--head-mode both` only when you
want to compare sparse and dense progress curves.

By default, `compute_rabc_weights.py` now writes local files only and does not
upload to Hugging Face Hub. Add `--push-to-hub` explicitly if you intentionally
want to upload `sarm_progress.parquet` to the dataset repo.

This command should produce `sarm_progress.parquet` next to the local dataset.
If you only need the parquet and do not want visualization images, set
`--num-visualizations 0`. If the parquet already exists and you only want to
regenerate visualizations, add `--visualize-only`.

## 4. Export SARM-Reward ConRFT PKL

For long-horizon tasks whose reward is defined by SARM, the recommended path is:

```text
LeRobotDataset + sarm_progress.parquet -> ConRFT transition pkl
```

This avoids needing a hand-written sparse reward before the reward model exists.
Run this step in the `lerobot` environment because it reads LeRobot parquet/video
files.

Fast state/action/reward smoke test without images:

```bash
source /home/robot/Applications/miniforge3/etc/profile.d/conda.sh
conda activate lerobot
cd /home/robot/VLA-RL/conrft-r1lite

python examples/sarm/relabel_rosbag_or_conrft_with_sarm_reward.py \
  --source_lerobot_dataset=/home/robot/VLA-RL/conrft-r1lite/data/lerobot/r1lite_dual_mango_box \
  --output_pkl=/tmp/r1lite_dual_mango_box_sarm_reward_smoke.pkl \
  --head_mode=dense \
  --success_threshold=0.95 \
  --success_reward=10.0 \
  --max_episodes=1 \
  --max_transitions=20
```

Full visual ConRFT pkl export, before Octo embeddings:

```bash
source /home/robot/Applications/miniforge3/etc/profile.d/conda.sh
conda activate lerobot
cd /home/robot/VLA-RL/conrft-r1lite

python examples/sarm/relabel_rosbag_or_conrft_with_sarm_reward.py \
  --source_lerobot_dataset=/home/robot/VLA-RL/conrft-r1lite/data/lerobot/r1lite_dual_mango_box \
  --output_pkl=/home/robot/VLA-RL/conrft-r1lite/data/transition/r1lite_dual_mango_box/r1lite_dual_mango_box_sarm_reward_no_octo.pkl \
  --head_mode=dense \
  --success_threshold=0.95 \
  --success_reward=10.0 \
  --include_images \
  --image_max_width=320
```

Then add real Octo embeddings in the `RWRL` environment:

```bash
source /home/robot/Applications/miniforge3/etc/profile.d/conda.sh
conda activate RWRL
cd /home/robot/VLA-RL/conrft-r1lite

python examples/sarm/add_octo_embeddings_to_conrft_pkl.py \
  --exp_name=r1lite_dual_mango_box \
  --input_pkl=/home/robot/VLA-RL/conrft-r1lite/data/transition/r1lite_dual_mango_box/r1lite_dual_mango_box_sarm_reward_no_octo.pkl \
  --output_pkl=/home/robot/VLA-RL/conrft-r1lite/data/transition/r1lite_dual_mango_box/r1lite_dual_mango_box_sarm_reward.pkl
```

### Optional: Preview Camera Videos From Any PKL

Use this after generating a ConRFT/debug pkl to quickly verify that camera frames
were written correctly. The script is independent from the RL env and supports
both trajectory-list pkl files and flat transition-list pkl files. For flat
transition pkl files, `--trajectory_index` means episode index after grouping by
`infos["episode_index"]`; if that metadata is missing, episodes are split by
`dones=True`.

First list image keys:

```bash
source /home/robot/Applications/miniforge3/etc/profile.d/conda.sh
conda activate RWRL
cd /home/robot/VLA-RL/conrft-r1lite

python examples/visualize_pkl_cameras.py \
  --input_file=/home/robot/VLA-RL/conrft-r1lite/data/transition/r1lite_dual_mango_box/r1lite_dual_mango_box_sarm_reward.pkl \
  --trajectory_index=0 \
  --list_keys
```

Export a single camera video:

```bash
python examples/visualize_pkl_cameras.py \
  --input_file=/home/robot/VLA-RL/conrft-r1lite/data/transition/r1lite_dual_mango_box/r1lite_dual_mango_box_sarm_reward.pkl \
  --trajectory_index=0 \
  --fps=10 \
  --image_keys=head
```

Export a three-camera side-by-side grid:

```bash
python examples/visualize_pkl_cameras.py \
  --input_file=/home/robot/VLA-RL/conrft-r1lite/data/transition/r1lite_dual_mango_box/r1lite_dual_mango_box_sarm_reward.pkl \
  --trajectory_index=0 \
  --fps=10 \
  --image_keys=head,left_wrist,right_wrist \
  --no_separate
```

Outputs are written under:

```text
data/transition/r1lite_dual_mango_box/pkl_video_preview/
```

Expected log for the SARM pkl should look like:

```text
layout=flat_transitions trajectories=23 trajectory_index=0 section=observations steps=...
```

If it says `steps=2`, you are either using an old version of the script or the
script is reading only one transition's two-frame observation stack instead of a
full episode.

Notes:

- `--head_mode=dense` matches the current `dense_only` SARM training setup.
- `--include_images` reads the LeRobot MP4 files and writes image histories into the pkl. This can create a large file; use `--image_max_width` to control size.
- `--image_max_width=320` means the head camera becomes `180x320`, and wrist cameras become `180x320`. Set `--image_max_width<=0` to keep the original LeRobot video resolution: head `720x1280`, wrists `360x640`.
- Current LeRobot videos may be AV1. The exporter falls back to PyAV software decoding if OpenCV cannot decode AV1.
- Do not use placeholder embeddings for real training. `--embedding_mode=zeros --allow_zero_embeddings` exists only as a smoke-test escape hatch; it satisfies current buffer fields but removes Octo conditioning information. The final training pkl should be the output of `add_octo_embeddings_to_conrft_pkl.py`.

Legacy relabel mode is still available when you already have a ConRFT pkl and
only want to rewrite rewards:

```bash
cd /home/robot/VLA-RL/conrft-r1lite
conda activate lerobot

python examples/sarm/relabel_rosbag_or_conrft_with_sarm_reward.py \
  --input_pkl=/home/robot/VLA-RL/conrft-r1lite/data/transition/r1lite_dual_mango_box/r1lite_dual_mango_box_demos.pkl \
  --source_lerobot_dataset=/home/robot/VLA-RL/conrft-r1lite/data/lerobot/r1lite_dual_mango_box \
  --output_pkl=/home/robot/VLA-RL/conrft-r1lite/data/transition/r1lite_dual_mango_box/r1lite_dual_mango_box_sarm_reward.pkl \
  --head_mode=dense \
  --success_threshold=0.95 \
  --success_reward=10.0
```

Then train as usual:

```bash
DEMO_PATH=/home/robot/VLA-RL/conrft-r1lite/data/transition/r1lite_dual_mango_box/r1lite_dual_mango_box_sarm_reward.pkl \
bash examples/experiments/r1lite_dual_mango_box/run_learner_conrft_pretrain.sh
```

## 5. Online SARM Reward

Offline pretraining consumes the SARM reward already stored in the ConRFT pkl.
Online RL is different: the env needs a live progress estimate for every real
robot step. The online chain is:

```text
actor obs -> SARM progress sidecar -> progress(next_obs) - progress(prev_obs) -> RL reward
```

The current env wrapper only talks to an HTTP endpoint. It does not load the
SARM checkpoint inside the RWRL process. Keep SARM inference in the `lerobot`
environment and keep robot/RL training in the `RWRL` environment.

### 5.1 Start the SARM progress sidecar

Run this in a separate terminal in the `lerobot` environment:

```bash
source /home/robot/Applications/miniforge3/etc/profile.d/conda.sh
conda activate lerobot
cd /home/robot/VLA-RL/conrft-r1lite
export PYTHONPATH=/home/robot/VLA-RL/lerobot/src:$PYTHONPATH

python examples/sarm/sarm_progress_sidecar.py \
  --reward_model_path=/home/robot/VLA-RL/conrft-r1lite/examples/sarm/outputs/train/r1lite_dual_mango_box_sarm_20260422_122234/checkpoints/005000/pretrained_model \
  --host=127.0.0.1 \
  --port=8010 \
  --device=cuda \
  --default_head_mode=dense
```

Check that the server is alive:

```bash
curl -s http://127.0.0.1:8010/health
```

Expected response:

```json
{"status": "ok"}
```

The sidecar exposes:

```text
POST /predict_progress
{
  "image_key": "head",
  "head_mode": "dense",
  "task": "...",
  "image_jpeg_base64": "...",
  "state": [...]
}
```

Expected response:

```json
{
  "progress": 0.73,
  "head_mode": "dense",
  "stage_index": 6,
  "stage_confidence": 0.82
}
```

Implementation note: online inference only has the current observation, while
offline SARM progress computation used a temporal video window. The sidecar
fills the SARM context window by repeating the current frame/state. Treat this
as an online approximation and validate it in `log_only` mode before using it
as the actual RL reward.

### 5.2 Enable wrapper in the dual mango-box config

Edit `examples/experiments/r1lite_dual_mango_box/config.yaml`:

```yaml
reward_model:
  enabled: true
  log_only: true
  endpoint_url: "http://127.0.0.1:8010"
  checkpoint_path: "/home/robot/VLA-RL/conrft-r1lite/examples/sarm/outputs/train/r1lite_dual_mango_box_sarm_20260422_122234/checkpoints/005000/pretrained_model"
  head_mode: "dense"
  image_key: "head"
  success_threshold: 0.95
  success_reward: 10.0
  reward_scale: 1.0
  reward_bias: 0.0
  reward_clip_low: -1.0
  reward_clip_high: 1.0
  timeout: 2.0
```

`checkpoint_path` is documentation/bookkeeping for the experiment config. The
current wrapper uses `endpoint_url`; the actual checkpoint is loaded by the
sidecar command above.

### 5.3 Run online training in log-only mode first

Start the robot body service as usual, then launch learner / actor from the
`RWRL` environment:

```bash
source /home/robot/Applications/miniforge3/etc/profile.d/conda.sh
conda activate RWRL
cd /home/robot/VLA-RL/conrft-r1lite/examples/experiments/r1lite_dual_mango_box

export DEMO_PATH=/home/robot/VLA-RL/conrft-r1lite/data/transition/r1lite_dual_mango_box/r1lite_dual_mango_box_sarm_reward.pkl
export CHECKPOINT_PATH=/home/robot/VLA-RL/conrft-r1lite/examples/experiments/r1lite_dual_mango_box/conrft_sarm

bash run_learner_conrft.sh
```

In another `RWRL` terminal:

```bash
cd /home/robot/VLA-RL/conrft-r1lite/examples/experiments/r1lite_dual_mango_box
export CHECKPOINT_PATH=/home/robot/VLA-RL/conrft-r1lite/examples/experiments/r1lite_dual_mango_box/conrft_sarm

bash run_actor_conrft.sh
```

With `log_only: true`, the wrapper records SARM diagnostics in `info` but does
not replace the env reward:

```text
info["sarm_progress"]
info["sarm_prev_progress"]
info["sarm_reward_delta"]
info["sarm_reward"]
info["sarm_succeed"]
info["sarm_log_only"] == True
```

Use this mode for short real-robot rollouts and verify that
`sarm_progress` generally increases over the full task, including the reset
phase. If it is flat, noisy, or jumps to 1.0 too early, do not enable reward
override yet.

### 5.4 Switch from logging to actual SARM reward

After the progress signal looks reasonable, change only:

```yaml
reward_model:
  enabled: true
  log_only: false
```

Now the wrapper returns SARM reward to RL:

```text
delta = clip(progress_next - progress_prev, reward_clip_low, reward_clip_high)
reward = reward_scale * delta + reward_bias
if progress_next >= success_threshold:
    reward += success_reward
    done = True
    info["succeed"] = True
```

For the current dense-only model, keep:

```yaml
reward_model:
  head_mode: "dense"
```

Use `head_mode: "sparse"` only if you trained or want to evaluate the sparse
head.
