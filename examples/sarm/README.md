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
  --batch_size=16 \
  --steps=5000 \
  --num_workers=6 \
  --prefetch_factor=4 \
  --persistent_workers=true
```

Compute progress values:

```bash
python src/lerobot/policies/sarm/compute_rabc_weights.py \
  --dataset-repo-id r1lite_dual_mango_box \
  --reward-model-path outputs/train/r1lite_dual_mango_box_sarm \
  --head-mode sparse \
  --num-visualizations 5
```

This should produce `sarm_progress.parquet` next to the local dataset.

## 4. Relabel Existing ConRFT Demo PKL

This step is only needed for ConRFT offline/pretrain. It consumes the progress
parquet and recomputes rewards plus `mc_returns`.

```bash
python examples/sarm/relabel_rosbag_or_conrft_with_sarm_reward.py \
  --input_pkl=/home/robot/VLA-RL/conrft-r1lite/data/transition/r1lite_dual_mango_box/r1lite_dual_mango_box_demos.pkl \
  --source_lerobot_dataset=/home/robot/VLA-RL/conrft-r1lite/data/lerobot/r1lite_dual_mango_box \
  --output_pkl=/home/robot/VLA-RL/conrft-r1lite/data/transition/r1lite_dual_mango_box/r1lite_dual_mango_box_sarm_reward.pkl \
  --success_threshold=0.95 \
  --success_reward=10.0
```

Then train as usual:

```bash
DEMO_PATH=/home/robot/VLA-RL/conrft-r1lite/data/transition/r1lite_dual_mango_box/r1lite_dual_mango_box_sarm_reward.pkl \
bash examples/experiments/r1lite_dual_mango_box/run_learner_conrft_pretrain.sh
```

## 5. Online Reward Wrapper

Start a SARM progress sidecar that exposes:

```text
POST /predict_progress
{
  "image_jpeg_base64": "...",
  "state": [...],
  "task": "...",
  "head_mode": "sparse",
  "image_key": "head"
}
```

Expected response:

```json
{"progress": 0.73}
```

Then enable the wrapper in `examples/experiments/r1lite_dual_mango_box/config.yaml`:

```yaml
reward_model:
  enabled: true
  log_only: true
  endpoint_url: "http://127.0.0.1:8010"
```

Use `log_only: true` first to verify `info["sarm_progress"]` increases through
the whole task including robot reset. Switch to `log_only: false` only after the
progress signal is stable.
