#!/usr/bin/env python3

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from rosbag_sarm_utils import (  # noqa: E402
    DEFAULT_TOPICS,
    EEF_ACTION_NAMES,
    JOINT_ACTION_NAMES,
    STATE_NAMES,
    action_vector,
    build_episode_samples,
    parse_csv,
    resolve_input_dirs,
    state_vector,
)


DEFAULT_TASK_DESC = "左臂抓住白色的框放在右臂的周围，右臂抓住发红的芒果，把它放入框内，然后左右机械臂复位。"


def _default_lerobot_home() -> Path:
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")).expanduser()
    return Path(os.environ.get("HF_LEROBOT_HOME", hf_home / "lerobot")).expanduser()


def _dataset_root(output_repo_id: str, output_dir: str | None, root: str | None) -> Path:
    explicit_root = output_dir or root
    if explicit_root:
        return Path(explicit_root).expanduser().resolve()
    return (_default_lerobot_home() / output_repo_id).resolve()


def _load_lerobot_dataset_class():
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise RuntimeError(
            "LeRobot is required for direct LeRobotDataset export. Install it in a sidecar env, e.g. "
            "`pip install -e '.[sarm]'` from the LeRobot repository."
        ) from exc
    return LeRobotDataset


def _normalize_lerobot_fps(fps: float) -> int:
    rounded = int(round(float(fps)))
    if rounded <= 0:
        raise ValueError(f"--fps must be positive, got {fps}")
    if abs(float(fps) - rounded) > 1e-6:
        raise ValueError(
            f"Video-backed LeRobot export requires an integer fps for PyAV video encoding, got {fps}. "
            "Use an integer value such as --fps=10."
        )
    return rounded


def _image_feature(image_shape, image_dtype: str) -> Dict:
    h, w, c = image_shape
    return {"dtype": image_dtype, "shape": (h, w, c), "names": ["height", "width", "channel"]}


def _features(image_shapes: Dict[str, tuple], state_dim: int, action_names: List[str], image_dtype: str) -> Dict:
    return {
        "observation.images.head": _image_feature(image_shapes["head"], image_dtype),
        "observation.images.left_wrist": _image_feature(image_shapes["left_wrist"], image_dtype),
        "observation.images.right_wrist": _image_feature(image_shapes["right_wrist"], image_dtype),
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": list(STATE_NAMES),
        },
        "action": {
            "dtype": "float32",
            "shape": (len(action_names),),
            "names": action_names,
        },
    }


def _topic_overrides(args) -> Dict[str, str]:
    topics = dict(DEFAULT_TOPICS)
    for key in list(DEFAULT_TOPICS):
        override = getattr(args, f"{key}_topic", None)
        if override:
            topics[key] = override
    return topics


def _add_episode(dataset, samples, fps: float, task_desc: str, action_space: str):
    if len(samples) < 2:
        raise ValueError("Need at least two synchronized samples to export an episode.")
    for idx, sample in enumerate(samples):
        next_sample = samples[idx + 1] if idx + 1 < len(samples) else sample
        dt = max(1e-6, (next_sample["timestamp_ns"] - sample["timestamp_ns"]) / 1e9) if idx + 1 < len(samples) else 1.0 / fps
        frame = {
            "observation.images.head": sample["head"],
            "observation.images.left_wrist": sample["left_wrist"],
            "observation.images.right_wrist": sample["right_wrist"],
            "observation.state": state_vector(sample, next_sample, dt),
            "action": action_vector(sample, next_sample, action_space),
            "task": task_desc,
        }
        dataset.add_frame(frame)
    dataset.save_episode()


def main():
    parser = argparse.ArgumentParser(description="Export R1Lite RAW rosbag episodes to a LeRobotDataset for SARM.")
    parser.add_argument(
        "--input_dirs",
        required=True,
        help=(
            "Comma-separated RAW episode directories or parent directories. If a parent directory is passed, "
            "all child directories matching --raw_dir_glob are exported as episodes."
        ),
    )
    parser.add_argument("--raw_dir_glob", default="*_RAW", help="Glob used when --input_dirs contains parent directories.")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan parent directories for RAW episodes.")
    parser.add_argument("--task_name", default="r1lite_dual_mango_box")
    parser.add_argument("--task_desc", default=DEFAULT_TASK_DESC)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--action_space", choices=("eef", "joint"), default="eef")
    parser.add_argument("--output_repo_id", required=True, help="LeRobot repo_id or local repo id.")
    parser.add_argument("--output_dir", default=None, help="Actual local LeRobotDataset root directory to write.")
    parser.add_argument("--root", default=None, help="Deprecated alias for --output_dir.")
    parser.add_argument("--overwrite", action="store_true", help="Delete the existing output dataset directory before export.")
    parser.add_argument("--no_videos", action="store_true", help="Store images directly instead of MP4 videos.")
    parser.add_argument("--vcodec", default="h264", help="Video codec for LeRobot MP4 export. Default: h264 for browser compatibility.")
    parser.add_argument("--image_writer_threads", type=int, default=4)
    for key, default_topic in DEFAULT_TOPICS.items():
        parser.add_argument(f"--{key}_topic", default=None, help=f"Override topic for {key}. Default: {default_topic}")
    parser.add_argument("--dry_run_manifest", default=None, help="Write a JSON manifest instead of creating a LeRobotDataset.")
    args = parser.parse_args()

    lerobot_fps = _normalize_lerobot_fps(args.fps)
    input_dirs = resolve_input_dirs(parse_csv(args.input_dirs), args.raw_dir_glob, args.recursive)
    topics = _topic_overrides(args)
    print(f"Resolved {len(input_dirs)} RAW episode director{'y' if len(input_dirs) == 1 else 'ies'}")
    for idx, path in enumerate(input_dirs):
        print(f"  [{idx:03d}] {path}")
    episode_samples = [build_episode_samples(path, args.fps, topics) for path in input_dirs]
    action_names = list(EEF_ACTION_NAMES if args.action_space == "eef" else JOINT_ACTION_NAMES)
    image_shapes = {
        "head": tuple(np.asarray(episode_samples[0][0]["head"]).shape),
        "left_wrist": tuple(np.asarray(episode_samples[0][0]["left_wrist"]).shape),
        "right_wrist": tuple(np.asarray(episode_samples[0][0]["right_wrist"]).shape),
    }
    image_dtype = "image" if args.no_videos else "video"
    features = _features(image_shapes, len(STATE_NAMES), action_names, image_dtype)

    if args.dry_run_manifest:
        output_root = _dataset_root(args.output_repo_id, args.output_dir, args.root)
        manifest = {
            "task_name": args.task_name,
            "task_desc": args.task_desc,
            "fps": lerobot_fps,
            "action_space": args.action_space,
            "output_repo_id": args.output_repo_id,
            "output_dir": str(output_root),
            "input_dirs": [str(path) for path in input_dirs],
            "raw_dir_glob": args.raw_dir_glob,
            "recursive": args.recursive,
            "image_storage": image_dtype,
            "vcodec": args.vcodec,
            "topics": topics,
            "num_episodes": len(episode_samples),
            "episode_lengths": [len(samples) for samples in episode_samples],
            "features": features,
        }
        output = Path(args.dry_run_manifest).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote dry-run manifest to {output}")
        return

    LeRobotDataset = _load_lerobot_dataset_class()
    output_root = _dataset_root(args.output_repo_id, args.output_dir, args.root)
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"LeRobot output dataset already exists: {output_root}\n"
                "Re-run with --overwrite to delete and recreate it, or choose a new --output_dir/--output_repo_id."
            )
        print(f"Overwriting existing LeRobot dataset: {output_root}")
        shutil.rmtree(output_root)
    create_kwargs = {
        "repo_id": args.output_repo_id,
        "fps": lerobot_fps,
        "features": features,
        "robot_type": "r1lite_dual",
        "use_videos": not args.no_videos,
        "vcodec": args.vcodec,
        "image_writer_threads": args.image_writer_threads,
    }
    if args.output_dir or args.root:
        create_kwargs["root"] = output_root
    dataset = LeRobotDataset.create(**create_kwargs)
    for samples in episode_samples:
        _add_episode(dataset, samples, lerobot_fps, args.task_desc, args.action_space)
    finalize = getattr(dataset, "finalize", None)
    if callable(finalize):
        finalize()
    print(f"Exported {len(episode_samples)} episode(s) to LeRobotDataset {args.output_repo_id}")


if __name__ == "__main__":
    main()
