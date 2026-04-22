#!/usr/bin/env python3

import argparse
import csv
import copy
import pickle as pkl
from pathlib import Path
from typing import Dict, List

import numpy as np

_EXAMPLES_DIR = Path(__file__).resolve().parents[1]
import sys

if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

from data_util import add_mc_returns_to_trajectory  # noqa: E402


def _split_trajectories(transitions: List[Dict]) -> List[List[Dict]]:
    trajectories: List[List[Dict]] = []
    current: List[Dict] = []
    for transition in transitions:
        current.append(transition)
        if bool(transition.get("dones", False)):
            trajectories.append(current)
            current = []
    if current:
        trajectories.append(current)
    return trajectories


def _progress_path(source_lerobot_dataset: str | None, progress_parquet: str | None) -> Path:
    if progress_parquet:
        return Path(progress_parquet).expanduser().resolve()
    if not source_lerobot_dataset:
        raise ValueError("Provide --progress_parquet or --source_lerobot_dataset")
    candidate = Path(source_lerobot_dataset).expanduser().resolve() / "sarm_progress.parquet"
    if not candidate.exists():
        raise FileNotFoundError(
            f"Could not find {candidate}. Run LeRobot `compute_rabc_weights.py` first, "
            "or pass --progress_parquet explicitly."
        )
    return candidate


def _load_progress(path: Path, head_mode: str) -> Dict[int, List[float]]:
    column = f"progress_{head_mode}"

    if path.suffix.lower() == ".csv":
        rows = []
        with path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"Progress CSV {path} is empty")
            required = {"episode_index", "frame_index", column}
            missing = required.difference(reader.fieldnames)
            if missing:
                raise ValueError(f"Progress CSV {path} missing columns: {sorted(missing)}")
            for row in reader:
                rows.append(
                    {
                        "episode_index": int(row["episode_index"]),
                        "frame_index": int(row["frame_index"]),
                        column: float(row[column]),
                    }
                )
    else:
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "Reading LeRobot/SARM parquet progress requires pandas. Run this script in the LeRobot "
                "environment, install pandas/pyarrow, or pass a CSV with episode_index, frame_index, "
                f"and {column} columns."
            ) from exc
        df = pd.read_parquet(path)
        if column not in df.columns:
            raise ValueError(f"Progress file {path} does not contain column {column}. Columns: {list(df.columns)}")
        required = {"episode_index", "frame_index", column}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"Progress file {path} missing columns: {sorted(missing)}")
        rows = [
            {
                "episode_index": int(row.episode_index),
                "frame_index": int(row.frame_index),
                column: float(getattr(row, column)),
            }
            for row in df.sort_values(["episode_index", "frame_index"]).itertuples(index=False)
        ]

    result: Dict[int, List[float]] = {}
    rows.sort(key=lambda item: (item["episode_index"], item["frame_index"]))
    for row in rows:
        result.setdefault(row["episode_index"], []).append(row[column])
    return result


def _relabel_trajectory(
    trajectory: List[Dict],
    progress: List[float],
    gamma: float,
    reward_scale: float,
    reward_bias: float,
    reward_clip_low: float,
    reward_clip_high: float,
    success_threshold: float,
    success_reward: float,
    truncate_after_success: bool,
) -> List[Dict]:
    if len(progress) < len(trajectory) + 1:
        raise ValueError(
            f"Need at least len(trajectory)+1 progress values, got progress={len(progress)} trajectory={len(trajectory)}"
        )

    relabeled: List[Dict] = []
    for idx, transition in enumerate(trajectory):
        current_progress = float(progress[idx])
        next_progress = float(progress[idx + 1])
        dense_reward = float(np.clip(next_progress - current_progress, reward_clip_low, reward_clip_high))
        reward = reward_scale * dense_reward + reward_bias
        succeed = bool(next_progress >= success_threshold)
        done = bool(succeed or idx == len(trajectory) - 1)
        if succeed:
            reward += success_reward

        item = copy.deepcopy(transition)
        item["rewards"] = float(reward)
        item["dones"] = done
        item["masks"] = float(1.0 - done)
        info = dict(item.get("infos", {}))
        info.update(
            {
                "sarm_progress": current_progress,
                "sarm_next_progress": next_progress,
                "sarm_reward_delta": dense_reward,
                "succeed": succeed,
            }
        )
        item["infos"] = info
        relabeled.append(item)
        if truncate_after_success and succeed:
            break

    relabeled = add_mc_returns_to_trajectory(
        relabeled,
        gamma=gamma,
        reward_scale=1.0,
        reward_bias=0.0,
        reward_neg=0.0,
        is_sparse_reward=False,
    )
    return relabeled


def main():
    parser = argparse.ArgumentParser(description="Relabel a ConRFT transition pkl with LeRobot SARM progress rewards.")
    parser.add_argument("--input_pkl", required=True, help="Existing ConRFT transition pkl to relabel.")
    parser.add_argument("--source_lerobot_dataset", default=None, help="Local LeRobotDataset directory containing sarm_progress.parquet.")
    parser.add_argument("--progress_parquet", default=None, help="Explicit path to sarm_progress.parquet, or CSV for lightweight tests.")
    parser.add_argument("--sarm_model", default=None, help="Recorded for metadata only; run LeRobot compute_rabc_weights.py before this script.")
    parser.add_argument("--head_mode", choices=("sparse", "dense"), default="sparse")
    parser.add_argument("--output_pkl", required=True)
    parser.add_argument("--success_threshold", type=float, default=0.95)
    parser.add_argument("--success_reward", type=float, default=10.0)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--reward_scale", type=float, default=1.0)
    parser.add_argument("--reward_bias", type=float, default=0.0)
    parser.add_argument("--reward_clip_low", type=float, default=-1.0)
    parser.add_argument("--reward_clip_high", type=float, default=1.0)
    parser.add_argument("--no_truncate_after_success", action="store_true")
    args = parser.parse_args()

    input_pkl = Path(args.input_pkl).expanduser().resolve()
    with input_pkl.open("rb") as f:
        transitions = pkl.load(f)
    trajectories = _split_trajectories(transitions)
    progress_by_episode = _load_progress(_progress_path(args.source_lerobot_dataset, args.progress_parquet), args.head_mode)

    output: List[Dict] = []
    for episode_index, trajectory in enumerate(trajectories):
        if episode_index not in progress_by_episode:
            raise ValueError(f"Progress parquet does not contain episode_index={episode_index}")
        output.extend(
            _relabel_trajectory(
                trajectory,
                progress_by_episode[episode_index],
                gamma=args.gamma,
                reward_scale=args.reward_scale,
                reward_bias=args.reward_bias,
                reward_clip_low=args.reward_clip_low,
                reward_clip_high=args.reward_clip_high,
                success_threshold=args.success_threshold,
                success_reward=args.success_reward,
                truncate_after_success=not args.no_truncate_after_success,
            )
        )

    output_pkl = Path(args.output_pkl).expanduser().resolve()
    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    with output_pkl.open("wb") as f:
        pkl.dump(output, f)
    print(f"Relabeled {len(trajectories)} trajectory(ies), {len(output)} transition(s)")
    print(f"Wrote SARM-reward ConRFT pkl to {output_pkl}")


if __name__ == "__main__":
    main()
