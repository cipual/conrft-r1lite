#!/usr/bin/env python3

import argparse
import csv
import copy
import pickle as pkl
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np


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


def _calc_return_to_go(rewards: Iterable[float], terminals: Iterable[bool], gamma: float) -> np.ndarray:
    rewards = list(rewards)
    terminals = list(terminals)
    returns = [0.0] * len(rewards)
    prev_return = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        returns[i] = float(rewards[i]) + gamma * prev_return * (1.0 - float(terminals[i]))
        prev_return = returns[i]
    return np.asarray(returns, dtype=np.float32)


def _add_mc_returns_to_trajectory(trajectory: List[Dict], gamma: float) -> List[Dict]:
    mc_returns = _calc_return_to_go(
        (transition["rewards"] for transition in trajectory),
        (transition["dones"] for transition in trajectory),
        gamma=gamma,
    )
    for idx, transition in enumerate(trajectory):
        transition["mc_returns"] = mc_returns[idx]
    return trajectory


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


def _find_lerobot_parquets(dataset_root: Path) -> List[Path]:
    data_dir = dataset_root / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"LeRobot data directory does not exist: {data_dir}")
    files = sorted(data_dir.glob("**/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No LeRobot data parquet files found under {data_dir}")
    return files


def _load_lerobot_frame_table(dataset_root: Path):
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "Direct LeRobotDataset -> ConRFT pkl export requires pandas/pyarrow. "
            "Run this script in the `lerobot` conda environment."
        ) from exc
    frames = [pd.read_parquet(path) for path in _find_lerobot_parquets(dataset_root)]
    df = pd.concat(frames, ignore_index=True)
    required = {"observation.state", "action", "episode_index", "frame_index"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"LeRobot dataset is missing required columns: {sorted(missing)}")
    if "index" not in df.columns:
        df["index"] = np.arange(len(df), dtype=np.int64)
    return df.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)


def _parse_csv_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_image_key_map(value: str) -> Dict[str, str]:
    mapping = {}
    for item in _parse_csv_list(value):
        if "=" not in item:
            raise ValueError(f"Expected image mapping item as output_key=lerobot_column, got {item!r}")
        key, column = item.split("=", 1)
        mapping[key.strip()] = column.strip()
    if not mapping:
        raise ValueError("--image_key_map cannot be empty when --include_images is used")
    return mapping


class _VideoFrameReader:
    def __init__(self, dataset_root: Path, image_columns: Iterable[str], image_max_width: int):
        self.dataset_root = dataset_root
        self.image_max_width = int(image_max_width)
        self.video_paths = {column: self._video_path(column) for column in image_columns}
        self.backends = {column: self._open_backend(column) for column in image_columns}
        self.cache: Dict[Tuple[str, int], np.ndarray] = {}

    def _video_path(self, column: str) -> Path:
        path = self.dataset_root / "videos" / column / "chunk-000" / "file-000.mp4"
        if not path.exists():
            matches = sorted((self.dataset_root / "videos" / column).glob("**/*.mp4"))
            if not matches:
                raise FileNotFoundError(f"No MP4 video found for {column} under {self.dataset_root / 'videos'}")
            path = matches[0]
        return path

    def _open_backend(self, column: str) -> Dict:
        path = self.video_paths[column]
        cap = cv2.VideoCapture(str(path))
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                return {"kind": "cv2", "capture": cap}
        cap.release()

        try:
            import av
        except ImportError as exc:
            raise RuntimeError(
                f"OpenCV could not decode {path}, and PyAV is not installed. "
                "Run this image export in the `lerobot` environment or re-export the dataset as H.264."
            ) from exc
        container = av.open(str(path))
        stream = container.streams.video[0]
        return {
            "kind": "av",
            "container": container,
            "stream": stream,
            "frames": container.decode(stream),
            "current_index": -1,
        }

    def _read_with_av(self, column: str, frame_index: int) -> np.ndarray:
        backend = self.backends[column]
        if frame_index <= backend["current_index"]:
            backend["container"].close()
            import av

            container = av.open(str(self.video_paths[column]))
            stream = container.streams.video[0]
            backend.update(
                {
                    "container": container,
                    "stream": stream,
                    "frames": container.decode(stream),
                    "current_index": -1,
                }
            )
        for frame in backend["frames"]:
            backend["current_index"] += 1
            if backend["current_index"] == frame_index:
                return frame.to_ndarray(format="rgb24").astype(np.uint8)
        raise RuntimeError(f"Failed to decode frame {frame_index} from LeRobot video column {column}")

    def read(self, column: str, frame_index: int) -> np.ndarray:
        frame_index = int(frame_index)
        cache_key = (column, frame_index)
        if cache_key in self.cache:
            return self.cache[cache_key]
        backend = self.backends[column]
        if backend["kind"] == "cv2":
            cap = backend["capture"]
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame_bgr = cap.read()
            if not ok:
                raise RuntimeError(f"Failed to read frame {frame_index} from LeRobot video column {column}")
            frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        else:
            frame = self._read_with_av(column, frame_index)
        if self.image_max_width > 0 and frame.shape[1] > self.image_max_width:
            scale = self.image_max_width / float(frame.shape[1])
            new_hw = (max(1, int(round(frame.shape[0] * scale))), self.image_max_width)
            frame = cv2.resize(frame, (new_hw[1], new_hw[0]), interpolation=cv2.INTER_AREA)
        frame = frame.astype(np.uint8)
        self.cache[cache_key] = frame
        if len(self.cache) > 256:
            for key in list(self.cache)[:64]:
                self.cache.pop(key, None)
        return frame

    def close(self):
        for backend in self.backends.values():
            if backend["kind"] == "cv2":
                backend["capture"].release()
            else:
                backend["container"].close()


def _episode_global_bounds(episode_df) -> Tuple[int, int]:
    return int(episode_df["index"].min()), int(episode_df["index"].max())


def _history_global_indices(global_index: int, episode_start_index: int, obs_horizon: int) -> List[int]:
    indices = [global_index - offset for offset in range(obs_horizon - 1, -1, -1)]
    return [max(episode_start_index, int(index)) for index in indices]


def _state_history(frame_table, global_index: int, episode_start_index: int, obs_horizon: int) -> np.ndarray:
    indices = _history_global_indices(global_index, episode_start_index, obs_horizon)
    rows = frame_table.set_index("index", drop=False)
    states = []
    for index in indices:
        if index in rows.index:
            states.append(np.asarray(rows.loc[index]["observation.state"], dtype=np.float32))
        else:
            states.append(np.asarray(rows.loc[episode_start_index]["observation.state"], dtype=np.float32))
    return np.stack(states, axis=0).astype(np.float32)


def _image_history(
    reader: _VideoFrameReader,
    image_key_map: Dict[str, str],
    global_index: int,
    episode_start_index: int,
    obs_horizon: int,
) -> Dict[str, np.ndarray]:
    indices = _history_global_indices(global_index, episode_start_index, obs_horizon)
    return {
        output_key: np.stack([reader.read(column, index) for index in indices], axis=0)
        for output_key, column in image_key_map.items()
    }


def _build_observation(
    row,
    episode_df,
    image_reader: Optional[_VideoFrameReader],
    image_key_map: Dict[str, str],
    episode_start_index: int,
    obs_horizon: int,
) -> Dict:
    obs = {
        "state": _state_history(
            episode_df,
            global_index=int(row["index"]),
            episode_start_index=episode_start_index,
            obs_horizon=obs_horizon,
        )
    }
    if image_reader is not None:
        obs.update(
            _image_history(
                image_reader,
                image_key_map,
                global_index=int(row["index"]),
                episode_start_index=episode_start_index,
                obs_horizon=obs_horizon,
            )
        )
    return obs


def _transition_reward(
    current_progress: float,
    next_progress: float,
    reward_scale: float,
    reward_bias: float,
    reward_clip_low: float,
    reward_clip_high: float,
    success_threshold: float,
    success_reward: float,
    is_last_transition: bool,
) -> Tuple[float, bool, bool, float]:
    dense_reward = float(np.clip(next_progress - current_progress, reward_clip_low, reward_clip_high))
    reward = reward_scale * dense_reward + reward_bias
    succeed = bool(next_progress >= success_threshold)
    done = bool(succeed or is_last_transition)
    if succeed:
        reward += success_reward
    return float(reward), done, succeed, dense_reward


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
        reward, done, succeed, dense_reward = _transition_reward(
            current_progress=current_progress,
            next_progress=next_progress,
            reward_scale=reward_scale,
            reward_bias=reward_bias,
            reward_clip_low=reward_clip_low,
            reward_clip_high=reward_clip_high,
            success_threshold=success_threshold,
            success_reward=success_reward,
            is_last_transition=idx == len(trajectory) - 1,
        )

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

    relabeled = _add_mc_returns_to_trajectory(relabeled, gamma=gamma)
    return relabeled


def _build_conrft_from_lerobot(
    dataset_root: Path,
    progress_by_episode: Dict[int, List[float]],
    head_mode: str,
    gamma: float,
    reward_scale: float,
    reward_bias: float,
    reward_clip_low: float,
    reward_clip_high: float,
    success_threshold: float,
    success_reward: float,
    truncate_after_success: bool,
    obs_horizon: int,
    include_images: bool,
    image_key_map: Dict[str, str],
    image_max_width: int,
    embedding_mode: str,
    max_episodes: Optional[int],
    max_transitions: Optional[int],
) -> List[Dict]:
    frame_table = _load_lerobot_frame_table(dataset_root)
    image_reader = None
    if include_images:
        image_reader = _VideoFrameReader(dataset_root, image_key_map.values(), image_max_width=image_max_width)

    output: List[Dict] = []
    episode_count = 0
    try:
        for episode_index, episode_df in frame_table.groupby("episode_index", sort=True):
            episode_index = int(episode_index)
            if max_episodes is not None and episode_count >= max_episodes:
                break
            episode_df = episode_df.sort_values("frame_index").reset_index(drop=True)
            if len(episode_df) < 2:
                continue
            if episode_index not in progress_by_episode:
                raise ValueError(f"Progress parquet does not contain episode_index={episode_index}")
            progress = progress_by_episode[episode_index]
            if len(progress) < len(episode_df):
                raise ValueError(
                    f"Need at least one progress value per frame for episode {episode_index}, "
                    f"got progress={len(progress)} frames={len(episode_df)}"
                )

            episode_start_index, _ = _episode_global_bounds(episode_df)
            trajectory: List[Dict] = []
            for idx in range(len(episode_df) - 1):
                current = episode_df.iloc[idx]
                nxt = episode_df.iloc[idx + 1]
                current_progress = float(progress[idx])
                next_progress = float(progress[idx + 1])
                reward, done, succeed, dense_reward = _transition_reward(
                    current_progress=current_progress,
                    next_progress=next_progress,
                    reward_scale=reward_scale,
                    reward_bias=reward_bias,
                    reward_clip_low=reward_clip_low,
                    reward_clip_high=reward_clip_high,
                    success_threshold=success_threshold,
                    success_reward=success_reward,
                    is_last_transition=idx == len(episode_df) - 2,
                )

                transition = {
                    "observations": _build_observation(
                        current,
                        episode_df,
                        image_reader=image_reader,
                        image_key_map=image_key_map,
                        episode_start_index=episode_start_index,
                        obs_horizon=obs_horizon,
                    ),
                    "actions": np.asarray(current["action"], dtype=np.float32),
                    "next_observations": _build_observation(
                        nxt,
                        episode_df,
                        image_reader=image_reader,
                        image_key_map=image_key_map,
                        episode_start_index=episode_start_index,
                        obs_horizon=obs_horizon,
                    ),
                    "rewards": float(reward),
                    "masks": float(1.0 - done),
                    "dones": bool(done),
                    "infos": {
                        "succeed": bool(succeed),
                        "sarm_progress": current_progress,
                        "sarm_next_progress": next_progress,
                        "sarm_reward_delta": dense_reward,
                        "grasp_penalty": 0.0,
                        "sarm_head_mode": head_mode,
                        "episode_index": episode_index,
                        "frame_index": int(current["frame_index"]),
                        "next_frame_index": int(nxt["frame_index"]),
                        "conversion_source": "lerobot_sarm_progress",
                    },
                    "grasp_penalty": np.float32(0.0),
                }
                if embedding_mode == "zeros":
                    transition["embeddings"] = np.zeros((384,), dtype=np.float32)
                    transition["next_embeddings"] = np.zeros((384,), dtype=np.float32)
                trajectory.append(transition)
                if truncate_after_success and succeed:
                    break
                if max_transitions is not None and len(output) + len(trajectory) >= max_transitions:
                    break

            output.extend(_add_mc_returns_to_trajectory(trajectory, gamma=gamma))
            episode_count += 1
            if max_transitions is not None and len(output) >= max_transitions:
                output = output[:max_transitions]
                break
    finally:
        if image_reader is not None:
            image_reader.close()

    return output


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create SARM-reward ConRFT transition pkl files. Either relabel an existing ConRFT pkl "
            "with SARM progress, or directly export a LeRobotDataset + sarm_progress.parquet to ConRFT pkl."
        )
    )
    parser.add_argument("--input_pkl", default=None, help="Existing ConRFT transition pkl to relabel. If omitted, export directly from --source_lerobot_dataset.")
    parser.add_argument("--source_lerobot_dataset", default=None, help="Local LeRobotDataset directory containing sarm_progress.parquet.")
    parser.add_argument("--progress_parquet", default=None, help="Explicit path to sarm_progress.parquet, or CSV for lightweight tests.")
    parser.add_argument("--sarm_model", default=None, help="Recorded for metadata only; run LeRobot compute_rabc_weights.py before this script.")
    parser.add_argument("--head_mode", choices=("sparse", "dense"), default="dense")
    parser.add_argument("--output_pkl", required=True)
    parser.add_argument("--success_threshold", type=float, default=0.95)
    parser.add_argument("--success_reward", type=float, default=10.0)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--reward_scale", type=float, default=1.0)
    parser.add_argument("--reward_bias", type=float, default=0.0)
    parser.add_argument("--reward_clip_low", type=float, default=-1.0)
    parser.add_argument("--reward_clip_high", type=float, default=1.0)
    parser.add_argument("--no_truncate_after_success", action="store_true")
    parser.add_argument("--obs_horizon", type=int, default=2, help="Image history length for direct LeRobot export.")
    parser.add_argument("--include_images", action="store_true", help="Direct LeRobot export: include image histories in the output pkl.")
    parser.add_argument(
        "--image_key_map",
        default=(
            "head=observation.images.head,"
            "left_wrist=observation.images.left_wrist,"
            "right_wrist=observation.images.right_wrist"
        ),
        help="Direct LeRobot export image mapping: output_key=lerobot_video_column,...",
    )
    parser.add_argument(
        "--image_max_width",
        type=int,
        default=320,
        help=(
            "Direct LeRobot export: resize images to this max width before pickling. "
            "Default 320. Set <=0 to keep original LeRobot resolution."
        ),
    )
    parser.add_argument(
        "--embedding_mode",
        choices=("none", "zeros"),
        default="none",
        help=(
            "Direct LeRobot export: how to populate Octo embedding fields. "
            "`none` omits embeddings. `zeros` writes placeholder 384-dim vectors and is only for plumbing tests."
        ),
    )
    parser.add_argument(
        "--allow_zero_embeddings",
        action="store_true",
        help="Required with --embedding_mode=zeros to acknowledge that zero embeddings are placeholders, not real Octo features.",
    )
    parser.add_argument("--max_episodes", type=int, default=None, help="Debug: export at most this many episodes.")
    parser.add_argument("--max_transitions", type=int, default=None, help="Debug: export at most this many transitions.")
    args = parser.parse_args()
    if args.embedding_mode == "zeros" and not args.allow_zero_embeddings:
        raise ValueError(
            "--embedding_mode=zeros writes placeholder embeddings that satisfy current buffer fields but remove "
            "Octo conditioning information. Pass --allow_zero_embeddings only for smoke tests, or generate real "
            "Octo embeddings before using the pkl for training."
        )

    progress_by_episode = _load_progress(_progress_path(args.source_lerobot_dataset, args.progress_parquet), args.head_mode)

    if args.input_pkl:
        input_pkl = Path(args.input_pkl).expanduser().resolve()
        with input_pkl.open("rb") as f:
            transitions = pkl.load(f)
        trajectories = _split_trajectories(transitions)

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
        summary = f"Relabeled {len(trajectories)} trajectory(ies), {len(output)} transition(s)"
    else:
        if not args.source_lerobot_dataset:
            raise ValueError("Direct LeRobot export requires --source_lerobot_dataset when --input_pkl is omitted.")
        output = _build_conrft_from_lerobot(
            dataset_root=Path(args.source_lerobot_dataset).expanduser().resolve(),
            progress_by_episode=progress_by_episode,
            head_mode=args.head_mode,
            gamma=args.gamma,
            reward_scale=args.reward_scale,
            reward_bias=args.reward_bias,
            reward_clip_low=args.reward_clip_low,
            reward_clip_high=args.reward_clip_high,
            success_threshold=args.success_threshold,
            success_reward=args.success_reward,
            truncate_after_success=not args.no_truncate_after_success,
            obs_horizon=args.obs_horizon,
            include_images=bool(args.include_images),
            image_key_map=_parse_image_key_map(args.image_key_map),
            image_max_width=args.image_max_width,
            embedding_mode=args.embedding_mode,
            max_episodes=args.max_episodes,
            max_transitions=args.max_transitions,
        )
        summary = f"Exported LeRobotDataset to {len(output)} SARM-reward ConRFT transition(s)"

    output_pkl = Path(args.output_pkl).expanduser().resolve()
    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    with output_pkl.open("wb") as f:
        pkl.dump(output, f)
    print(summary)
    print(f"Wrote SARM-reward ConRFT pkl to {output_pkl}")


if __name__ == "__main__":
    main()
