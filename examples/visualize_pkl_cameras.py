#!/usr/bin/env python3
"""Visualize camera streams stored inside ConRFT-style pickle trajectories.

The script is intentionally independent from the RL environment. It loads a pkl,
auto-discovers image-like observation keys, and writes MP4 previews.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def _looks_like_flat_transition(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    if "observations" not in item:
        return False
    if not any(key in item for key in ("next_observations", "actions", "rewards", "dones", "infos")):
        return False
    # A single transition stores one observation dict. A trajectory dict usually
    # stores observations as a list/tuple or a time-major dict.
    return isinstance(item.get("observations"), dict)


def _group_flat_transitions(transitions: Sequence[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    if not transitions:
        return []

    has_episode_index = all(
        isinstance(item.get("infos"), dict) and "episode_index" in item["infos"] for item in transitions
    )
    if has_episode_index:
        grouped: Dict[int, List[Dict[str, Any]]] = {}
        for item in transitions:
            grouped.setdefault(int(item["infos"]["episode_index"]), []).append(item)
        return [grouped[key] for key in sorted(grouped)]

    trajectories: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    for item in transitions:
        current.append(item)
        if bool(item.get("dones", False)):
            trajectories.append(current)
            current = []
    if current:
        trajectories.append(current)
    return trajectories


def _as_trajectory_list(data: Any) -> Tuple[List[Any], str]:
    if isinstance(data, dict):
        for key in ("trajectories", "demos", "episodes", "data"):
            value = data.get(key)
            if isinstance(value, list):
                if value and _looks_like_flat_transition(value[0]):
                    return _group_flat_transitions(value), f"flat_transitions:{key}"
                return value, key
        if "observations" in data:
            return [data], "single_trajectory"
    if isinstance(data, list):
        if data and _looks_like_flat_transition(data[0]):
            return _group_flat_transitions(data), "flat_transitions"
        return data, "trajectory_list"
    raise ValueError(f"Unsupported pkl top-level type: {type(data).__name__}")


def _section_steps(traj: Any, section: str) -> List[Dict[str, Any]]:
    if isinstance(traj, list):
        if not traj:
            return []
        if isinstance(traj[0], dict) and section in traj[0]:
            return [item[section] for item in traj]
        if isinstance(traj[0], dict):
            return traj
        raise ValueError("Trajectory list entries are not dictionaries")

    if not isinstance(traj, dict):
        raise ValueError(f"Unsupported trajectory type: {type(traj).__name__}")

    if section not in traj:
        if section == "observations" and all(isinstance(k, str) for k in traj.keys()):
            return [traj]
        raise KeyError(f"Trajectory does not contain section {section!r}")

    obs = traj[section]
    if isinstance(obs, list):
        return obs
    if isinstance(obs, tuple):
        return list(obs)
    if isinstance(obs, dict):
        lengths = []
        for value in obs.values():
            if hasattr(value, "shape") and getattr(value, "ndim", 0) >= 1:
                lengths.append(int(value.shape[0]))
            elif isinstance(value, (list, tuple)):
                lengths.append(len(value))
        if not lengths:
            return [obs]
        n = min(lengths)
        steps = []
        for i in range(n):
            step = {}
            for key, value in obs.items():
                if hasattr(value, "shape") and getattr(value, "ndim", 0) >= 1 and value.shape[0] >= n:
                    step[key] = value[i]
                elif isinstance(value, (list, tuple)) and len(value) >= n:
                    step[key] = value[i]
                else:
                    step[key] = value
            steps.append(step)
        return steps

    raise ValueError(f"Unsupported {section!r} type: {type(obs).__name__}")


def _flatten_dict(value: Any, prefix: str = "") -> Dict[str, Any]:
    if not isinstance(value, dict):
        return {prefix: value} if prefix else {}
    out: Dict[str, Any] = {}
    for key, item in value.items():
        child = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, dict):
            out.update(_flatten_dict(item, child))
        else:
            out[child] = item
    return out


def _looks_like_image(value: Any) -> bool:
    arr = np.asarray(value)
    if arr.ndim < 3:
        return False
    if arr.shape[-1] not in (1, 3, 4):
        return False
    if arr.shape[-2] < 8 or arr.shape[-3] < 8:
        return False
    if arr.dtype == object:
        return False
    return True


def _discover_image_keys(steps: Sequence[Dict[str, Any]], max_probe_steps: int = 20) -> List[str]:
    counts: Dict[str, int] = {}
    for step in steps[:max_probe_steps]:
        for key, value in _flatten_dict(step).items():
            try:
                if _looks_like_image(value):
                    counts[key] = counts.get(key, 0) + 1
            except Exception:
                continue
    return sorted(counts, key=lambda k: (-counts[k], k))


def _nested_get(step: Dict[str, Any], key: str) -> Any:
    cur: Any = step
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(key)
        cur = cur[part]
    return cur


def _select_frame(image: Any, stack_index: int) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 5:
        # Common accidental batch dimension: keep the first batch element.
        arr = arr[0]
    if arr.ndim == 4:
        idx = stack_index
        if idx < 0:
            idx = arr.shape[0] + idx
        idx = max(0, min(idx, arr.shape[0] - 1))
        arr = arr[idx]
    if arr.ndim != 3:
        raise ValueError(f"Expected image or image stack, got shape {arr.shape}")
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return _to_uint8_rgb(arr)


def _to_uint8_rgb(frame: np.ndarray) -> np.ndarray:
    if frame.dtype == np.uint8:
        return np.ascontiguousarray(frame)
    frame = frame.astype(np.float32)
    finite = np.isfinite(frame)
    if not finite.all():
        frame = np.where(finite, frame, 0.0)
    max_value = float(frame.max()) if frame.size else 0.0
    min_value = float(frame.min()) if frame.size else 0.0
    if min_value >= 0.0 and max_value <= 1.5:
        frame = frame * 255.0
    return np.ascontiguousarray(np.clip(frame, 0, 255).astype(np.uint8))


def _resize_to_height(frame: np.ndarray, height: int) -> np.ndarray:
    if frame.shape[0] == height:
        return frame
    width = max(1, int(round(frame.shape[1] * (height / float(frame.shape[0])))))
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def _write_video(path: Path, frames: Sequence[np.ndarray], fps: float, codec: str) -> None:
    if not frames:
        raise ValueError(f"No frames to write for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {path} with codec={codec!r}")
    try:
        for frame in frames:
            if frame.shape[:2] != (h, w):
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def _safe_name(key: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in key)


def _collect_frames(
    steps: Sequence[Dict[str, Any]],
    image_keys: Sequence[str],
    start_step: int,
    max_steps: Optional[int],
    stride: int,
    stack_index: int,
) -> Dict[str, List[np.ndarray]]:
    end = len(steps) if max_steps is None else min(len(steps), start_step + max_steps)
    frames_by_key = {key: [] for key in image_keys}
    for step in steps[start_step:end:stride]:
        for key in image_keys:
            try:
                frames_by_key[key].append(_select_frame(_nested_get(step, key), stack_index))
            except Exception as exc:
                raise RuntimeError(f"Failed to read image key {key!r} at selected step: {exc}") from exc
    return frames_by_key


def _make_grid_frames(frames_by_key: Dict[str, List[np.ndarray]], labels: bool) -> List[np.ndarray]:
    keys = list(frames_by_key.keys())
    if not keys:
        return []
    n = min(len(frames_by_key[key]) for key in keys)
    grid_frames: List[np.ndarray] = []
    for i in range(n):
        row = [frames_by_key[key][i] for key in keys]
        target_h = min(frame.shape[0] for frame in row)
        row = [_resize_to_height(frame, target_h) for frame in row]
        if labels:
            labeled = []
            for key, frame in zip(keys, row):
                frame = frame.copy()
                cv2.putText(
                    frame,
                    key,
                    (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    key,
                    (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )
                labeled.append(frame)
            row = labeled
        grid_frames.append(np.concatenate(row, axis=1))
    return grid_frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_file", required=True, help="Path to a ConRFT/debug transition pkl.")
    parser.add_argument("--trajectory_index", type=int, default=0)
    parser.add_argument("--section", choices=("observations", "next_observations"), default="observations")
    parser.add_argument(
        "--image_keys",
        default="auto",
        help="Comma-separated image keys, e.g. head,left_wrist. Use 'auto' to discover image-like keys.",
    )
    parser.add_argument("--output_dir", default=None, help="Directory for output videos. Defaults next to input pkl.")
    parser.add_argument("--prefix", default=None, help="Output filename prefix. Defaults to input stem + trajectory id.")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--codec", default="mp4v", help="OpenCV fourcc, e.g. mp4v or avc1.")
    parser.add_argument("--start_step", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--stack_index",
        type=int,
        default=-1,
        help="For stacked images shaped T,H,W,C, choose which stack frame to show. Default: last frame.",
    )
    parser.add_argument("--no_grid", action="store_true", help="Do not write the multi-camera side-by-side video.")
    parser.add_argument("--no_separate", action="store_true", help="Do not write one video per camera key.")
    parser.add_argument("--no_labels", action="store_true", help="Do not draw camera key labels on the grid video.")
    parser.add_argument("--list_keys", action="store_true", help="Only list discovered image keys and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_file = Path(args.input_file).expanduser().resolve()
    data = _load_pickle(input_file)
    trajectories, layout = _as_trajectory_list(data)
    if not trajectories:
        raise ValueError("No trajectories found in pkl")
    if args.trajectory_index < 0 or args.trajectory_index >= len(trajectories):
        raise IndexError(f"trajectory_index={args.trajectory_index} out of range for {len(trajectories)} trajectories")

    steps = _section_steps(trajectories[args.trajectory_index], args.section)
    if not steps:
        raise ValueError(f"Trajectory {args.trajectory_index} has no {args.section} steps")

    discovered = _discover_image_keys(steps)
    if args.image_keys == "auto":
        image_keys = discovered
    else:
        image_keys = [item.strip() for item in args.image_keys.split(",") if item.strip()]

    print(f"Loaded {input_file}")
    print(
        f"layout={layout} trajectories={len(trajectories)} "
        f"trajectory_index={args.trajectory_index} section={args.section} steps={len(steps)}"
    )
    print("Discovered image keys:")
    for key in discovered:
        sample = _nested_get(steps[0], key)
        print(f"  {key}: shape={np.asarray(sample).shape} dtype={np.asarray(sample).dtype}")

    if args.list_keys:
        return
    if not image_keys:
        raise ValueError("No image keys selected. Pass --image_keys explicitly if auto-discovery missed them.")

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else input_file.parent / "pkl_video_preview"
    prefix = args.prefix or f"{input_file.stem}_traj{args.trajectory_index:03d}_{args.section}"
    frames_by_key = _collect_frames(
        steps=steps,
        image_keys=image_keys,
        start_step=max(0, args.start_step),
        max_steps=args.max_steps,
        stride=max(1, args.stride),
        stack_index=args.stack_index,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_separate:
        for key, frames in frames_by_key.items():
            path = output_dir / f"{prefix}_{_safe_name(key)}.mp4"
            _write_video(path, frames, args.fps / max(1, args.stride), args.codec)
            print(f"Wrote {path} frames={len(frames)} shape={frames[0].shape}")

    if not args.no_grid and len(frames_by_key) > 1:
        grid_frames = _make_grid_frames(frames_by_key, labels=not args.no_labels)
        path = output_dir / f"{prefix}_grid.mp4"
        _write_video(path, grid_frames, args.fps / max(1, args.stride), args.codec)
        print(f"Wrote {path} frames={len(grid_frames)} shape={grid_frames[0].shape}")


if __name__ == "__main__":
    main()
