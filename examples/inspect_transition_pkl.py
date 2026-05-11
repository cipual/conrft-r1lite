#!/usr/bin/env python3
"""Inspect transition-style pickle files and view one selected trajectory."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def _looks_like_flat_transition(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    if "observations" not in item:
        return False
    return any(key in item for key in ("next_observations", "actions", "rewards", "dones", "infos"))


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


def _trajectory_steps(traj: Any) -> List[Dict[str, Any]]:
    if isinstance(traj, list):
        if not traj:
            return []
        if not isinstance(traj[0], dict):
            raise ValueError("Trajectory list entries are not dictionaries")
        return traj
    if isinstance(traj, dict):
        if "observations" in traj and isinstance(traj["observations"], list):
            n = len(traj["observations"])
            steps = []
            for i in range(n):
                step = {}
                for key, value in traj.items():
                    if isinstance(value, list) and len(value) == n:
                        step[key] = value[i]
                    else:
                        step[key] = value
                steps.append(step)
            return steps
        return [traj]
    raise ValueError(f"Unsupported trajectory type: {type(traj).__name__}")


def _summary_scalar(value: Any) -> str:
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6g}"
    return repr(value)


def _describe_value(value: Any, max_list_items: int = 6) -> str:
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return f"ndarray scalar={_summary_scalar(value.item())}"
        desc = f"ndarray shape={tuple(value.shape)} dtype={value.dtype}"
        if value.size <= max_list_items:
            desc += f" values={np.array2string(value, precision=4, suppress_small=True)}"
        return desc
    if isinstance(value, (list, tuple)):
        prefix = type(value).__name__
        if len(value) <= max_list_items and not any(isinstance(v, (dict, list, tuple, np.ndarray)) for v in value):
            return f"{prefix}[len={len(value)}] values={value}"
        return f"{prefix}[len={len(value)}]"
    if isinstance(value, dict):
        keys = list(value.keys())
        preview = keys[:max_list_items]
        suffix = "" if len(keys) <= max_list_items else " ..."
        return f"dict keys={preview}{suffix}"
    return _summary_scalar(value)


def _format_action_lines(action: Any) -> List[str]:
    arr = np.asarray(action, dtype=np.float32).reshape(-1)
    lines = [
        f"raw: shape={tuple(arr.shape)} values={np.array2string(arr, precision=4, suppress_small=True)}"
    ]
    if arr.size == 14:
        left = arr[:7]
        right = arr[7:14]
        lines.extend(
            [
                (
                    "left : "
                    f"xyz={np.array2string(left[:3], precision=4, suppress_small=True)} "
                    f"rpy={np.array2string(left[3:6], precision=4, suppress_small=True)} "
                    f"grip={left[6]:.4f}"
                ),
                (
                    "right: "
                    f"xyz={np.array2string(right[:3], precision=4, suppress_small=True)} "
                    f"rpy={np.array2string(right[3:6], precision=4, suppress_small=True)} "
                    f"grip={right[6]:.4f}"
                ),
            ]
        )
    elif arr.size == 7:
        lines.append(
            "arm  : "
            f"xyz={np.array2string(arr[:3], precision=4, suppress_small=True)} "
            f"rpy={np.array2string(arr[3:6], precision=4, suppress_small=True)} "
            f"grip={arr[6]:.4f}"
        )
    return lines


def _print_step(step: Dict[str, Any], step_index: int, indent: str = "  ") -> None:
    print(f"{indent}step[{step_index}]")
    for key in sorted(step.keys()):
        if key == "actions":
            print(f"{indent}  actions:")
            for line in _format_action_lines(step[key]):
                print(f"{indent}    {line}")
            continue
        print(f"{indent}  {key}: {_describe_value(step[key])}")
        if isinstance(step[key], dict):
            for child_key in sorted(step[key].keys()):
                child = step[key][child_key]
                print(f"{indent}    {key}.{child_key}: {_describe_value(child)}")


def _trajectory_meta(steps: List[Dict[str, Any]]) -> Dict[str, Any]:
    rewards = [float(s.get("rewards", 0.0)) for s in steps if "rewards" in s]
    dones = [bool(s.get("dones", False)) for s in steps]
    infos = [s.get("infos") for s in steps if isinstance(s.get("infos"), dict)]
    succeed = None
    for info in reversed(infos):
        if "succeed" in info:
            succeed = bool(info["succeed"])
            break
        if "success" in info:
            succeed = bool(info["success"])
            break
    first_nonzero_action_step = None
    for i, step in enumerate(steps):
        action = step.get("actions")
        if action is None:
            continue
        arr = np.asarray(action)
        if arr.size > 0 and np.any(np.abs(arr) > 1e-8):
            first_nonzero_action_step = i
            break
    return {
        "steps": len(steps),
        "done_last": dones[-1] if dones else None,
        "reward_sum": float(sum(rewards)) if rewards else None,
        "reward_first": rewards[0] if rewards else None,
        "reward_last": rewards[-1] if rewards else None,
        "succeed": succeed,
        "first_nonzero_action_step": first_nonzero_action_step,
    }


def _parse_step_selection(spec: str, num_steps: int) -> List[int]:
    if ":" in spec:
        left, right = spec.split(":", 1)
        start = int(left) if left else 0
        end = int(right) if right else num_steps
        start = max(0, start)
        end = min(num_steps, end)
        return list(range(start, end))
    idx = int(spec)
    if idx < 0:
        idx += num_steps
    if idx < 0 or idx >= num_steps:
        raise IndexError(f"step index {idx} out of range for {num_steps} steps")
    return [idx]


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a transition pkl and select one trajectory to view.")
    parser.add_argument("--input_file", required=True, help="Path to .pkl file")
    parser.add_argument("--trajectory_index", type=int, default=0, help="Trajectory index to inspect")
    parser.add_argument("--list_only", action="store_true", help="Only list trajectory summaries")
    parser.add_argument(
        "--steps",
        default="0",
        help="Step selection inside the chosen trajectory. Examples: 0, -1, 10:20, :5, 100:",
    )
    args = parser.parse_args()

    input_file = Path(args.input_file).expanduser().resolve()
    data = _load_pickle(input_file)
    trajectories, source = _as_trajectory_list(data)

    print(f"Loaded: {input_file}")
    print(f"Top-level source: {source}")
    print(f"Num trajectories: {len(trajectories)}")
    for i, traj in enumerate(trajectories):
        steps = _trajectory_steps(traj)
        meta = _trajectory_meta(steps)
        print(
            f"  trajectory[{i}]: steps={meta['steps']}, succeed={meta['succeed']}, "
            f"reward_sum={meta['reward_sum']}, first_nonzero_action_step={meta['first_nonzero_action_step']}"
        )

    if args.list_only:
        return

    if args.trajectory_index < 0 or args.trajectory_index >= len(trajectories):
        raise IndexError(f"trajectory_index={args.trajectory_index} out of range for {len(trajectories)} trajectories")

    steps = _trajectory_steps(trajectories[args.trajectory_index])
    meta = _trajectory_meta(steps)
    print()
    print(f"Selected trajectory[{args.trajectory_index}]")
    for key in ("steps", "done_last", "reward_sum", "reward_first", "reward_last", "succeed", "first_nonzero_action_step"):
        print(f"  {key}: {meta[key]}")

    if not steps:
        print("  empty trajectory")
        return

    selected_steps = _parse_step_selection(args.steps, len(steps))
    print(f"  inspect_steps: {selected_steps[0]}" if len(selected_steps) == 1 else f"  inspect_steps: {args.steps}")
    print()
    for step_index in selected_steps:
        _print_step(steps[step_index], step_index)
        print()


if __name__ == "__main__":
    main()
