#!/usr/bin/env python3
"""Probe a ConRFT checkpoint on offline training observations.

This script does not connect to the robot. It loads the experiment config,
restores an offline checkpoint, samples policy actions on observations from a
transition PKL, and compares those sampled actions against the training action
distribution.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import pickle as pkl
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import cv2
import jax
import jax.numpy as jnp
import numpy as np
from flax.training import checkpoints


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
for rel in ("examples", "serl_robot_infra", "serl_launcher", "octo"):
    sys.path.insert(0, str(REPO_ROOT / rel))

from experiments.mappings import CONFIG_MAPPING  # noqa: E402
from octo.model.octo_model import OctoModel  # noqa: E402
from serl_launcher.utils.launcher import make_conrft_octo_cp_pixel_agent_single_arm  # noqa: E402


DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "examples/experiments/r1lite_dual_mango_box/conrft_sarm"
)
DEFAULT_TRANSITIONS = (
    REPO_ROOT
    / "data/transition/r1lite_dual_mango_box/r1lite_dual_mango_box_sarm_reward_fixed.pkl"
)
DEFAULT_OUTPUT_DIR = SCRIPT_PATH.parent / "outputs"
DEFAULT_CACHE = DEFAULT_OUTPUT_DIR / "checkpoint_action_probe_cache.npz"

ACTION_LABELS = (
    "left_x",
    "left_y",
    "left_z",
    "left_r",
    "left_p",
    "left_yaw",
    "left_grip",
    "right_x",
    "right_y",
    "right_z",
    "right_r",
    "right_p",
    "right_yaw",
    "right_grip",
)
ACTION_GROUPS = {
    "left_xyz": slice(0, 3),
    "left_rpy": slice(3, 6),
    "right_xyz": slice(7, 10),
    "right_rpy": slice(10, 13),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample a restored ConRFT checkpoint on offline training observations."
    )
    parser.add_argument("--exp_name", default="r1lite_dual_mango_box")
    parser.add_argument("--checkpoint_path", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--checkpoint_step", type=int, default=None)
    parser.add_argument("--transitions_pkl", type=Path, default=DEFAULT_TRANSITIONS)
    parser.add_argument("--num_obs", type=int, default=64)
    parser.add_argument("--samples_per_obs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--selection",
        choices=("uniform", "first", "random"),
        default="uniform",
        help="Which observations to sample from the transition PKL.",
    )
    parser.add_argument("--q_weight", type=float, default=0.1)
    parser.add_argument("--bc_weight", type=float, default=1.0)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache_path", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--rebuild_cache", action="store_true")
    parser.add_argument(
        "--prepare_cache_only",
        action="store_true",
        help="Only build the small NPZ cache from the large transition PKL, then exit.",
    )
    parser.add_argument(
        "--continue_after_cache_build",
        action="store_true",
        help="Continue into model inference after building a missing cache in this process.",
    )
    parser.add_argument("--no_csv", action="store_true")
    return parser.parse_args()


def checkpoint_root_and_step(path: Path, step: int | None) -> Tuple[Path, int | None]:
    path = path.expanduser().resolve()
    if path.name.startswith("checkpoint_"):
        parsed_step = int(path.name.removeprefix("checkpoint_"))
        return path.parent, parsed_step if step is None else step
    return path, step


def select_indices(n_items: int, n_select: int, selection: str, seed: int) -> np.ndarray:
    n_select = min(int(n_select), int(n_items))
    if selection == "first":
        return np.arange(n_select, dtype=np.int64)
    if selection == "random":
        rng = np.random.default_rng(seed)
        return np.sort(rng.choice(n_items, size=n_select, replace=False))
    if n_select == 1:
        return np.array([0], dtype=np.int64)
    return np.linspace(0, n_items - 1, n_select, dtype=np.int64)


def resize_transition_obs(obs: Dict, observation_space, image_keys: Iterable[str]) -> Dict:
    updated = {}
    for key, value in obs.items():
        if key == "state":
            state = np.asarray(value, dtype=np.float32)
            target_shape = tuple(observation_space["state"].shape)
            if state.shape != target_shape:
                if state.ndim == 1 and len(target_shape) == 2 and state.shape[0] == target_shape[-1]:
                    state = np.repeat(state[None, :], target_shape[0], axis=0)
                elif state.ndim == 2 and len(target_shape) == 2 and state.shape[-1] == target_shape[-1]:
                    if state.shape[0] < target_shape[0]:
                        pad = np.repeat(state[:1], target_shape[0] - state.shape[0], axis=0)
                        state = np.concatenate([pad, state], axis=0)
                    elif state.shape[0] > target_shape[0]:
                        state = state[-target_shape[0] :]
                else:
                    raise ValueError(f"state shape {state.shape} does not match {target_shape}")
            updated[key] = state.astype(np.float32)
            continue

        if key in image_keys:
            target_shape = tuple(observation_space[key].shape)
            target_hw = target_shape[-3:-1]
            images = np.asarray(value)
            if images.shape[-3:-1] != target_hw:
                if images.ndim != 4:
                    raise ValueError(f"expected stacked images for {key}, got {images.shape}")
                images = np.stack(
                    [
                        cv2.resize(frame, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_AREA)
                        for frame in images
                    ],
                    axis=0,
                )
            updated[key] = images.astype(np.uint8)
            continue

        updated[key] = value
    return updated


def create_agent(config, env, tasks, octo_model, seed: int, q_weight: float, bc_weight: float):
    if config.setup_mode not in {
        "single-arm-fixed-gripper",
        "single-arm-learned-gripper",
        "dual-arm-learned-gripper",
    }:
        raise NotImplementedError(f"Unknown setup mode: {config.setup_mode}")

    return make_conrft_octo_cp_pixel_agent_single_arm(
        seed=seed,
        sample_obs=env.observation_space.sample(),
        sample_action=env.action_space.sample(),
        sample_tasks=tasks,
        octo_model=octo_model,
        image_keys=config.image_keys,
        encoder_type=config.encoder_type,
        discount=config.discount,
        fix_gripper=config.setup_mode == "single-arm-fixed-gripper",
        q_weight=q_weight,
        bc_weight=bc_weight,
    )


def summarize_actions(name: str, actions: np.ndarray) -> Dict:
    result = {"name": name, "count": int(actions.shape[0])}
    quantiles = [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]
    result["dims"] = {}
    for i, label in enumerate(ACTION_LABELS):
        values = actions[:, i]
        result["dims"][label] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "quantiles": {str(q): float(v) for q, v in zip(quantiles, np.quantile(values, quantiles))},
        }
    result["groups"] = {}
    for group, sl in ACTION_GROUPS.items():
        values = np.linalg.norm(actions[:, sl], axis=1)
        result["groups"][group] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "quantiles": {str(q): float(v) for q, v in zip(quantiles, np.quantile(values, quantiles))},
        }
    return result


def percentile_against(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    sorted_ref = np.sort(reference)
    return np.searchsorted(sorted_ref, values, side="right") / max(len(sorted_ref), 1) * 100.0


def comparison_summary(train_actions: np.ndarray, sampled_actions: np.ndarray) -> Dict:
    out = {"dims": {}, "groups": {}}
    for i, label in enumerate(ACTION_LABELS):
        abs_pct = percentile_against(np.abs(train_actions[:, i]), np.abs(sampled_actions[:, i]))
        signed_pct = percentile_against(train_actions[:, i], sampled_actions[:, i])
        out["dims"][label] = {
            "sample_abs_min": float(np.min(np.abs(sampled_actions[:, i]))),
            "sample_abs_median": float(np.median(np.abs(sampled_actions[:, i]))),
            "sample_abs_max": float(np.max(np.abs(sampled_actions[:, i]))),
            "signed_percentile_median": float(np.median(signed_pct)),
            "abs_percentile_median": float(np.median(abs_pct)),
            "abs_percentile_95": float(np.quantile(abs_pct, 0.95)),
        }
    for group, sl in ACTION_GROUPS.items():
        train_norm = np.linalg.norm(train_actions[:, sl], axis=1)
        sample_norm = np.linalg.norm(sampled_actions[:, sl], axis=1)
        pct = percentile_against(train_norm, sample_norm)
        out["groups"][group] = {
            "sample_norm_min": float(np.min(sample_norm)),
            "sample_norm_median": float(np.median(sample_norm)),
            "sample_norm_max": float(np.max(sample_norm)),
            "norm_percentile_median": float(np.median(pct)),
            "norm_percentile_95": float(np.quantile(pct, 0.95)),
            "fraction_above_train_p95": float(np.mean(pct >= 95.0)),
            "fraction_above_train_p99": float(np.mean(pct >= 99.0)),
        }
    return out


def write_action_csv(path: Path, indices: np.ndarray, sampled_actions: np.ndarray, samples_per_obs: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["transition_index", "sample_index", *ACTION_LABELS])
        row_id = 0
        for transition_index in indices:
            for sample_index in range(samples_per_obs):
                writer.writerow(
                    [
                        int(transition_index),
                        sample_index,
                        *[f"{v:.8f}" for v in sampled_actions[row_id]],
                    ]
                )
                row_id += 1


def prepare_cache(args: argparse.Namespace, env, config) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], bool]:
    cache_path = args.cache_path.expanduser().resolve()
    if cache_path.exists() and not args.rebuild_cache:
        data = np.load(cache_path)
        train_actions = data["train_actions"].astype(np.float32)
        indices = data["indices"].astype(np.int64)
        selected_observations = {
            "state": data["obs_state"].astype(np.float32),
            **{key: data[f"obs_{key}"].astype(np.uint8) for key in config.image_keys},
        }
        return train_actions, indices, selected_observations, False

    print("[debug] loading transitions to build cache...")
    with args.transitions_pkl.expanduser().resolve().open("rb") as f:
        transitions = pkl.load(f)

    train_actions = np.stack(
        [np.asarray(t["actions"], dtype=np.float32).reshape(-1) for t in transitions],
        axis=0,
    )
    indices = select_indices(len(transitions), args.num_obs, args.selection, args.seed)
    selected = [
        resize_transition_obs(
            transitions[int(idx)]["observations"],
            env.observation_space,
            config.image_keys,
        )
        for idx in indices
    ]
    selected_observations = {
        "state": np.stack([obs["state"] for obs in selected], axis=0).astype(np.float32),
    }
    for key in config.image_keys:
        selected_observations[key] = np.stack([obs[key] for obs in selected], axis=0).astype(np.uint8)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        train_actions=train_actions,
        indices=indices,
        obs_state=selected_observations["state"],
        **{f"obs_{key}": selected_observations[key] for key in config.image_keys},
    )
    del transitions
    gc.collect()
    print(f"[debug] wrote cache: {cache_path}")
    return train_actions, indices, selected_observations, True


def main():
    args = parse_args()
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    checkpoint_root, checkpoint_step = checkpoint_root_and_step(args.checkpoint_path, args.checkpoint_step)
    latest = checkpoints.latest_checkpoint(str(checkpoint_root))
    if checkpoint_step is None and latest is None:
        raise FileNotFoundError(f"No checkpoint found under {checkpoint_root}")

    print(f"[debug] repo_root={REPO_ROOT}")
    print(f"[debug] checkpoint_root={checkpoint_root}")
    print(f"[debug] checkpoint_step={checkpoint_step if checkpoint_step is not None else 'latest'}")
    print(f"[debug] transitions_pkl={args.transitions_pkl}")

    config = CONFIG_MAPPING[args.exp_name]()
    if hasattr(config, "reward_model_config"):
        config.reward_model_config.enabled = False
    env = config.get_environment(fake_env=True, classifier=True, stack_obs_num=2)

    train_actions, indices, selected_observations, cache_built = prepare_cache(args, env, config)
    num_transitions = int(train_actions.shape[0])
    print(f"[debug] selected {len(indices)} observation(s), samples_per_obs={args.samples_per_obs}")
    if args.prepare_cache_only:
        print("[debug] prepare_cache_only requested; exiting before model load.")
        return
    if cache_built and not args.continue_after_cache_build:
        print("[debug] cache was built in this process. Re-run the same command to load the cache and run inference.")
        return

    print("[debug] loading Octo model...")
    octo_model = OctoModel.load_pretrained(config.octo_path)
    tasks = octo_model.create_tasks(texts=[config.task_desc])
    agent = create_agent(config, env, tasks, octo_model, args.seed, args.q_weight, args.bc_weight)

    print("[debug] restoring checkpoint...")
    restored = checkpoints.restore_checkpoint(
        str(checkpoint_root),
        agent.state,
        step=checkpoint_step,
    )
    agent = agent.replace(
        state=agent.state.replace(
            params=restored.params,
            target_params=restored.target_params,
        )
    )

    rng = jax.random.PRNGKey(args.seed)
    sampled = []
    for obs_idx in range(len(indices)):
        obs = {
            "state": selected_observations["state"][obs_idx],
            **{key: selected_observations[key][obs_idx] for key in config.image_keys},
        }
        obs = jax.device_put(obs)
        for _ in range(args.samples_per_obs):
            rng, key = jax.random.split(rng)
            actions, _ = agent.sample_actions(
                observations=obs,
                tasks=jax.device_put(tasks),
                seed=key,
            )
            sampled.append(np.asarray(jax.device_get(actions), dtype=np.float32).reshape(-1))
    sampled_actions = np.stack(sampled, axis=0)

    summary = {
        "checkpoint_root": str(checkpoint_root),
        "checkpoint_step": checkpoint_step,
        "latest_checkpoint": latest,
        "transitions_pkl": str(args.transitions_pkl),
        "num_transitions": num_transitions,
        "selected_indices": indices.tolist(),
        "samples_per_obs": int(args.samples_per_obs),
        "train_actions": summarize_actions("train_actions", train_actions),
        "sampled_actions": summarize_actions("sampled_actions", sampled_actions),
        "comparison": comparison_summary(train_actions, sampled_actions),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "checkpoint_action_probe_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if not args.no_csv:
        write_action_csv(
            args.output_dir / "checkpoint_action_probe_samples.csv",
            indices,
            sampled_actions,
            args.samples_per_obs,
        )

    print("\n[group norm comparison: sampled action percentiles vs train action distribution]")
    for group, stats in summary["comparison"]["groups"].items():
        print(
            f"{group:10s} "
            f"sample_norm median={stats['sample_norm_median']:.4f} max={stats['sample_norm_max']:.4f} "
            f"pct_median={stats['norm_percentile_median']:.1f} "
            f"pct95={stats['norm_percentile_95']:.1f} "
            f"frac>=p95={stats['fraction_above_train_p95']:.2%} "
            f"frac>=p99={stats['fraction_above_train_p99']:.2%}"
        )

    print("\n[largest sampled-action abs percentiles by dimension]")
    dim_rows = sorted(
        summary["comparison"]["dims"].items(),
        key=lambda item: item[1]["abs_percentile_95"],
        reverse=True,
    )
    for label, stats in dim_rows[:8]:
        print(
            f"{label:10s} "
            f"abs_median={stats['sample_abs_median']:.4f} abs_max={stats['sample_abs_max']:.4f} "
            f"abs_pct_median={stats['abs_percentile_median']:.1f} "
            f"abs_pct95={stats['abs_percentile_95']:.1f}"
        )

    print(f"\n[debug] wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
