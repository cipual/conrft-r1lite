#!/usr/bin/env python3

import argparse
import gc
import os
import pickle as pkl
import sys
from pathlib import Path
from typing import Dict, List

_EXAMPLES_DIR = Path(__file__).resolve().parents[1]
_REPO_ROOT = _EXAMPLES_DIR.parent
for path in (
    _EXAMPLES_DIR,
    _REPO_ROOT / "serl_launcher",
    _REPO_ROOT / "serl_robot_infra",
):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


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


def _load_octo_model():
    try:
        from octo.model.octo_model import OctoModel
    except ImportError as exc:
        raise RuntimeError(
            "Octo is required to add real embeddings. Run this script in the RWRL environment."
        ) from exc
    return OctoModel


def _configure_native_runtime(args):
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", str(args.xla_mem_fraction))
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    if args.jax_platform == "cpu":
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif args.jax_platform == "gpu":
        os.environ["JAX_PLATFORM_NAME"] = "gpu"
        os.environ["JAX_PLATFORMS"] = "cuda"


def _disable_tensorflow_gpu_if_needed(args):
    if args.jax_platform == "cpu":
        return
    import tensorflow as tf

    # Octo uses TensorFlow only for checkpoint/file utilities here. Let JAX own
    # CUDA; otherwise TF and JAX can both initialize cuDNN/cuBLAS and crash in
    # native code before Python can raise an exception.
    tf.config.set_visible_devices([], "GPU")


def main():
    parser = argparse.ArgumentParser(
        description="Add real Octo embeddings/next_embeddings to a ConRFT transition pkl."
    )
    parser.add_argument("--exp_name", default="r1lite_dual_mango_box")
    parser.add_argument("--input_pkl", required=True)
    parser.add_argument("--output_pkl", required=True)
    parser.add_argument(
        "--image_keys",
        default=None,
        help="Comma-separated image keys to feed to Octo. Defaults to the experiment config image_keys.",
    )
    parser.add_argument(
        "--jax_platform",
        choices=("auto", "cpu", "gpu"),
        default="auto",
        help="Set before importing JAX/Octo. Use cpu when GPU native libs segfault.",
    )
    parser.add_argument(
        "--xla_mem_fraction",
        type=float,
        default=0.35,
        help="Default XLA_PYTHON_CLIENT_MEM_FRACTION when not already set.",
    )
    parser.add_argument("--start_trajectory", type=int, default=0)
    parser.add_argument("--max_trajectories", type=int, default=None)
    parser.add_argument("--max_transitions_per_trajectory", type=int, default=None)
    args = parser.parse_args()
    _configure_native_runtime(args)
    _disable_tensorflow_gpu_if_needed(args)

    from experiments.mappings import CONFIG_MAPPING
    from data_util import add_embeddings_to_trajectory, add_next_embeddings_to_trajectory

    cfg = CONFIG_MAPPING[args.exp_name]()
    image_keys = (
        [item.strip() for item in args.image_keys.split(",") if item.strip()]
        if args.image_keys
        else list(cfg.image_keys)
    )
    if len(image_keys) < 2:
        raise ValueError(f"Octo embeddings require at least two image keys, got {image_keys}")

    input_pkl = Path(args.input_pkl).expanduser().resolve()
    with input_pkl.open("rb") as f:
        transitions = pkl.load(f)

    OctoModel = _load_octo_model()
    model = OctoModel.load_pretrained(cfg.octo_path)
    tasks = model.create_tasks(texts=[cfg.task_desc])

    output: List[Dict] = []
    trajectories = _split_trajectories(transitions)
    stop_trajectory = (
        len(trajectories)
        if args.max_trajectories is None
        else min(len(trajectories), args.start_trajectory + args.max_trajectories)
    )
    if args.start_trajectory < 0 or args.start_trajectory >= len(trajectories):
        raise IndexError(f"--start_trajectory={args.start_trajectory} out of range for {len(trajectories)} trajectories")

    for index in range(args.start_trajectory, stop_trajectory):
        trajectory = trajectories[index]
        if args.max_transitions_per_trajectory is not None:
            trajectory = trajectory[: max(0, args.max_transitions_per_trajectory)]
        print(f"Adding Octo embeddings: trajectory {index + 1}/{len(trajectories)} ({len(trajectory)} transitions)")
        trajectory = add_embeddings_to_trajectory(
            trajectory,
            model,
            tasks=tasks,
            image_keys=tuple(image_keys),
        )
        trajectory = add_next_embeddings_to_trajectory(trajectory)
        output.extend(trajectory)
        gc.collect()

    if (
        args.start_trajectory > 0
        or stop_trajectory < len(trajectories)
        or args.max_transitions_per_trajectory is not None
    ):
        print(
            f"Wrote only trajectories [{args.start_trajectory}, {stop_trajectory}) "
            "or a transition limit was selected. Do not use this partial pkl for training."
        )

    output_pkl = Path(args.output_pkl).expanduser().resolve()
    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    with output_pkl.open("wb") as f:
        pkl.dump(output, f)
    print(f"Wrote Octo-embedding ConRFT pkl to {output_pkl}")


if __name__ == "__main__":
    main()
