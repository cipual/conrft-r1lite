#!/usr/bin/env python3

import argparse
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

from data_util import add_embeddings_to_trajectory, add_next_embeddings_to_trajectory  # noqa: E402
from experiments.mappings import CONFIG_MAPPING  # noqa: E402


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
    args = parser.parse_args()

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
    for index, trajectory in enumerate(trajectories):
        print(f"Adding Octo embeddings: trajectory {index + 1}/{len(trajectories)} ({len(trajectory)} transitions)")
        trajectory = add_embeddings_to_trajectory(
            trajectory,
            model,
            tasks=tasks,
            image_keys=tuple(image_keys),
        )
        trajectory = add_next_embeddings_to_trajectory(trajectory)
        output.extend(trajectory)

    output_pkl = Path(args.output_pkl).expanduser().resolve()
    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    with output_pkl.open("wb") as f:
        pkl.dump(output, f)
    print(f"Wrote Octo-embedding ConRFT pkl to {output_pkl}")


if __name__ == "__main__":
    main()
