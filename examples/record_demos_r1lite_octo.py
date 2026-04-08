import copy
import datetime
import os
import pickle as pkl
import time

import numpy as np
from absl import app, flags
from tqdm import tqdm

from data_util import (
    add_embeddings_to_trajectory,
    add_mc_returns_to_trajectory,
    add_next_embeddings_to_trajectory,
)
from experiments.mappings import CONFIG_MAPPING
from octo.model.octo_model import OctoModel

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "R1Lite experiment name.")
flags.DEFINE_integer("successes_needed", 10, "Number of successful demos to collect.")
flags.DEFINE_float("reward_scale", 1.0, "Reward scale used for mc_returns.")
flags.DEFINE_float("reward_bias", 0.0, "Reward bias used for mc_returns.")
flags.DEFINE_string(
    "output_dir",
    None,
    "Optional output directory. Defaults to examples/experiments/<exp_name>/demo_data.",
)
flags.DEFINE_float(
    "reset_wait_sec",
    2.0,
    "Sleep after each reset so the operator can stabilize the setup before the next rollout.",
)


def _default_output_dir(exp_name: str) -> str:
    root = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(root, "experiments", exp_name, "demo_data")


def main(_):
    assert FLAGS.exp_name in CONFIG_MAPPING, "Experiment folder not found."
    config = CONFIG_MAPPING[FLAGS.exp_name]()

    # 先加载 Octo，再创建带 SpaceMouse wrapper 的 env。
    # 否则 SpaceMouse 会过早校准，等模型加载完成后再开始录制时零偏可能已经漂了。
    model = OctoModel.load_pretrained(config.octo_path)
    tasks = model.create_tasks(texts=[config.task_desc])
    env = config.get_environment(
        fake_env=False,
        save_video=False,
        classifier=True,
        stack_obs_num=2,
    )
    pbar = tqdm(total=FLAGS.successes_needed)
    try:
        obs, info = env.reset()
        print(f"Recording R1Lite demos for {FLAGS.exp_name}")
        print(f"Environment reset complete. Observation keys: {list(obs.keys())}")

        transitions = []
        trajectory = []
        success_count = 0
        returns = 0.0

        while success_count < FLAGS.successes_needed:
            policy_action = np.zeros(env.action_space.sample().shape, dtype=np.float32)
            next_obs, rew, done, truncated, info = env.step(policy_action)
            returns += rew
            recorded_action = info.get("intervene_action", policy_action)

            transition = copy.deepcopy(
                dict(
                    observations=obs,
                    actions=recorded_action,
                    next_observations=next_obs,
                    rewards=rew,
                    masks=1.0 - done,
                    dones=done,
                    infos=info,
                )
            )
            trajectory.append(transition)
            pbar.set_description(f"Return: {returns:.2f}")

            obs = next_obs
            if done:
                if info.get("succeed", False):
                    trajectory = add_mc_returns_to_trajectory(
                        trajectory,
                        config.discount,
                        FLAGS.reward_scale,
                        FLAGS.reward_bias,
                        config.reward_neg,
                        is_sparse_reward=True,
                    )
                    trajectory = add_embeddings_to_trajectory(
                        trajectory,
                        model,
                        tasks=tasks,
                        image_keys=tuple(config.image_keys),
                    )
                    trajectory = add_next_embeddings_to_trajectory(trajectory)
                    for transition in trajectory:
                        transitions.append(copy.deepcopy(transition))
                    success_count += 1
                    pbar.update(1)
                trajectory = []
                returns = 0.0
                obs, info = env.reset()
                time.sleep(max(0.0, FLAGS.reset_wait_sec))

        output_dir = FLAGS.output_dir or _default_output_dir(FLAGS.exp_name)
        os.makedirs(output_dir, exist_ok=True)
        uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_name = os.path.join(output_dir, f"{FLAGS.exp_name}_{FLAGS.successes_needed}_demos_{uuid}.pkl")
        with open(file_name, "wb") as f:
            pkl.dump(transitions, f)
        print(f"Saved {FLAGS.successes_needed} successful demos to {file_name}")
    finally:
        pbar.close()
        env.close()


if __name__ == "__main__":
    app.run(main)
