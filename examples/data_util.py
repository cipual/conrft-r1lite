import numpy as np
import jax
import cv2


def calc_return_to_go(rewards, terminals, gamma, reward_scale, reward_bias, reward_neg, is_sparse_reward):
    """
    A config dict for getting the default high/low rewrd values for each envs
    """
    if len(rewards) == 0:
        return np.array([])

    if is_sparse_reward:
        reward_neg = reward_neg * reward_scale + reward_bias
    else:
        assert not is_sparse_reward, "If you want to try on a sparse reward env, please add the reward_neg value in the ENV_CONFIG dict."

    if is_sparse_reward and np.all(np.array(rewards) == reward_neg):
        """
        If the env has sparse reward and the trajectory is all negative rewards,
        we use r / (1-gamma) as return to go.
        For exapmle, if gamma = 0.99 and the rewards = [-1, -1, -1],
        then return_to_go = [-100, -100, -100]
        """
        return_to_go = [float(reward_neg / (1-gamma))] * len(rewards)
    else:
        return_to_go = [0] * len(rewards)
        prev_return = 0
        for i in range(len(rewards)):
            return_to_go[-i-1] = rewards[-i-1] + gamma * \
                prev_return * (1 - terminals[-i-1])
            prev_return = return_to_go[-i-1]

    return np.array(return_to_go, dtype=np.float32)


def add_mc_returns_to_trajectory(trajectory, gamma, reward_scale, reward_bias, reward_neg, is_sparse_reward):
    """
    undate every transition in the trajectory and add mc_returns
    return the updated trajectory
    """
    rewards = [t['rewards'] for t in trajectory]
    terminals = [t['dones'] for t in trajectory]

    mc_returns = calc_return_to_go(
        rewards=rewards,
        terminals=terminals,
        gamma=gamma,
        reward_scale=reward_scale,
        reward_bias=reward_bias,
        reward_neg=reward_neg,
        is_sparse_reward=is_sparse_reward,
    )

    for i, transition in enumerate(trajectory):
        transition['mc_returns'] = mc_returns[i]

    return trajectory


def add_embeddings_to_trajectory(
    trajectory,
    model,
    tasks,
    image_keys=("side_policy_256", "wrist_1"),
):
    """
    undate every transition in the trajectory and add embeddings
    return the updated trajectory
    """
    if len(image_keys) < 2:
        raise ValueError("add_embeddings_to_trajectory requires two image keys")

    primary_key, wrist_key = image_keys[:2]
    example_obs = model.example_batch["observation"]
    primary_hw = tuple(example_obs["image_primary"].shape[2:4])
    wrist_hw = tuple(example_obs["image_wrist"].shape[2:4])
    task_completed_shape = tuple(example_obs["task_completed"].shape)

    def _resize_history(images, target_hw):
        images = np.asarray(images)
        if images.ndim != 4:
            raise ValueError(f"Expected image history with shape (T, H, W, C), got {images.shape}")
        resized = [
            cv2.resize(frame, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_AREA)
            for frame in images
        ]
        return np.stack(resized, axis=0).astype(np.uint8)

    for i in range(len(trajectory)):
        observation = trajectory[i]['observations']

        image_primary = _resize_history(observation[primary_key], primary_hw)
        image_wrist = _resize_history(observation[wrist_key], wrist_hw)
        # Add batch dimension
        image_primary = image_primary[np.newaxis, ...]
        image_wrist = image_wrist[np.newaxis, ...]
        horizon = image_primary.shape[1]
        timestep_pad_mask = np.ones((1, horizon), dtype=bool)
        timestep = np.broadcast_to(np.arange(horizon, dtype=np.int32)[None, :], (1, horizon))
        pad_mask_dict = {
            "image_primary": np.ones((1, horizon), dtype=bool),
            "image_wrist": np.ones((1, horizon), dtype=bool),
            "timestep": np.ones((1, horizon), dtype=bool),
        }
        task_completed = np.zeros((1, horizon, task_completed_shape[-1]), dtype=bool)

        observation = {"image_primary": image_primary,
                       "image_wrist": image_wrist,
                       "pad_mask_dict": pad_mask_dict,
                       "task_completed": task_completed,
                       "timestep": timestep,
                       "timestep_pad_mask": timestep_pad_mask,
                       }

        action_embeddings = model.sample_transformer(observation, tasks,)
        # Now, action_embeddings is (batch_size, window_size, embedding_size)

        # remove window_size dimension
        action_embeddings = action_embeddings[:, -1, :]
        action_embeddings = np.asarray(jax.device_get(action_embeddings), dtype=np.float32)
        if action_embeddings.shape[0] == 1:
            action_embeddings = action_embeddings[0]

        trajectory[i]['embeddings'] = action_embeddings.copy()

    return trajectory


def add_next_embeddings_to_trajectory(trajectory):
    """
    undate every transition in the trajectory and add next_embeddings
    return the updated trajectory
    """
    for i in range(len(trajectory)):
        if i == len(trajectory) - 1:
            trajectory[i]['next_embeddings'] = trajectory[i]['embeddings']
        else:
            trajectory[i]['next_embeddings'] = trajectory[i+1]['embeddings']

    return trajectory
