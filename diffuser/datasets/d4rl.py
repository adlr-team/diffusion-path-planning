import collections
import os
import pdb
from contextlib import contextmanager, redirect_stderr, redirect_stdout

import gymnasium as gym
import numpy as np
from minari import DataCollector, list_local_datasets, load_dataset

from .d4rl_adapter import (PointMazeStepDataCallback, QIteration,
                           WaypointController)


@contextmanager
def suppress_output():
    """
    A context manager that redirects stdout and stderr to devnull
    https://stackoverflow.com/a/52442331
    """
    with open(os.devnull, "w") as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)


# with suppress_output():
#     ## d4rl prints out a variety of warnings

# -----------------------------------------------------------------------------#
# -------------------------------- general api --------------------------------#
# -----------------------------------------------------------------------------#


def load_environment(name):
    print(f"Name of the enviornment: {name}")
    if type(name) != str:
        ## name is already an environment
        return name
    with suppress_output():
        wrapped_env = gym.make(name)
    env = wrapped_env.unwrapped
    env.max_episode_steps = wrapped_env._max_episode_steps
    env.name = name
    return env


def get_dataset():
    
    ##Code from Minari documentation
    dataset_name = "pointmaze-umaze-v0"
    print(f"Dataset_name:{dataset_name}")
    # Define steps
    total_steps = 100000

    # continuing task => the episode doesn't terminate or truncate when reaching a goal
    # it will generate a new target. For this reason we set the maximum episode steps to
    # the desired size of our Minari dataset (evade truncation due to time limit)
    # Continuing task will always generate a new goal if the previous one is reached
    env = gym.make(
        "PointMaze_Medium-v3", continuing_task=True, max_episode_steps=total_steps
    )

    # Load exisiting datasets:
    if dataset_name in list_local_datasets():
        print("dataset exists!")
        dataset = load_dataset(dataset_name)
    else:
        dataset = None

        # Data collector wrapper to save temporary data while stepping. Characteristics:
        #   * Custom StepDataCallback to add extra state information to 'infos' and divide dataset in different episodes by overridng
        #     truncation value to True when target is reached
        #   * Record the 'info' value of every step
        collector_env = DataCollector(
            env, step_data_callback=PointMazeStepDataCallback, record_infos=True
        )

        obs, _ = collector_env.reset(seed=123)

        waypoint_controller = WaypointController(maze=env.maze)

        for n_step in range(int(total_steps)):
            action = waypoint_controller.compute_action(obs)
            # Add some noise to each step action
            action += np.random.randn(*action.shape) * 0.5
            action = np.clip(
                action, env.action_space.low, env.action_space.high, dtype=np.float32
            )

            obs, rew, terminated, truncated, info = collector_env.step(action)

        dataset = collector_env.create_dataset(
            dataset_id=dataset_name,
            algorithm_name="QIteration",
            code_permalink="https://github.com/Farama-Foundation/Minari/blob/main/docs/tutorials/dataset_creation/point_maze_dataset.py",
            author="Rodrigo Perez-Vicente",
            author_email="rperezvicente@farama.org",
        )

    return dataset


# def sequence_dataset(dataset, env):
#     #N = dataset['rewards'].shape[0]
#     N = dataset._data.total_episodes
#     data_ = collections.defaultdict(list)

#     # The newer version of the dataset adds an explicit
#     # timeouts field. Keep old method for backwards compatability.
#     use_timeouts = 'timeouts' in dataset

#     episode_step = 0
#     for i in range(N):
#         done_bool = bool(dataset['terminals'][i])
#         if use_timeouts:
#             final_timestep = dataset['timeouts'][i]
#         else:
#             final_timestep = (episode_step == env._max_episode_steps - 1)

#         for k in dataset:
#             if 'metadata' in k: continue
#             data_[k].append(dataset[k][i])

#         if done_bool or final_timestep:
#             episode_step = 0
#             episode_data = {}
#             for k in data_:
#                 episode_data[k] = np.array(data_[k])
#             if 'maze2d' in env.name:
#                 episode_data = process_maze2d_episode(episode_data)
#             yield episode_data
#             data_ = collections.defaultdict(list)

#         episode_step += 1


# def sequence_dataset(env, preprocess_fn):
#     """
#     Returns an iterator through trajectories.
#     Args:
#         env: An OfflineEnv object.
#         dataset: An optional dataset to pass in for processing. If None,
#             the dataset will default to env.get_dataset()
#         **kwargs: Arguments to pass to env.get_dataset().
#     Returns:
#         An iterator through dictionaries with keys:
#             observations
#             actions
#             rewards
#             terminals
#     """
#     dataset = get_dataset()
#     dataset = preprocess_fn(dataset)

#     #N = dataset['rewards'].shape[0]
#     N = dataset._data.total_episodes
#     data_ = collections.defaultdict(list)

#     # The newer version of the dataset adds an explicit
#     # timeouts field. Keep old method for backwards compatability.
#     use_timeouts = 'timeouts' in dataset

#     episode_step = 0
#     for i in range(N):
#         done_bool = bool(dataset['terminals'][i])
#         if use_timeouts:
#             final_timestep = dataset['timeouts'][i]
#         else:
#             final_timestep = (episode_step == env._max_episode_steps - 1)

#         for k in dataset:
#             if 'metadata' in k: continue
#             data_[k].append(dataset[k][i])

#         if done_bool or final_timestep:
#             episode_step = 0
#             episode_data = {}
#             for k in data_:
#                 episode_data[k] = np.array(data_[k])
#             if 'maze2d' in env.name:
#                 episode_data = process_maze2d_episode(episode_data)
#             yield episode_data
#             data_ = collections.defaultdict(list)

#         episode_step += 1


# #-----------------------------------------------------------------------------#
# #-------------------------------- maze2d fixes -------------------------------#
# #-----------------------------------------------------------------------------#

# def process_maze2d_episode(episode):
#     '''
#         adds in `next_observations` field to episode
#     '''
#     assert 'next_observations' not in episode
#     length = len(episode['observations'])
#     next_observations = episode['observations'][1:].copy()
#     for key, val in episode.items():
#         episode[key] = val[:-1]
#     episode['next_observations'] = next_observations
#     return episode


def sequence_dataset(env, preprocess_fn):
    """
    Returns an iterator through trajectories.
    Args:
        env: An OfflineEnv object.
        dataset: An optional dataset to pass in for processing. If None,
            the dataset will default to env.get_dataset()
        **kwargs: Arguments to pass to env.get_dataset().
    Returns:
        An iterator through dictionaries with keys:
            observations
            actions
            rewards
            terminals
    """
    dataset = get_dataset()
    dataset = preprocess_fn(dataset)

    N = dataset[0].rewards.shape[0]
    # N = dataset._data.total_episodes
    data_ = collections.defaultdict(list)

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = False
    print(N)
    episode_step = 0

    for episode_number in range(dataset._data.total_episodes):
        episode_data = process_maze2d_episode(dataset[episode_number])
        yield episode_data

    # for attr, value in vars(dataset[episode_step]).items():
    #     #if 'metadata' in k: continue
    #     #data_[k].append(dataset[k][i])
    #     data_[attr].append(value)

    # episode_data = {}
    # for k in data_:
    #     episode_data[k] = np.array(data_[k])

    # for i in range(N):
    #     #print(dataset[i])
    #     #dataset[i].observations
    #     # done_bool = bool(dataset['terminals'][i])
    #     done_bool = bool(dataset[episode_step].terminations[i])

    #     final_timestep = (episode_step == env.max_episode_steps - 1)

    #     #for k in dataset:

    #     if done_bool or final_timestep:
    #         print("heree")
    #         print(i)
    #         episode_step = 0

    #         episode_data = process_maze2d_episode(dataset[episode_step])
    #         yield episode_data
    #         data_ = collections.defaultdict(list)
    #         episode_data = {}

    #         episode_step += 1


# def process_maze2d_episode(episode):
#     '''
#         adds in `next_observations` field to episode
#     '''
#     #assert 'next_observations' not in episode
#     ep = {}
#     length = len(episode.observations)
#     next_observations = episode.observations["observation"][1:].copy()
#     for key, val in episode.observations.items():
#         ep[key] = val[:-1]
#     ep['next_observations'] = next_observations
#     return ep


def process_maze2d_episode(episode):
    """
    adds in `next_observations` field to episode
    """
    # assert 'next_observations' not in episode
    ep = {}
    length = len(episode.observations)
    next_observations = episode.observations["observation"][1:].copy()
    for key, val in episode.observations.items():
        if key == "observation":
            ep["observations"] = val[:-1]
            ep["observation"] = val[:-1]
        else:
            ep[key] = val[:-1]
    ep["next_observations"] = next_observations
    for attr, value in vars(episode).items():
        if attr == "observations":
            continue
        elif attr == "terminations":
            ep["terminals"] = value
        elif attr == "truncations":
            ep["timeouts"] = value
        elif attr == "actions":
            ep["actions"] = value
        elif attr == "id" or attr == "seed" or attr == "total_timesteps" or "infos":
            # do nothing
            pass
        else:
            ep[attr] = value

    return ep
