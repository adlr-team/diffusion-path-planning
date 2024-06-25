import collections
import os
import pdb
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import minari

import gymnasium as gym
import numpy as np
from minari import list_local_datasets, load_dataset



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
        #TODO: I am not sure if this is correct here
        wrapped_env.reset()
    env = wrapped_env.unwrapped
    env.max_episode_steps = wrapped_env._max_episode_steps
    env.name = name
    return env


def get_dataset():
    #dataset_name = "pointmaze-medium-v2" #TODO: change this to a parameter
    dataset_name = "kitchen-complete-v1"
    print(f"Dataset_name:{dataset_name}")


    if dataset_name in list_local_datasets():
        print("dataset exists!")       
    else:
        print("dataset does not exist! downloading it now...")
        minari.download_dataset(dataset_name )

    dataset = load_dataset(dataset_name)

    return dataset



def sequence_dataset(preprocess_fn):
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
    print("here")

    for episode_number in range(dataset._data.total_episodes):
        #episode_data = process_maze2d_episode(dataset[episode_number])
        episode_data = process_franka_episode(dataset[episode_number])
        print(episode_data)
        yield episode_data


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


def process_franka_episode(episode):
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
            ep[key] = val["slide cabinet"][:-1] #TODO: this only looks at the cabinet tasks goal, don't know how to handle other tasks

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
