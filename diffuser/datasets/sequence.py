import pdb
from collections import namedtuple
from multiprocessing import Pool

import numpy as np
import torch

from justin_arm.helper import (
    condition_start_end_per_trajectory,
    create_state_action_array,
    interpolate_trajectories,
)

from .buffer import ReplayBuffer
from .d4rl import load_environment, sequence_dataset
from .normalization import DatasetNormalizer, LimitsNormalizer
from .preprocessing import get_preprocess_fn

Batch = namedtuple("Batch", "trajectories conditions")
ValueBatch = namedtuple("ValueBatch", "trajectories conditions values")


class SequenceDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        env="hopper-medium-replay",
        horizon=64,
        normalizer="LimitsNormalizer",
        preprocess_fns=[],
        max_path_length=1000,
        max_n_episodes=10000,
        termination_penalty=0,
        use_padding=True,
        seed=None,
    ):
        self.preprocess_fn = get_preprocess_fn(preprocess_fns, env)
        self.env = env = load_environment(env)
        # self.env.seed(seed)
        self.horizon = horizon
        self.max_path_length = max_path_length
        self.use_padding = use_padding
        itr = sequence_dataset(env, self.preprocess_fn)

        fields = ReplayBuffer(max_n_episodes, max_path_length, termination_penalty)
        for i, episode in enumerate(itr):
            fields.add_path(episode)
        fields.finalize()

        self.normalizer = DatasetNormalizer(
            fields, normalizer, path_lengths=fields["path_lengths"]
        )
        self.indices = self.make_indices(fields.path_lengths, horizon)

        self.observation_dim = fields.observations.shape[-1]
        # self.action_dim = fields.actions.shape[-1]
        self.action_dim = 2
        self.fields = fields
        self.n_episodes = fields.n_episodes
        self.path_lengths = fields.path_lengths
        self.normalize()

        print(fields)
        # shapes = {key: val.shape for key, val in self.fields.items()}
        # print(f'[ datasets/mujoco ] Dataset fields: {shapes}')

    def normalize(self, keys=["observations", "actions"]):
        """
        normalize fields that will be predicted by the diffusion model
        """
        # for key in keys:
        #     array = self.fields[key].reshape(self.n_episodes*self.max_path_length, -1)
        #     normed = self.normalizer(array, key)
        #     self.fields[f'normed_{key}'] = normed.reshape(self.n_episodes, self.max_path_length, -1)
        print(self.fields)

        array = self.fields.observations.reshape(
            self.n_episodes * self.max_path_length, -1
        )
        normed = self.normalizer(array, "observations")
        self.fields[f"normed_observations"] = normed.reshape(
            self.n_episodes, self.max_path_length, -1
        )

        array = self.fields.actions.reshape(self.n_episodes * self.max_path_length, -1)
        normed = self.normalizer(array, "actions")
        self.fields[f"normed_actions"] = normed.reshape(
            self.n_episodes, self.max_path_length, -1
        )

    def make_indices(self, path_lengths, horizon):
        """
        makes indices for sampling from dataset;
        each index maps to a datapoint
        """
        indices = []
        for i, path_length in enumerate(path_lengths):
            max_start = min(path_length - 1, self.max_path_length - horizon)
            if not self.use_padding:
                max_start = min(max_start, path_length - horizon)
            for start in range(max_start):
                end = start + horizon
                indices.append((i, start, end))
        indices = np.array(indices)
        return indices

    def get_conditions(self, observations):
        """
        condition on current observation for planning
        """
        return {0: observations[0]}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.indices[idx]

        observations = self.fields.normed_observations[path_ind, start:end]
        actions = self.fields.normed_actions[path_ind, start:end]
        conditions = self.get_conditions(observations)
        trajectories = np.concatenate([actions, observations], axis=-1)
        batch = Batch(trajectories, conditions)
        return batch


class GoalDataset(SequenceDataset):

    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.indices[idx]

        observations = self.fields.normed_observations[path_ind, start:end]
        actions = self.fields.normed_actions[path_ind, start:end]
        conditions = self.get_conditions(observations)
        trajectories = np.concatenate([actions, observations], axis=-1)
        batch = Batch(trajectories, conditions)
        return batch

    def get_conditions(self, observations):
        """
        condition on both the current observation and the last observation in the plan
        """
        return {
            0: observations[0],
            self.horizon - 1: observations[-1],
        }


class ValueDataset(SequenceDataset):
    """
    adds a value field to the datapoints for training the value function
    """

    def __init__(self, *args, discount=0.99, normed=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.discount = discount
        self.discounts = self.discount ** np.arange(self.max_path_length)[:, None]
        self.normed = False
        if normed:
            self.vmin, self.vmax = self._get_bounds()
            self.normed = True

    def _get_bounds(self):
        print(
            "[ datasets/sequence ] Getting value dataset bounds...", end=" ", flush=True
        )
        vmin = np.inf
        vmax = -np.inf
        for i in range(len(self.indices)):
            value = self.__getitem__(i).values.item()
            vmin = min(value, vmin)
            vmax = max(value, vmax)
        print("✓")
        return vmin, vmax

    def normalize_value(self, value):
        ## [0, 1]
        normed = (value - self.vmin) / (self.vmax - self.vmin)
        ## [-1, 1]
        normed = normed * 2 - 1
        return normed

    def __getitem__(self, idx):
        batch = super().__getitem__(idx)
        path_ind, start, end = self.indices[idx]
        rewards = self.fields["rewards"][path_ind, start:]
        discounts = self.discounts[: len(rewards)]
        value = (discounts * rewards).sum()
        if self.normed:
            value = self.normalize_value(value)
        value = np.array([value], dtype=np.float32)
        value_batch = ValueBatch(*batch, value)
        return value_batch


def process_batch(batch, horizon):
    interpolated_data = interpolate_trajectories(batch, horizon)
    print("Interpolated_data shape: ", interpolated_data.shape)
    normalizer = LimitsNormalizer(interpolated_data)
    normalized_data = normalizer.normalize(interpolated_data)
    print(f"Normalized data shape: {normalized_data.shape}")
    return normalizer.normalize(interpolated_data)


class TrajectoryDataset(torch.utils.data.Dataset):

    def __init__(self, dataset, horizon, image, num_workers=6, batch_size=100000):
        self.horizon = horizon
        self.image = image
        self.batch_size = batch_size

        # Process dataset in batches
        self.dataset = self._batch_process(dataset, num_workers)
        self.action_dim = (7,)
        self.normalizer = LimitsNormalizer(
            interpolate_trajectories(dataset, horizon)
        )  # Store normalizer for later use

    def _batch_process(self, dataset, num_workers):
        # Split dataset into batches
        batches = [
            dataset[i : i + self.batch_size]
            for i in range(0, len(dataset), self.batch_size)
        ]

        # Process batches in parallel
        with Pool(num_workers) as pool:
            normalized_batches = pool.starmap(
                process_batch, [(batch, self.horizon) for batch in batches]
            )

        # Concatenate normalized batches
        return np.concatenate(normalized_batches, axis=0)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        trajectory = self.dataset[idx]
        conditions = condition_start_end_per_trajectory(trajectory)
        stacked_array, actions = create_state_action_array(trajectory)
        batch = Batch(stacked_array, conditions)
        return batch
