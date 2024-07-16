import copy
import os
import pdb

import einops
import numpy as np
import torch
from rokin import robots, vis

import wandb
from diffuser.datasets.d4rl import load_environment
from diffuser.datasets.normalization import LimitsNormalizer
from diffuser.utils.arrays import apply_dict, batch_to_device, to_device, to_np
from diffuser.utils.cloud import sync_logs
from diffuser.utils.timer import Timer
from justin_arm.helper import analyze_distance, robot_env_dist
from justin_arm.visualize import (
    plot_q_values_per_trajectory,
    plot_trajectory_per_frames,
)


def cycle(dl):
    while True:
        for data in dl:
            yield data


class EMA:
    """
    empirical moving average
    """

    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(
            current_model.parameters(), ma_model.parameters()
        ):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class Justin_Trainer(object):
    def __init__(
        self,
        diffusion_model,
        dataset,
        val_dataset,
        device,
        robot,
        name,
        ema_decay=0.995,
        train_batch_size=1,
        train_lr=2e-5,
        gradient_accumulate_every=2,
        step_start_ema=2000,
        update_ema_every=10,
        log_freq=100,
        sample_freq=1000,
        save_freq=1000,
        label_freq=100000,
        save_parallel=False,
        results_folder="./results",
        n_reference=1,
        n_samples=2,
        bucket=None,
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every
        self.name = name
        self.step_start_ema = step_start_ema
        self.log_freq = log_freq
        self.sample_freq = sample_freq
        self.save_freq = save_freq
        self.label_freq = label_freq
        self.save_parallel = save_parallel

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.device = device
        self.robot = robot
        self.dataset = dataset
        self.val_dataset = val_dataset
        self.dataloader = cycle(
            torch.utils.data.DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                num_workers=0,
                shuffle=True,
                pin_memory=True,
                generator=torch.Generator(device="cpu"),
            )
        )
        self.val_dataloader = cycle(
            torch.utils.data.DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                num_workers=0,
                shuffle=True,
                pin_memory=True,
                generator=torch.Generator(device="cpu"),
            )
        )
        self.dataloader_vis = cycle(
            torch.utils.data.DataLoader(
                self.dataset,
                batch_size=1,
                num_workers=0,
                shuffle=True,
                pin_memory=True,
                generator=torch.Generator(device="cpu"),
            )
        )
        # self.renderer = renderer

        self.optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=train_lr)

        self.logdir = results_folder
        self.bucket = bucket
        self.n_reference = n_reference
        self.n_samples = n_samples

        self.reset_parameters()
        self.step = 0

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema_model.to(self.device)
        self.ema.update_model_average(self.ema_model, self.model)

    # -----------------------------------------------------------------------------#
    # ------------------------------------ api ------------------------------------#
    # -----------------------------------------------------------------------------#

    def train_single_datapoint(self, n_train_steps, single_input):
        timer = Timer()
        for step in range(n_train_steps):
            wandb.log({"Training_steps": step})

            # No need for gradient accumulation when overfitting a single datapoint
            for i in range(self.gradient_accumulate_every):
                # Use the single datapoint
                loss, infos = self.model.loss(*single_input)
                loss = loss / self.gradient_accumulate_every

                loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            # if self.step % self.update_ema_every == 0:
            #     self.step_ema()

            # Log
            if self.step % self.log_freq == 0:
                infos_str = " | ".join(
                    [f"{key}: {val:8.4f}" for key, val in infos.items()]
                )
                wandb.log({"Model Loss": loss})
                print(f"{self.step}: {loss:8.4f} | {infos_str} | t: {timer():8.4f}")

            self.step += 1
        wandb.log({"Time per episode": timer()})

        # Save model
        if self.step % self.save_freq == 0:
            self.save(self.step)

    def train(self, n_train_steps):
        timer = Timer()
        best_val_loss = float("inf")
        patience = 2
        epochs_no_improve = 0

        for step in range(n_train_steps):
            # Training step
            self.model.train()
            for i in range(self.gradient_accumulate_every):
                batch = next(self.dataloader)
                batch = batch_to_device(batch, self.device)

                loss, infos = self.model.loss(*batch)
                loss = loss / self.gradient_accumulate_every

                loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()
            # Log
            if self.step % self.log_freq == 0:
                infos_str = " | ".join(
                    [f"{key}: {val:8.4f}" for key, val in infos.items()]
                )
                wandb.log({"Model Loss per step": loss.item()})
                print(
                    f"{self.step}: {loss.item():8.4f} | {infos_str} | t: {timer():8.4f}"
                )

                self.step += 1

        self.model.eval()
        val_loss = 0.0
        val_infos = {}
        with torch.no_grad():
            # Get only one batch from the validation dataloader
            val_batch = next(self.val_dataloader)
            val_batch = batch_to_device(val_batch, self.device)
            val_batch_loss, val_batch_infos = self.model.loss(*val_batch)
            val_loss = val_batch_loss.item()

            for key, val in val_batch_infos.items():
                val_infos[key] = val

        val_infos_str = " | ".join(
            [f"{key}: {val:8.4f}" for key, val in val_infos.items()]
        )
        wandb.log({"Validation Loss": val_loss})
        print(f"Validation {self.step}: {val_loss:8.4f} | {val_infos_str}")

        # Early stopping based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            print(f"Validation loss improved, saving model at step {self.step}")
            self.save(self.step)
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {self.step} steps")

        # Render samples at the specified frequency
        print(f"Rendering samples")
        collision_score = self.render_sample()
        wandb.log({"Collision Score": collision_score})
        wandb.log({"Time per episode": timer()})
        wandb.log({"Training_steps": self.step})

    def save(self, epoch):
        """
        saves model and ema to disk;
        syncs to storage bucket if a bucket is specified
        """
        data = {
            "step": self.step,
            "model": self.model.state_dict(),
            "ema": self.ema_model.state_dict(),
        }
        savepath = os.path.join(self.logdir, f"state_{epoch}.pt")
        torch.save(data, savepath)
        print(f"[ utils/training ] Saved model to {savepath}")
        if self.bucket is not None:
            sync_logs(self.logdir, bucket=self.bucket, background=self.save_parallel)

    def load(self, epoch, directory=None):
        """
        loads model and ema from disk
        """
        if directory is not None:
            direc = directory
        else:
            direc = os.path.join(direc, f"state_{epoch}.pt")

        loadpath = direc
        data = torch.load(loadpath, map_location=torch.device("cpu"))

        self.step = data["step"]
        self.model.load_state_dict(data["model"])
        self.ema_model.load_state_dict(data["ema"])

    # -----------------------------------------------------------------------------#
    # --------------------------------- rendering ---------------------------------#
    # -----------------------------------------------------------------------------#

    def render_sample(
        self,
        n_samples=1,
        render_3d=False,
    ):
        """
        renders samples from (ema) diffusion model
        Note: set get_cond_from_env to True if you want to get the conditions from the environment
        """

        batch = self.dataloader_vis.__next__()
        conditions = to_device(batch.conditions, self.device)
        normed_c = batch.conditions[0]

        ## repeat each item in conditions `n_samples` times
        conditions = apply_dict(
            einops.repeat,
            conditions,
            "b d -> (repeat b) d",
            repeat=n_samples,
        )

        ## [ n_samples x horizon x (action_dim + observation_dim) ]
        samples = self.model.conditional_sample(conditions)
        samples = to_np(samples)

        ## [ n_samples x horizon x observation_dim ]
        normed_observations = samples[:, :, self.dataset.action_dim[0] :]
        # print(f"Before unnormalizing: {normed_observations.shape}")
        # [ 1 x 1 x observation_dim ]+
        normed_conditions = to_np(normed_c)[:, None]

        # ## [ n_samples x (horizon + 1) x observation_dim ]
        # normed_observations = np.concatenate(
        #     [np.repeat(normed_conditions, n_samples, axis=0), normed_observations],
        #     axis=1,
        # )
        ## [ n_samples x (horizon + 1) x observation_dim ]
        observations = self.dataset.normalizer.unnormalize(normed_observations[0])
        # print(f"After unnormalizing: {observations.shape}")

        # Get collision_metric:
        distance = robot_env_dist(
            q=observations, robot=self.robot, img=self.dataset.image[0]
        )
        # Get collision score
        collision_score = analyze_distance(distance)

        # Plot original_reference_data:
        self.plot_reference_data(batch)

        print("Diffused trajectory:")
        print(f"Collision score: {collision_score}")
        plot_trajectory_per_frames(
            observations, savepath=self.logdir, name=str(self.step)
        )
        plot_q_values_per_trajectory(
            observations, savepath=self.logdir, name=str(self.step)
        )

        # Plot diffused

        # Render 3D
        if render_3d:
            limits = np.array([[-1.25, +1.25], [-1.25, +1.25], [-1.25, +1.25]])
            vis.three_pv.animate_path(
                robot=self.robot,
                q=observations,
                kwargs_robot=dict(color="red"),
                kwargs_world=dict(
                    img=self.dataset.image[0], limits=limits, color="yellow"
                ),
            )

        return collision_score

    def plot_reference_data(self, batch, render_3d=False):
        """
        Plots the reference data for the batch
        """
        trajectory = batch.trajectories.numpy()

        ## [ n_samples x horizon x observation_dim ]
        normed_observations = trajectory[:, :, : self.dataset.action_dim[0]]

        ## [ n_samples x (horizon + 1) x observation_dim ]
        observations = self.dataset.normalizer.unnormalize(normed_observations[0])
        # print(f"After unnormalizing: {observations.shape}")

        # Get collision_metric:
        distance = robot_env_dist(
            q=observations, robot=self.robot, img=self.dataset.image[0]
        )

        score = analyze_distance(distance)

        print("Reference trajectory:")
        print(f"Collision score: {score}")
        plot_trajectory_per_frames(
            observations, savepath=self.logdir, name="reference_" + str(self.step)
        )
        plot_q_values_per_trajectory(
            observations, savepath=self.logdir, name="_reference_" + str(self.step)
        )

        # Render 3D
        if render_3d:
            limits = np.array([[-1.25, +1.25], [-1.25, +1.25], [-1.25, +1.25]])
            vis.three_pv.animate_path(
                robot=self.robot,
                q=observations,
                kwargs_robot=dict(color="red"),
                kwargs_world=dict(
                    img=self.dataset.image[0], limits=limits, color="yellow"
                ),
            )
        return score

    def render_given_sample(self, given_sample, n_samples=1, render_3d=False):
        """
        renders samples from (ema) diffusion model
        Note: set get_cond_from_env to True if you want to get the conditions from the environment
        """

        conditions = to_device(given_sample.conditions, self.device)
        normed_c = given_sample.conditions[0]

        ## repeat each item in conditions `n_samples` times
        conditions = apply_dict(
            einops.repeat,
            conditions,
            "b d -> (repeat b) d",
            repeat=n_samples,
        )

        ## [ n_samples x horizon x (action_dim + observation_dim) ]
        samples = self.model.conditional_sample(conditions)
        samples = to_np(samples)

        ## [ n_samples x horizon x observation_dim ]
        normed_observations = samples[:, :, : self.dataset.action_dim[0]]
        # print(f"Before unnormalizing: {normed_observations.shape}")
        # [ 1 x 1 x observation_dim ]
        normed_conditions = to_np(normed_c)[:, None]

        ## [ n_samples x (horizon + 1) x observation_dim ]
        observations = self.dataset.normalizer.unnormalize(normed_observations[0])
        # print(f"After unnormalizing: {observations.shape}")

        # Get collision_metric:
        distance = robot_env_dist(
            q=observations, robot=self.robot, img=self.dataset.image[0]
        )
        # Get collision score
        score = analyze_distance(distance)

        # Plot trajectories for all 8 frames
        plot_trajectory_per_frames(observations)
        # Plot Q values for each joint
        plot_q_values_per_trajectory(observations)

        # Render trajecotry in 3d space
        if render_3d:
            limits = np.array([[-1.25, +1.25], [-1.25, +1.25], [-1.25, +1.25]])
            vis.three_pv.animate_path(
                robot=self.robot,
                q=observations,
                kwargs_robot=dict(color="red"),
                kwargs_world=dict(
                    img=self.dataset.image[0], limits=limits, color="yellow"
                ),
            )
        return observations, score
