import copy
import os
import pdb

import einops
import numpy as np
import torch

import wandb
from diffuser.datasets.d4rl import load_environment

from .arrays import apply_dict, batch_to_device, to_device, to_np
from .cloud import sync_logs
from .timer import Timer


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


class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        dataset,
        renderer,
        device,
        name,
        ema_decay=0.995,
        train_batch_size=32,
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
        self.dataset = dataset
        self.dataloader = cycle(
            torch.utils.data.DataLoader(
                self.dataset,
                batch_size=train_batch_size,
                num_workers=8,
                shuffle=True,
                pin_memory=True,
                generator=torch.Generator(device="cpu"),
            )
        )
        self.dataloader_vis = cycle(
            torch.utils.data.DataLoader(
                self.dataset,
                batch_size=1,
                num_workers=8,
                shuffle=True,
                pin_memory=True,
                generator=torch.Generator(device="cpu"),
            )
        )
        self.renderer = renderer

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
        self.ema_model.to("cuda")
        self.ema.update_model_average(self.ema_model, self.model)

    # -----------------------------------------------------------------------------#
    # ------------------------------------ api ------------------------------------#
    # -----------------------------------------------------------------------------#

    def train(self, n_train_steps):

        timer = Timer()
        for step in range(n_train_steps):
            wandb.log({"Training_steps": step})
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

            # if self.step % self.save_freq == 0:
            #     label = self.step
            #     self.save(label)

            if self.step % self.log_freq == 0:
                infos_str = " | ".join(
                    [f"{key}: {val:8.4f}" for key, val in infos.items()]
                )
                wandb.log({"Model Loss": loss})
                print(f"{self.step}: {loss:8.4f} | {infos_str} | t: {timer():8.4f}")

            if (
                self.sample_freq
                and self.step % self.sample_freq == 0
                and self.step % 1000 == 0
            ):
                print(f"Step: {self.step} - Rendering samples")
                self.render_samples(
                    self.renderer.env, n_samples=self.n_samples, get_cond_from_env=False
                )

            self.step += 1
        wandb.log({"Time per episode": timer()})

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

    def render_reference(self, batch):
        """
        renders training points
        """

        # ## get a temporary dataloader to load a single batch
        # dataloader_tmp = cycle(
        #     torch.utils.data.DataLoader(
        #         self.dataset,
        #         batch_size=batch_size,
        #         num_workers=0,
        #         shuffle=True,
        #         pin_memory=False,
        #         generator=torch.Generator(device="cpu"),
        #     )
        # )
        # batch = dataloader_tmp.__next__()
        # dataloader_tmp.close()

        ## get trajectories and condition at t=0 from batch
        trajectories = to_np(batch.trajectories)
        conditions = to_np(batch.conditions[0])[:, None]

        ## [ batch_size x horizon x observation_dim ]
        normed_observations = trajectories[:, :, self.dataset.action_dim :]
        observations = self.dataset.normalizer.unnormalize(
            normed_observations, "observations"
        )

        savepath = os.path.join(self.logdir, f"_sample-reference.png")
        collision_count, collision_obs = self.count_and_get_collisions(observations)

        print(f"Collision_count_reference  : {collision_count}")

        self.renderer.composite(savepath, observations, env=self.renderer.env)
        # Renmder collisions:
        if collision_count > 0:
            print(f"Collision observations:")
            self.renderer.composite(savepath, collision_obs, env=self.renderer.env)

    def count_and_get_collisions(self, observations):

        # Map from Cartesian coordinates to maze indices
        def map_to_maze_indices(x, y, rows, cols):
            maze_x = int((x + cols / 2) // 1)
            maze_y = int((rows / 2 - y) // 1)
            return maze_y, maze_x

        maze = np.array(self.renderer.env.maze.maze_map)

        # Check collisions
        collisions = []
        rows, cols = maze.shape

        for point in observations[0]:
            x, y = point[0], point[1]
            maze_y, maze_x = map_to_maze_indices(x, y, rows, cols)
            if 0 <= maze_x < cols and 0 <= maze_y < rows:
                if maze[maze_y, maze_x] == 1:
                    collisions.append(point)

        # Convert collisions list to numpy array in the specified format
        if collisions:
            collision_array = np.array(collisions).reshape(1, len(collisions), 4)
        else:
            collision_array = np.array(collisions).reshape(1, 0, 4)

        # Return the number of collisions and the collision array
        return len(collisions), collision_array

    def render_samples(
        self,
        env,
        batch_size=1,
        n_samples=1,
        get_cond_from_env=False,
        render_reference=False,
    ):
        """
        renders samples from (ema) diffusion model
        Note: set get_cond_from_env to True if you want to get the conditions from the environment
        """
        for i in range(batch_size):

            # print(batch.conditions)
            # {0: tensor([[-0.5193,  0.4605, -0.1075, -0.0416]]), 255: tensor([[ 0.1836, -0.5136,  0.0587,  0.5159]])}

            ## get a single datapoint
            if get_cond_from_env:
                zeros = torch.zeros((1, 2))
                conditions = {}
                conditions[255] = torch.cat(
                    (torch.tensor(env.unwrapped.goal).reshape(1, -1), zeros), dim=0
                ).reshape(1, -1)

                conditions[0] = torch.cat(
                    (
                        torch.tensor(env.unwrapped.point_env.init_qpos[:2]).reshape(
                            1, -1
                        ),
                        zeros,
                    ),
                    dim=0,
                ).reshape(1, -1)

                # We have to normalize the conditions before passign them to the model.
                conditions[0] = self.dataset.normalizer.normalize(
                    conditions[0], "observations"
                )
                conditions[255] = self.dataset.normalizer.normalize(
                    conditions[255], "observations"
                )
                # Is this wrong?
                normed_c = conditions[0]
                conditions = to_device(conditions, self.device)

            else:
                batch = self.dataloader_vis.__next__()
                conditions = to_device(batch.conditions, self.device)
                normed_c = batch.conditions[0]
                render_reference = True

            ## repeat each item in conditions `n_samples` times
            conditions = apply_dict(
                einops.repeat,
                conditions,
                "b d -> (repeat b) d",
                repeat=n_samples,
            )

            ## [ n_samples x horizon x (action_dim + observation_dim) ]
            samples = self.ema_model.conditional_sample(conditions)
            samples = to_np(samples)

            ## [ n_samples x horizon x observation_dim ]
            normed_observations = samples[:, :, self.dataset.action_dim :]

            # [ 1 x 1 x observation_dim ]
            normed_conditions = to_np(normed_c)[:, None]

            # from diffusion.datasets.preprocessing import blocks_cumsum_quat
            # observations = conditions + blocks_cumsum_quat(deltas)
            # observations = conditions + deltas.cumsum(axis=1)

            ## [ n_samples x (horizon + 1) x observation_dim ]
            normed_observations = np.concatenate(
                [np.repeat(normed_conditions, n_samples, axis=0), normed_observations],
                axis=1,
            )

            ## [ n_samples x (horizon + 1) x observation_dim ]
            observations = self.dataset.normalizer.unnormalize(
                normed_observations, "observations"
            )

            #### @TODO: remove block-stacking specific stuff
            # from diffusion.datasets.preprocessing import blocks_euler_to_quat, blocks_add_kuka
            # observations = blocks_add_kuka(observations)
            ####
            savepath = os.path.join(self.logdir, f"sample-{self.step}-{i}.png")
            if render_reference:
                print("The trajectory from the dataset:")

                self.render_reference(batch)

            print("The trajectory generated by the diffusion model:")
            collision_count, collisions_obs = self.count_and_get_collisions(
                observations
            )
            print(f"Collision_count_diffusion_model : {collision_count}")
            wandb.log({"Collision count:": collision_count})

            self.renderer.composite(savepath, observations, env=env)
            # print(f"Collision observations: {collisions_obs}")
            if collision_count > 0:
                print(f"Collision observations:")
                self.renderer.composite(savepath, collisions_obs, env=env)
