import multiprocessing
import torch.multiprocessing as mp

mp.set_start_method('spawn', force=True)


import gymnasium as gym
from diffuser.utils.config import Config
import os
import collections
import numpy as np
import pdb
from minari import DataCollector, StepDataCallback
import torch
import torch

print(f"Cuda is available: {torch.cuda.is_available()}")


class Args:
    def __init__(
        self,
        loader,
        savepath,
        dataset,
        horizon,
        normalizer,
        preprocess_fns,
        use_padding,
        max_path_length,
        renderer,
        model,
        dim_mults,
        device,
    ):
        self.loader = loader
        self.savepath = savepath
        self.dataset = dataset
        self.horizon = horizon
        self.normalizer = normalizer
        self.preprocess_fns = preprocess_fns
        self.use_padding = use_padding
        self.max_path_length = max_path_length
        self.renderer = renderer
        self.model = model
        # self.transition_dim=transition_dim
        # self.cond_dim=cond_dim
        self.dim_mults = dim_mults
        self.device = device


args = Args(
    loader="datasets.sequence.GoalDataset",
    savepath="saved_medium/",
    dataset="PointMaze_Medium-v3",
    horizon=256,
    normalizer="LimitsNormalizer",
    preprocess_fns=["maze2d_set_terminals"],
    use_padding=False,
    max_path_length=1000,
    renderer="utils.rendering.Maze2dRenderer",
    model="models.temporal.TemporalUnet",
    dim_mults=(1, 4, 8),
    device="cuda",
)


dataset_config = Config(
    args.loader,
    savepath=(args.savepath, "dataset_config.pkl"),
    env=args.dataset,
    horizon=args.horizon,
    normalizer=args.normalizer,
    preprocess_fns=args.preprocess_fns,
    use_padding=args.use_padding,
    max_path_length=args.max_path_length,
)

dataset = dataset_config()


render_config = Config(
    args.renderer,
    savepath=(args.savepath, "render_config.pkl"),
    env=args.dataset,
)


observation_dim = 4
action_dim = 2
# -----------------------------------------------------------------------------#
# ------------------------------ model & trainer ------------------------------#
# -----------------------------------------------------------------------------#

model_config = Config(
    args.model,
    savepath=(args.savepath, "model_config.pkl"),
    horizon=args.horizon,
    transition_dim=observation_dim + action_dim,
    cond_dim=observation_dim,
    dim_mults=args.dim_mults,
    device=args.device,
)


renderer = render_config()


model = model_config()


diffusion_config = Config(
    _class="models.diffuser.GaussianDiffusion",
    savepath=(args.savepath, "diffusion_config.pkl"),
    horizon=256,
    observation_dim=observation_dim,
    action_dim=2,
    n_timesteps=256,
    loss_type="l2",
    clip_denoised=True,
    predict_epsilon=False,
    # loss weighting
    action_weight=1,
    loss_weights=None,
    loss_discount=1,
    device=args.device,
)


diffuser = diffusion_config(model)

from diffuser.utils.arrays import report_parameters, batchify
import torch

torch.set_default_device('cuda')  # current device is 0

report_parameters(model)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
diffuser.to(device)

print("Testing forward...", end=" ", flush=True)
batch = batchify(dataset[0])
loss, _ = diffuser.loss(*batch)
loss.backward()
print("✓")


from diffuser.utils.training import Trainer

trainer_config = Config(
    Trainer,
    savepath=(args.savepath, "trainer_config.pkl"),
    train_batch_size=32,
    train_lr=2e-4,
    gradient_accumulate_every=2,
    ema_decay=0.005,
    sample_freq=1000,
    save_freq=5000,
    label_freq=int(2e4 // 50),
    save_parallel=False,
    results_folder=args.savepath,
    bucket=None,
    n_reference=1,
    n_samples=1,
)


trainer = trainer_config(diffuser, dataset, renderer, device)
import wandb
from datetime import datetime
use_wandb = True

current_time = datetime.now().strftime("%d_%m_%Y-%H-%M")

if use_wandb:
    run = wandb.init(
        config=trainer_config.to_dict(),
        project="Experiment-1",
        name=f"Name-1_{current_time}",
        group="Group-Name",
        job_type="training",
        reinit=True,
    )


# n_epochs = int(args.n_train_steps // args.n_steps_per_epoch)
n_epochs = 70
for i in range(n_epochs):
    print(f"Epoch {i} / {n_epochs} | {args.savepath}")
    trainer.train(n_train_steps=1000)