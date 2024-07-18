# %%

import gymnasium as gym
from diffuser.utils.config import Config, get_params, get_device_settings
from justin_arm.training_justin_vm import Justin_Trainer

from diffuser.utils.arrays import report_parameters, batchify
from diffuser.datasets.sequence import TrajectoryDataset
import numpy as np
import torch
import sys
import matplotlib.pyplot as plt
from datetime import datetime
import wandb
from tqdm import tqdm

# Render original and diffused trajectories:

from diffuser.utils.arrays import apply_dict, batch_to_device, to_device, to_np

import os

# %% [markdown]
# ## Parse Arguments and Paramters

# %%
# Get settings from the config file

parser = get_params()

# overwrite params for Justin Arm
args = args = parser.parse_args(
    [
        "--action_dim",
        "7",
        "--observation_dim",
        "7",
        "--train_batch_size",
        "32",
        "--savepath",
        "justin_ep10_n10000/",
        "--dataset",
        "new_dataset",
        "--horizon",
        "32",
        "--save_freq",
        "10000",
        "--train_lr",
        "0.0001",
        "--n_timesteps",
        "100",
        "--scenario_name",
        "Full_dataset_justin_ep10_10000",
    ]
)

# Set Seeds
seed = args.seed
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

# Get device settings
device = get_device_settings(args)

# Check if saved path exists else create it :
if not os.path.exists(args.savepath):
    print("Creating directory: ", args.savepath)
    os.makedirs(args.savepath)

# %%
dataset = np.load("q_paths_train_dataset.npy")
dataset_image = np.load("justin_arm/data/image_4123.npy")

validation_dataset = np.load("justin_arm/data/q_paths_6547.npy")
validation_dataset_image = np.load("justin_arm/data/image_6547.npy")


trajectory_dataset = TrajectoryDataset(
    dataset=dataset,
    horizon=args.horizon,
    image=dataset_image,
)

validation_dataset = TrajectoryDataset(
    dataset=validation_dataset,
    horizon=args.horizon,
    image=validation_dataset_image,
)
robot = [1]
model_config = Config(
    args.model,
    savepath=(args.savepath, "model_config.pkl"),
    horizon=args.horizon,
    transition_dim=args.observation_dim + args.action_dim,
    cond_dim=args.observation_dim,
    dim_mults=args.dim_mults,
    device=device,
)
diffusion_config = Config(
    _class="models.diffuser.GaussianDiffusion",
    savepath=(args.savepath, "diffusion_config.pkl"),
    horizon=args.horizon,
    observation_dim=args.observation_dim,
    action_dim=args.action_dim,
    n_timesteps=args.n_timesteps,
    loss_type=args.loss_type,
    clip_denoised=args.clip_denoised,
    predict_epsilon=args.predict_epsilon,
    # loss weighting
    action_weight=args.action_weight,
    loss_weights=args.loss_weights,
    loss_discount=args.loss_discount,
    device=device,
)

trainer_config = Config(
    Justin_Trainer,
    savepath=(args.savepath, "trainer_config.pkl"),
    train_batch_size=args.train_batch_size,
    train_lr=args.train_lr,
    name=args.env_name,
    gradient_accumulate_every=args.gradient_accumulate_every,
    ema_decay=args.ema_decay,
    sample_freq=args.sample_freq,
    save_freq=args.save_freq,
    label_freq=args.label_freq,
    save_parallel=args.save_parallel,
    results_folder=args.savepath,
    bucket=args.bucket,
    n_reference=args.n_reference,
    n_samples=args.n_samples,
    device=device,
)

# %%
# check If cuda avaialble:
if torch.cuda.is_available():
    print("CUDA is available")
    print("Device: ", device)

# %%
# Load objects
model = model_config()
diffuser = diffusion_config(model)
trainer = trainer_config(
    diffuser, trajectory_dataset, validation_dataset, device, robot
)

# %% [markdown]
# ## Forward pass is working

# %%
report_parameters(model)

print("Testing forward...", end=" ", flush=True)
batch = batchify(trajectory_dataset[0])
loss, _ = diffuser.loss(*batch)
loss.backward()
print("✓")

# %% [markdown]
# ## Using the trainer requires taking care of the 'device' in the folders

# %% [markdown]
# # Training process inlcluding rendering

# %%
current_time = datetime.now().strftime("%d_%m_%Y-%H-%M")

if args.use_wandb:
    run = wandb.init(
        config=args,
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=f"{args.scenario_name}_{current_time}",
        group="Group-Name",
        job_type="training",
        reinit=True,
    )

# %% [markdown]
# ## Training

# %%
# n_epochs = int(args.n_train_steps // args.n_steps_per_epoch)
n_epochs = 25
diffuser.to(device)
for i in tqdm(range(n_epochs)):
    print(f"Epoch {i} / {n_epochs} | {args.savepath}")
    trainer.train(n_train_steps=10000)
