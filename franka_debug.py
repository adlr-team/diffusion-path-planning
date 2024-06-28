import gymnasium as gym
from diffuser.utils.config import Config, get_params, get_device_settings
import os
import collections
import numpy as np
import pdb
from minari import DataCollector, StepDataCallback
import wandb
import torch
from datetime import datetime
from tqdm import tqdm



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
    savepath="saved/",
    dataset="",
    horizon=256,
    normalizer="LimitsNormalizer",
    preprocess_fns=["maze2d_set_terminals"],
    use_padding=True,
    max_path_length=280,  #from the dataset description
    renderer="utils.rendering.FrankaRenderer",
    model="models.temporal.TemporalUnet",
    dim_mults=(1, 4, 8),
    device="cpu",
)


dataset_config = Config(
    args.loader,
    savepath=(args.savepath, "dataset_config.pkl"),
    env="FrankaKitchen-v1",
    horizon=args.horizon,
    normalizer=args.normalizer,
    preprocess_fns=args.preprocess_fns,
    use_padding=args.use_padding,
    max_path_length=args.max_path_length,
)


dataset = dataset_config()


print(dataset[0])