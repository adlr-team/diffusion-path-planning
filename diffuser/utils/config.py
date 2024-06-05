import argparse
import collections
import importlib
import os
import pickle
import socket
from pathlib import Path

import numpy as np
import torch

import wandb


def import_class(_class):
    if type(_class) is not str:
        return _class

    print(f"_class:{_class}")
    ## 'diffusion' on standard installs
    repo_name = __name__.split(".")[0]
    ## eg, 'utils'
    module_name = ".".join(_class.split(".")[:-1])
    ## eg, 'Renderer'
    class_name = _class.split(".")[-1]
    ## eg, 'diffusion.utils'
    module = importlib.import_module(f"{repo_name}.{module_name}")
    ## eg, diffusion.utils.Renderer
    _class = getattr(module, class_name)
    print(f"[ utils/config ] Imported {repo_name}.{module_name}:{class_name}")
    return _class


class Config(collections.abc.Mapping):

    def __init__(self, _class, verbose=True, savepath=None, device="cpu", **kwargs):
        self._class = import_class(_class)
        self._device = device
        self._dict = {}

        for key, val in kwargs.items():
            self._dict[key] = val

        if verbose:
            print(self)

        if savepath is not None:
            savepath = os.path.join(*savepath) if type(savepath) is tuple else savepath
            savepath = (
                os.path.join(*savepath) if isinstance(savepath, tuple) else savepath
            )

            pickle.dump(self, open(savepath, "wb"))
            print(f"[ utils/config ] Saved config to: {savepath}\n")

    def __repr__(self):
        string = f"\n[utils/config ] Config: {self._class}\n"
        for key in sorted(self._dict.keys()):
            val = self._dict[key]
            string += f"    {key}: {val}\n"
        return string

    def __iter__(self):
        return iter(self._dict)

    def __getitem__(self, item):
        return self._dict[item]

    def __len__(self):
        return len(self._dict)

    def __getattr__(self, attr):
        if attr == "_dict" and "_dict" not in vars(self):
            self._dict = {}
            return self._dict
        try:
            return self._dict[attr]
        except KeyError:
            raise AttributeError(attr)

    def __call__(self, *args, **kwargs):
        instance = self._class(*args, **kwargs, **self._dict)
        if self._device:
            instance = instance.to(self._device)
        return instance

    def to_dict(self):
        return self.__dict__


def get_params():
    """
    Set up environment and parameters for training and evaluation.

    Returns
    -------
    parser : argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description="Setup configuration parameters for machine learning model training and evaluation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # General parameters for torch:
    parser.add_argument(
        "--seed", type=int, default=5, help="Random seed for numpy/torch"
    )
    parser.add_argument(
        "--n_training_threads",
        type=int,
        default=1,
        help="Number of torch threads for training",
    )

    parser.add_argument(
        "--loader",
        type=str,
        default="datasets.sequence.GoalDataset",
        help="Loader class path",
    )
    parser.add_argument(
        "--savepath", type=str, default="saved/", help="Path to save outputs"
    )
    parser.add_argument("--dataset", type=str, default="", help="Dataset identifier")
    parser.add_argument(
        "--horizon", type=int, default=256, help="Horizon length for episodes"
    )
    parser.add_argument(
        "--normalizer", type=str, default="LimitsNormalizer", help="Normalizer type"
    )
    parser.add_argument(
        "--preprocess_fns",
        nargs="+",
        default=["maze2d_set_terminals"],
        help="List of preprocessing functions",
    )
    parser.add_argument(
        "--use_padding",
        action="store_true",
        default=False,
        help="Whether to use padding",
    )
    parser.add_argument(
        "--max_path_length", type=int, default=10000, help="Maximum path length"
    )
    parser.add_argument(
        "--user_name",
        type=str,
        default="j-falkenstein",
        help="[for wandb usage], to specify user's name for simply collecting training data.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="ADLR",
        help="[for wandb usage], to specify user's name for simply collecting training data.",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default="magic_rabbit",
        help="[for wandb usage], to specify user's name for simply collecting training data.",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_false",
        default=True,
        help="[for wandb usage], by default True, will log date to wandb server. or else will use tensorboard to log data.",
    )
    parser.add_argument(
        "--scenario_name", type=str, default="Scenario", help="Which scenario to run on"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="by default None. set the path to pretrained model.",
    )

    parser.add_argument(
        "--renderer",
        type=str,
        default="utils.rendering.Maze2dRenderer",
        help="Renderer class path",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models.temporal.TemporalUnet",
        help="Model class path",
    )
    parser.add_argument(
        "--dim_mults",
        nargs="+",
        type=int,
        default=(1, 4, 8),
        help="Dimension multipliers for model layers",
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        default=True,
        help="Whether to use padding",
    )

    # Dataset configuration
    parser.add_argument(
        "--env_name",
        type=str,
        default="PointMaze_Medium-v3",
        help="Environment for dataset",
    )

    # Model configurations
    parser.add_argument(
        "--observation_dim",
        type=int,
        default=4,
        help="Dimensionality of transition variables",
    )
    # Model configurations
    parser.add_argument(
        "--action_dim",
        type=int,
        default=2,
        help="Dimensionality of transition variables",
    )

    # Diffusion configuration
    parser.add_argument(
        "--n_timesteps",
        type=int,
        default=256,
        help="Number of timesteps in the diffusion process",
    )
    parser.add_argument(
        "--loss_type", type=str, default="l2", help="Loss type for the diffusion model"
    )
    parser.add_argument(
        "--clip_denoised",
        action="store_true",
        default=True,
        help="Whether to clip denoised data",
    )
    parser.add_argument(
        "--predict_epsilon",
        action="store_false",
        default=False,
        help="Whether to predict epsilon instead of the noise",
    )
    parser.add_argument(
        "--action_weight",
        type=float,
        default=1,
        help="Weight for actions in loss computation",
    )
    parser.add_argument(
        "--loss_weights", nargs="+", type=float, help="Weights for loss components"
    )
    parser.add_argument(
        "--loss_discount", type=float, default=1, help="Discount factor for loss"
    )

    # Trainer configuration
    parser.add_argument(
        "--train_batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--train_lr", type=float, default=2e-4, help="Learning rate for training"
    )
    parser.add_argument(
        "--gradient_accumulate_every",
        type=int,
        default=2,
        help="Frequency of gradient accumulation",
    )
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.005,
        help="Decay rate for exponential moving average",
    )
    parser.add_argument(
        "--sample_freq",
        type=int,
        default=10,
        help="Frequency of sampling during training",
    )
    parser.add_argument(
        "--save_freq", type=int, default=500, help="Frequency of saving checkpoints"
    )
    parser.add_argument(
        "--label_freq",
        type=int,
        default=int(2e4 // 50),
        help="Frequency of labeling during training",
    )
    parser.add_argument(
        "--save_parallel",
        action="store_true",
        default=False,
        help="Whether to save in parallel",
    )
    parser.add_argument(
        "--bucket",
        type=str,
        default=None,
        help="Whether to save in parallel",
    )
    parser.add_argument(
        "--n_reference",
        type=int,
        default=1,
        help="Number of reference samples for evaluation",
    )
    parser.add_argument(
        "--n_samples", type=int, default=1, help="Number of samples for generation"
    )

    return parser


def get_device_settings(all_args):
    """
    Set up device settings for training and evaluation.
    Parameters
    ----------
    all_args : argparse.ArgumentParser
        The parser containing all the arguments.

    Returns
    -------
    device : torch.device
        The device to use for training and evaluation.

    """
    # Set system device
    if all_args.cuda:
        print("The used device is gpu!")
        device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        torch.set_num_threads(all_args.n_training_threads)
    else:
        print("The used device is cpu!")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    return device
