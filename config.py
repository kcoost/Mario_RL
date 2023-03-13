from dataclasses import dataclass
from typing import Literal, Tuple
from itertools import product


@dataclass
class EnvConfig:
    game: Literal["SuperMarioBros", "SuperMarioBros2"] = "SuperMarioBros"
    version: Literal["v0", "v1", "v2", "v3"] = "v1"
    """
    v0: standard ROM
    v1: downsampled ROM
    v2: pixel ROM
    v3: rectangle ROM
    """
    worlds: Tuple[int] = (1,)
    levels: Tuple[int] = (1,)


@dataclass
class ModelConfig:
    hidden_in_channels: Tuple[int] = (16, 24, 32, 48, 64, 64)
    """
    number of channels of the convolutional layers
    the number of in_channels of the first layer is determined by the environment
    """
    kernel_sizes: Tuple[int] = (6, 6, 4, 4, 3, 3)
    strides: Tuple[int] = (2, 2, 2, 2, 1, 1)

    mlp_hidden_sizes: Tuple[int] = (1024, 512, 256)
    """
    dimension of the "hidden" mlp layers
    the dimension of the first layer is determined by the convolutional layers
    and the dimension of the last is the number of actions
    """


@dataclass
class TrainerConfig:
    learning_rate: float = 1e-4
    save_directory: str = "checkpoints"
    save_checkpoint_every: int = int(1e10)
    n_episodes: int = int(1e2)
    log_to_wandb: bool = True
    project: str = "Mario_RL"
    gamma: float = 0.95
    # stage 1-1 can be completed in 20 seconds * 60 fps = 1200
    max_episode_length: int = 500
    seed: int = 3407  # https://arxiv.org/abs/2109.08203


@dataclass
class Config(EnvConfig, ModelConfig, TrainerConfig):
    def __post_init__(self):
        self.game_id = self.game + "-" + self.version
        self.stages = [f"{w}-{l}" for w, l in product(self.worlds, self.levels)]

        assert len(self.hidden_in_channels) == len(
            self.kernel_sizes
        ), "hidden_in_channels is not of the same length as kernel_sizes"
        assert len(self.hidden_in_channels) == len(
            self.strides
        ), "hidden_in_channels is not of the same length as strides"
