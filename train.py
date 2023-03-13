import os
from pathlib import Path
import torch
import yaml
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from trainer import MarioBrosTrainer
from config import Config

if __name__ == "__main__":
    try:
        with open(Path(__file__).parent / "wandb_config.yml", "r") as file:
            wandb_config = yaml.safe_load(file)
        os.environ["WANDB_BASE_URL"] = wandb_config["WANDB_BASE_URL"]
        os.environ["WANDB_API_KEY"] = wandb_config["WANDB_API_KEY"]
    except FileNotFoundError:
        print(
            "wandb config file not found. Continuing with WANDB_BASE_URL and WANDB_API_KEY set to their default values."
        )

    config = Config()
    torch.manual_seed(config.seed)

    env = gym_super_mario_bros.make(config.game_id)  # , stages=config.stages)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    mario_bros_trainer = MarioBrosTrainer(env, config)

    mario_bros_trainer.run_training()

    env.close()
