from pathlib import Path
from gym import Env
import numpy as np
from tqdm import tqdm
import wandb
import torch
import torch.nn as nn
from torch.distributions import Categorical
from model import Model
from config import Config


class MarioBrosTrainer:
    def __init__(self, env: Env, config: Config):
        self.env = env
        self.config = config

        self.model = Model(env, config).cuda()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.learning_rate
        )
        self.reset()

        if config.log_to_wandb:
            wandb.init(project=config.project)

    def reset(self):
        self.episode_actions = torch.tensor([], requires_grad=True).cuda()
        self.episode_rewards = []

    def save_checkpoint(self, episode):
        save_directory = Path(self.config.save_directory)
        save_directory.mkdir(exist_ok=True)
        filename = save_directory / f"checkpoint_{episode}.pt"
        torch.save(self.model.state_dict(), filename)

    def load_from_checkpoint(self, episode):
        save_directory = Path(self.config.save_directory)
        save_directory.mkdir(exist_ok=True)
        filename = save_directory / f"checkpoint_{episode}.pt"
        self.model.load_state_dict(torch.load(filename))

    def run_training(self):
        pbar = tqdm(range(self.config.n_episodes))
        for episode in pbar:
            state = self.env.reset()
            done = False
            for _ in range(self.config.max_episode_length):
                state = torch.tensor(state.copy()).float().cuda().unsqueeze(0)
                distribution = Categorical(self.model(state))  # todo set seed
                action = distribution.sample()
                state, reward, done, _ = self.env.step(action.item())
                self.episode_actions = torch.cat(
                    [self.episode_actions, distribution.log_prob(action).reshape(1)]
                )
                self.episode_rewards.append(reward)

                if done:
                    break

            loss = self.backward()
            pbar.set_description(f"Loss: {loss:.04f}")
            if self.config.log_to_wandb:
                wandb.log({"reward": np.sum(self.episode_rewards), "loss": loss})

            if episode % self.config.save_checkpoint_every == 0 and episode > 0:
                self.save_checkpoint(episode)

    def backward(self):
        future_reward = 0
        rewards = []
        for r in self.episode_rewards[::-1]:
            future_reward = r + self.config.gamma * future_reward
            rewards.append(future_reward)
        rewards = torch.tensor(rewards[::-1], dtype=torch.float32).cuda()
        rewards = (rewards - rewards.mean()) / (
            rewards.std() + np.finfo(np.float32).eps
        )
        loss = torch.sum(torch.mul(self.episode_actions, rewards).mul(-1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.reset()
        return loss.item()
