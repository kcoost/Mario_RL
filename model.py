from gym import Env
import torch.nn as nn
from config import Config


class Model(nn.Module):
    def __init__(self, env: Env, config: Config):
        super().__init__()
        width, height, n_channels = env.env.env.env.env.env.screen.shape  # lol

        layers = []

        # conv layers
        in_channels = (n_channels,) + config.hidden_in_channels[:-1]
        out_channels = config.hidden_in_channels
        for in_channel, out_channel, kernel_size, stride in zip(
            in_channels, out_channels, config.kernel_sizes, config.strides
        ):
            layers.append(nn.Conv2d(in_channel, out_channel, kernel_size, stride))
            layers.append(nn.GELU())

            width = int((width - kernel_size) / stride + 1)
            height = int((height - kernel_size) / stride + 1)

        layers.append(nn.Flatten())

        # mlps
        assert width > 0, "the convolutional layers reduce the width by too much"
        assert height > 0, "the convolutional layers reduce the height by too much"
        in_features = (
            width * height * config.hidden_in_channels[-1],
        ) + config.mlp_hidden_sizes
        out_features = config.mlp_hidden_sizes + (env.action_space.n,)
        for l, (in_feature, out_feature) in enumerate(zip(in_features, out_features)):
            layers.append(nn.Linear(in_feature, out_feature))
            if l < len(in_features) - 1:
                layers.append(nn.GELU())
            else:
                layers.append(nn.Softmax(dim=-1))

        self.layers = nn.Sequential(*layers)

    def forward(self, state):
        state = state.transpose(1, 3).transpose(2, 3)
        return self.layers(state)
