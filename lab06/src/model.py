"""The best Lab 3 model: a small CNN (20,490 parameters, 89.53 % test accuracy)."""
from torch import nn


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 10),
        )

    def forward(self, x):
        return self.net(x)


def build_model(name):
    if name != "cnn":
        raise ValueError(f"unknown model {name}")
    return CNN()
