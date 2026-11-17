"""Lab 7 models: an oversized teacher CNN, a small student CNN, and filter pruning."""
import torch
from torch import nn


class CNN(nn.Module):
    """conv(1->c1) -> pool -> conv(c1->c2) -> pool -> linear(c2*7*7 -> hidden) -> linear(hidden -> 10)"""

    def __init__(self, c1, c2, hidden):
        super().__init__()
        self.conv1 = nn.Conv2d(1, c1, 3, padding=1)
        self.conv2 = nn.Conv2d(c1, c2, 3, padding=1)
        self.fc1 = nn.Linear(c2 * 7 * 7, hidden)
        self.fc2 = nn.Linear(hidden, 10)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))         # 28x28 -> 14x14
        x = self.pool(torch.relu(self.conv2(x)))         # 14x14 -> 7x7
        x = torch.relu(self.fc1(x.flatten(1)))
        return self.fc2(x)


def baseline():
    return CNN(32, 64, 320)       # 1,025,866 parameters


def student():
    return CNN(16, 32, 64)        # 105,866 parameters


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def remove_filters(model, fraction):
    """Structured pruning: drop the conv filters with the smallest L2 norm.

    Returns a new, smaller dense CNN. Removing a filter also removes the input
    channel it fed in the next layer, and for conv2 the 49 inputs of fc1 that
    came from that channel.
    """
    keep1 = _strongest_filters(model.conv1.weight, fraction)
    keep2 = _strongest_filters(model.conv2.weight[:, keep1], fraction)
    small = CNN(len(keep1), len(keep2), model.fc1.out_features)
    with torch.no_grad():
        small.conv1.weight.copy_(model.conv1.weight[keep1])
        small.conv1.bias.copy_(model.conv1.bias[keep1])
        small.conv2.weight.copy_(model.conv2.weight[keep2][:, keep1])
        small.conv2.bias.copy_(model.conv2.bias[keep2])
        columns = (keep2[:, None] * 49 + torch.arange(49)).flatten()   # channel-major flatten
        small.fc1.weight.copy_(model.fc1.weight[:, columns])
        small.fc1.bias.copy_(model.fc1.bias)
        small.fc2.load_state_dict(model.fc2.state_dict())
    return small


def _strongest_filters(weight, fraction):
    norms = weight.flatten(1).norm(dim=1)
    n_keep = weight.shape[0] - int(round(fraction * weight.shape[0]))
    return norms.argsort(descending=True)[:n_keep].sort().values
