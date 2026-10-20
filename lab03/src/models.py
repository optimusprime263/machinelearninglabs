"""The three Lab 3 models: MLP, CNN and a small patch-attention network."""
from torch import nn


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


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


class PatchAttention(nn.Module):
    """28x28 image -> 16 patches of 7x7 -> 2 Transformer encoder layers -> mean pool."""

    def __init__(self, patch=7, d_model=64, nhead=4, dim_feedforward=128, num_layers=2):
        super().__init__()
        self.patch = patch
        self.embed = nn.Linear(patch * patch, d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.head = nn.Linear(d_model, 10)

    def forward(self, x):
        p = self.patch
        x = x.unfold(2, p, p).unfold(3, p, p)            # (B, 1, 4, 4, 7, 7)
        x = x.reshape(x.shape[0], -1, p * p)              # (B, 16, 49)
        x = self.encoder(self.embed(x))                   # (B, 16, 64)
        return self.head(x.mean(dim=1))


def build_models():
    return {"MLP": MLP(), "CNN": CNN(), "Attention": PatchAttention()}
