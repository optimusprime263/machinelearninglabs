"""Lab 3: MLP, CNN and attention on Fashion-MNIST, and what each one costs.

Run from the repository root: python lab03/src/lab3.py
Fashion-MNIST is downloaded to data/ if it is missing.
Outputs go to lab03/results/.
"""
import copy
import csv
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from thop import profile
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.utils.flop_counter import FlopCounterMode
from torchinfo import summary
from torchvision import datasets, transforms

from models import build_models

SEED = 42
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = Path(__file__).resolve().parents[1] / "results"
BATCH_SIZE = 128
EPOCHS = 5
LR = 1e-3


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------- 1. Data
def load_data():
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.2860,), (0.3530,))])
    train_full = datasets.FashionMNIST(DATA, train=True, download=True, transform=tf)
    test = datasets.FashionMNIST(DATA, train=False, download=True, transform=tf)
    train, val = random_split(train_full, [55000, 5000],
                              generator=torch.Generator().manual_seed(SEED))
    loader = torch.Generator().manual_seed(SEED)
    return (DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, generator=loader),
            DataLoader(val, batch_size=BATCH_SIZE),
            DataLoader(test, batch_size=BATCH_SIZE))


# ------------------------------------------------------- 3. FLOPs by hand
def hand_flops():
    """FLOPs for one image, matrix products only (no bias, activation, softmax, norm)."""
    seq, d, ff = 16, 64, 128
    attention_layer = {
        "QKV projections": 2 * seq * d * 3 * d,
        "scores Q·Kᵀ": 2 * seq * seq * d,
        "weighted sum A·V": 2 * seq * seq * d,
        "output projection": 2 * seq * d * d,
        "feed-forward": 2 * seq * d * ff * 2,
    }
    layers = {
        "MLP": {
            "Linear 784→256": 2 * 784 * 256,
            "Linear 256→128": 2 * 256 * 128,
            "Linear 128→10": 2 * 128 * 10,
        },
        "CNN": {
            "Conv2d 1→16, 28×28": 2 * 3 * 3 * 1 * 16 * 28 * 28,
            "Conv2d 16→32, 14×14": 2 * 3 * 3 * 16 * 32 * 14 * 14,
            "Linear 1568→10": 2 * 1568 * 10,
        },
        "Attention": {
            "patch embedding 49→64": 2 * seq * 49 * d,
            **{f"encoder layer 1: {k}": v for k, v in attention_layer.items()},
            **{f"encoder layer 2: {k}": v for k, v in attention_layer.items()},
            "Linear 64→10": 2 * d * 10,
        },
    }
    return layers


def measured_flops(model):
    x = torch.randn(1, 1, 28, 28)
    model.eval()
    macs, _ = profile(copy.deepcopy(model), inputs=(x,), verbose=False)
    # thop runs under no_grad, where the encoder layers use one fused kernel
    # and their submodules never run. Count again with that fast path off.
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        macs_no_fastpath, _ = profile(copy.deepcopy(model), inputs=(x,), verbose=False)
    finally:
        torch.backends.mha.set_fastpath_enabled(True)
    # With gradients enabled the encoder runs its normal (non-fused) path,
    # so PyTorch's counter sees every matrix product it has a formula for.
    with FlopCounterMode(display=False) as counter:
        model(x)
    return int(macs), int(macs_no_fastpath), counter.get_total_flops()


# --------------------------------------------------------------- Latency
def median_latency_ms(model, batch, runs):
    model.eval()
    x = torch.randn(batch, 1, 28, 28)
    with torch.inference_mode():
        model(x)                                  # warm-up
        times = []
        for _ in range(runs):
            start = time.perf_counter()
            model(x)
            times.append(time.perf_counter() - start)
    return float(np.median(times)) * 1000


# -------------------------------------------------------------- 4. Training
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            correct += (model(x).argmax(dim=1) == y).sum().item()
            total += y.numel()
    return correct / total


def train(name, model, train_loader, val_loader, log_rows):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        start = time.perf_counter()
        running, seen = 0.0, 0
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item() * y.numel()
            seen += y.numel()
        seconds = time.perf_counter() - start
        val_acc = evaluate(model, val_loader)
        log_rows.append({"model": name, "epoch": epoch,
                         "train_loss": f"{running / seen:.4f}",
                         "val_accuracy": f"{val_acc:.4f}",
                         "epoch_seconds": f"{seconds:.1f}"})
        print(f"{name:<9} epoch {epoch}/{EPOCHS}  loss {running / seen:.4f}  "
              f"val acc {val_acc:.4f}  {seconds:.1f} s")


# ----------------------------------------------------------------- 5. Plots
def plot_tradeoffs(rows, path):
    colors = {"MLP": "tab:blue", "CNN": "tab:orange", "Attention": "tab:green"}
    panels = [("params", "Parameters", True),
              ("latency_b1_ms", "Latency, batch 1 (ms)", False),
              ("latency_b128_per_image_ms", "Latency per image, batch 128 (ms)", False)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, (key, label, log_x) in zip(axes, panels):
        for r in rows:
            ax.scatter(r[key], r["test_accuracy"] * 100, s=80, color=colors[r["model"]],
                       label=r["model"])
            ax.annotate(r["model"], (r[key], r["test_accuracy"] * 100),
                        textcoords="offset points", xytext=(6, 6))
        if log_x:
            ax.set_xscale("log")
        ax.set_xlabel(label)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Test accuracy (%)")
    fig.suptitle("Fashion-MNIST: accuracy against size and latency (5 epochs, CPU)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    set_seed()
    RESULTS.mkdir(exist_ok=True)
    train_loader, val_loader, test_loader = load_data()
    print(f"train {len(train_loader.dataset)}, val {len(val_loader.dataset)}, "
          f"test {len(test_loader.dataset)}, torch threads {torch.get_num_threads()}")

    layers = hand_flops()
    summaries, flop_rows, rows, log_rows = [], [], [], []

    for name in ["MLP", "CNN", "Attention"]:
        set_seed()
        model = build_models()[name]

        info = summary(model, input_size=(1, 1, 28, 28), verbose=0,
                       col_names=("input_size", "output_size", "num_params"))
        summaries.append(f"===== {name}\n{info}\n")
        params = info.total_params

        hand_total = sum(layers[name].values())
        thop_macs, thop_macs_no_fastpath, torch_flops = measured_flops(model)
        flop_rows.append({"model": name, "hand_flops": hand_total,
                          "thop_macs": thop_macs, "thop_macs_x2": 2 * thop_macs,
                          "thop_macs_fastpath_off": thop_macs_no_fastpath,
                          "torch_flop_counter": torch_flops})

        lat1 = median_latency_ms(model, 1, runs=100)
        lat128 = median_latency_ms(model, 128, runs=20)

        train(name, model, train_loader, val_loader, log_rows)
        test_acc = evaluate(model, test_loader)
        print(f"{name:<9} test accuracy {test_acc:.4f}")

        rows.append({"model": name, "params": params, "hand_flops": hand_total,
                     "test_accuracy": test_acc, "latency_b1_ms": lat1,
                     "latency_b128_ms": lat128,
                     "latency_b128_per_image_ms": lat128 / 128})

    (RESULTS / "torchinfo_summary.txt").write_text("\n".join(summaries), encoding="utf-8")
    write_csv(RESULTS / "training_log.csv", log_rows)
    write_csv(RESULTS / "flops.csv", flop_rows)
    write_csv(RESULTS / "hand_flops_by_layer.csv",
              [{"model": m, "layer": layer, "flops": f}
               for m, d in layers.items() for layer, f in d.items()])
    write_csv(RESULTS / "results.csv", [{
        "model": r["model"], "params": r["params"], "hand_flops": r["hand_flops"],
        "test_accuracy": f"{r['test_accuracy']:.4f}",
        "latency_b1_ms": f"{r['latency_b1_ms']:.3f}",
        "latency_b128_ms": f"{r['latency_b128_ms']:.2f}",
        "latency_b128_per_image_ms": f"{r['latency_b128_per_image_ms']:.4f}",
    } for r in rows])
    plot_tradeoffs(rows, RESULTS / "tradeoffs.png")

    for r in flop_rows:
        print(r)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
