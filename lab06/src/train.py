"""YAML-driven training for the Lab 3 CNN on Fashion-MNIST, with checkpoints and MLflow.

Run from the repository root:
    python lab06/src/train.py --config lab06/config.yaml
    python lab06/src/train.py --config lab06/config.yaml --set train.lr=0.05 --set run_name=lr005
    python lab06/src/train.py --config lab06/config.yaml --stop-after 3     # simulate a crash
    python lab06/src/train.py --resume data/lab06/checkpoints/main/last.pt  # continue it
"""
import argparse
import copy
import csv
import random
import threading
import time
from pathlib import Path

import mlflow
import numpy as np
import psutil
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from model import build_model

ROOT = Path(__file__).resolve().parents[2]


# ----------------------------------------------------------------- config
def load_config(path, overrides=()):
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for item in overrides:
        key, value = item.split("=", 1)
        node = cfg
        *parents, last = key.split(".")
        for part in parents:
            node = node[part]
        node[last] = yaml.safe_load(value)          # "0.05" -> 0.05, "true" -> True
    return cfg


def flatten(cfg, prefix=""):
    out = {}
    for key, value in cfg.items():
        if isinstance(value, dict):
            out.update(flatten(value, f"{prefix}{key}."))
        else:
            out[f"{prefix}{key}"] = value
    return out


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------- components
def build_loaders(cfg):
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
    root = ROOT / cfg["data"]["root"]
    full = datasets.FashionMNIST(root, train=True, download=True, transform=tf)
    test = datasets.FashionMNIST(root, train=False, download=True, transform=tf)
    val_size = cfg["data"]["val_size"]
    train, val = random_split(full, [len(full) - val_size, val_size],
                              generator=torch.Generator().manual_seed(cfg["seed"]))
    workers = cfg["data"]["num_workers"]
    order = torch.Generator().manual_seed(cfg["seed"])
    train_loader = DataLoader(train, batch_size=cfg["data"]["batch_size"], shuffle=True,
                              generator=order, num_workers=workers,
                              persistent_workers=workers > 0)
    val_loader = DataLoader(val, batch_size=512)
    test_loader = DataLoader(test, batch_size=512)
    return train_loader, val_loader, test_loader, order


def build_optimizer(cfg, model):
    t = cfg["train"]
    if t["optimizer"] == "adam":
        return torch.optim.Adam(model.parameters(), lr=t["lr"])
    if t["optimizer"] == "sgd":
        return torch.optim.SGD(model.parameters(), lr=t["lr"], momentum=t["momentum"])
    raise ValueError(t["optimizer"])


def build_scheduler(cfg, optimizer):
    t = cfg["train"]
    kind = t["scheduler"]
    if kind == "none":
        return None
    if kind == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=t["step_size"], gamma=t["gamma"])
    if kind == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t["epochs"])
    if kind == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=t["gamma"],
                                                          patience=t["plateau_patience"])
    raise ValueError(kind)


# --------------------------------------------------------------- training
class PeakMemory:
    """Samples this process's RSS in a background thread; .peak_mb after stop()."""

    def __init__(self):
        self.process = psutil.Process()
        self.peak = self.process.memory_info().rss
        self.done = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.done.is_set():
            self.peak = max(self.peak, self.process.memory_info().rss)
            time.sleep(0.002)

    def start(self):
        self.thread.start()
        return self

    def stop(self):
        self.done.set()
        self.thread.join()
        return self.peak / 2**20


def train_one_epoch(model, loader, optimizer, accumulation):
    """One pass over the data. Gradients are summed over `accumulation` batches and divided by
    the number of samples in the group, so 4 x 64 gives the same update as one batch of 256."""
    model.train()
    total_loss, seen, group = 0.0, 0, 0
    n_batches = len(loader)
    optimizer.zero_grad()
    for i, (x, y) in enumerate(loader):
        loss = F.cross_entropy(model(x), y, reduction="sum")
        loss.backward()
        total_loss += loss.item()
        seen += len(y)
        group += len(y)
        if (i + 1) % accumulation == 0 or i + 1 == n_batches:
            for p in model.parameters():
                p.grad /= group
            optimizer.step()
            optimizer.zero_grad()
            group = 0
    return total_loss / seen, seen


def evaluate(model, loader):
    model.eval()
    loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            out = model(x)
            loss += F.cross_entropy(out, y, reduction="sum").item()
            correct += (out.argmax(1) == y).sum().item()
            total += len(y)
    return loss / total, correct / total


def save_checkpoint(path, epoch, model, optimizer, scheduler, history, order, cfg, run_id):
    torch.save({
        "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None, "history": history,
        "config": cfg, "mlflow_run_id": run_id,
        "rng": {"torch": torch.get_rng_state(), "numpy": np.random.get_state(),
                "python": random.getstate(), "loader": order.get_state()},
    }, path)


def write_history(path, history):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def run(cfg=None, resume=None, stop_after=None, quiet=False):
    """Train from cfg, or continue from a checkpoint. Returns history, test accuracy, run id."""
    checkpoint = torch.load(resume) if resume else None
    if checkpoint:
        cfg = checkpoint["config"]
    cfg = copy.deepcopy(cfg)
    set_seed(cfg["seed"])
    train_loader, val_loader, test_loader, order = build_loaders(cfg)
    model = build_model(cfg["model"]["name"])
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)
    history, start_epoch = [], 1
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler:
            scheduler.load_state_dict(checkpoint["scheduler"])
        history = checkpoint["history"]
        start_epoch = checkpoint["epoch"] + 1
        rng = checkpoint["rng"]
        torch.set_rng_state(rng["torch"])
        np.random.set_state(rng["numpy"])
        random.setstate(rng["python"])
        order.set_state(rng["loader"])

    ckpt_dir = ROOT / cfg["checkpoint"]["dir"] / cfg["run_name"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri((ROOT / cfg["mlflow"]["tracking_dir"]).as_uri())
    mlflow.set_experiment(cfg["mlflow"]["experiment"])
    run_id = checkpoint["mlflow_run_id"] if checkpoint else None

    with mlflow.start_run(run_id=run_id, run_name=None if run_id else cfg["run_name"]) as active:
        run_id = active.info.run_id
        if not checkpoint:
            mlflow.log_params(flatten(cfg))
        t = cfg["train"]
        rss_before = psutil.Process().memory_info().rss / 2**20
        for epoch in range(start_epoch, t["epochs"] + 1):
            memory = PeakMemory().start() if t["measure_memory"] else None
            start = time.perf_counter()
            train_loss, seen = train_one_epoch(model, train_loader, optimizer, t["accumulation_steps"])
            seconds = time.perf_counter() - start
            peak = memory.stop() if memory else None
            val_loss, val_acc = evaluate(model, val_loader)
            lr = optimizer.param_groups[0]["lr"]
            if scheduler:
                scheduler.step(val_loss) if t["scheduler"] == "plateau" else scheduler.step()
            row = {"epoch": epoch, "train_loss": round(train_loss, 5), "val_loss": round(val_loss, 5),
                   "val_accuracy": round(val_acc, 4), "lr": lr, "epoch_seconds": round(seconds, 2),
                   "samples_per_second": round(seen / seconds, 1)}
            if peak is not None:
                row["rss_before_training_mb"] = round(rss_before, 1)
                row["peak_rss_mb"] = round(peak, 1)
            history.append(row)
            mlflow.log_metrics({k: v for k, v in row.items() if k != "epoch"}, step=epoch)
            save_checkpoint(ckpt_dir / "last.pt", epoch, model, optimizer, scheduler, history,
                            order, cfg, run_id)
            if not quiet:
                print(f"  {cfg['run_name']:<24} epoch {epoch}/{t['epochs']}  loss {train_loss:.4f}  "
                      f"val {val_acc:.4f}  {seconds:.1f} s", flush=True)
            if stop_after and epoch == stop_after:
                print(f"  {cfg['run_name']}: stopped after epoch {epoch} (checkpoint saved)", flush=True)
                return {"history": history, "test_accuracy": None, "run_id": run_id,
                        "checkpoint": str(ckpt_dir / "last.pt")}

        _, test_acc = evaluate(model, test_loader)
        mlflow.log_metric("test_accuracy", test_acc)
        write_history(ckpt_dir / "history.csv", history)
        (ckpt_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        mlflow.log_artifact(str(ckpt_dir / "history.csv"))
        mlflow.log_artifact(str(ckpt_dir / "config.yaml"))
        if cfg["run_name"] == "main":
            mlflow.log_artifact(str(ckpt_dir / "last.pt"))
    return {"history": history, "test_accuracy": test_acc, "run_id": run_id,
            "checkpoint": str(ckpt_dir / "last.pt")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "lab06" / "config.yaml"))
    parser.add_argument("--set", action="append", default=[], help="override, e.g. train.lr=0.05")
    parser.add_argument("--resume", help="checkpoint to continue from")
    parser.add_argument("--stop-after", type=int, help="stop after this epoch (simulated crash)")
    args = parser.parse_args()
    cfg = None if args.resume else load_config(args.config, args.set)
    result = run(cfg, resume=args.resume, stop_after=args.stop_after)
    if result["test_accuracy"] is not None:
        print(f"test accuracy {result['test_accuracy']:.4f}, MLflow run {result['run_id']}")


if __name__ == "__main__":
    main()
