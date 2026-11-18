"""Lab 6 studies, all run through train.py and lab06/config.yaml.

Run from the repository root: python lab06/src/experiments.py
Each study is skipped if its CSV already exists; --fresh deletes results, checkpoints and
MLflow runs and starts again. Every training run is a separate process, so memory and
DataLoader workers start clean each time.
"""
import csv
import shutil
import subprocess
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile, record_function

from model import build_model
from train import ROOT, build_loaders, build_optimizer, load_config, set_seed

CONFIG = ROOT / "lab06" / "config.yaml"
RESULTS = ROOT / "lab06" / "results"
CHECKPOINTS = ROOT / "data" / "lab06" / "checkpoints"
PYTHON = sys.executable


def train_run(name, *overrides, stop_after=None, resume=None):
    """Run train.py in a fresh process; return its per-epoch history and test accuracy."""
    cmd = [PYTHON, "-B", "-W", "ignore", str(ROOT / "lab06" / "src" / "train.py")]
    if resume:
        cmd += ["--resume", str(resume)]
    else:
        cmd += ["--config", str(CONFIG), "--set", f"run_name={name}"]
        for item in overrides:
            cmd += ["--set", item]
    if stop_after:
        cmd += ["--stop-after", str(stop_after)]
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"{name} failed:\n{out.stderr[-2000:]}")
    for line in out.stdout.splitlines():
        if "epoch" in line or "stopped" in line:
            print(line, flush=True)
    test_acc = None
    for line in out.stdout.splitlines():
        if line.startswith("test accuracy"):
            test_acc = float(line.split()[2].rstrip(","))
    history = list(csv.DictReader(open(CHECKPOINTS / name / "history.csv"))) if test_acc else None
    return history, test_acc


def write_csv(name, rows):
    with open(RESULTS / name, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def done(name):
    return (RESULTS / name).exists()


# ------------------------------------------------------------ 2. profile
def profile_one_epoch():
    """One epoch at batch 128, num_workers 0, with every phase labelled for torch.profiler."""
    cfg = load_config(CONFIG)
    set_seed(cfg["seed"])
    train_loader, _, _, _ = build_loaders(cfg)
    model = build_model("cnn")
    optimizer = build_optimizer(cfg, model)
    phases = ["data_loading", "forward", "backward", "optimizer_step"]
    wall = dict.fromkeys(phases, 0.0)
    model.train()
    batches = iter(train_loader)
    start_epoch = time.perf_counter()
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        while True:
            t0 = time.perf_counter()
            with record_function("data_loading"):
                try:
                    x, y = next(batches)
                except StopIteration:
                    break
            t1 = time.perf_counter()
            with record_function("forward"):
                loss = F.cross_entropy(model(x), y)
            t2 = time.perf_counter()
            with record_function("backward"):
                loss.backward()
            t3 = time.perf_counter()
            with record_function("optimizer_step"):
                optimizer.step()
                optimizer.zero_grad()
            t4 = time.perf_counter()
            for phase, seconds in zip(phases, (t1 - t0, t2 - t1, t3 - t2, t4 - t3)):
                wall[phase] += seconds
        epoch_seconds = time.perf_counter() - start_epoch   # before the profiler post-processes
    events = {e.key: e for e in prof.key_averages()}
    prof_us = {p: events[p].cpu_time_total for p in phases}
    rows = [{"phase": p,
             "profiler_seconds": round(prof_us[p] / 1e6, 2),
             "profiler_share": round(prof_us[p] / sum(prof_us.values()), 4),
             "wall_clock_seconds": round(wall[p], 2),
             "wall_clock_share": round(wall[p] / sum(wall.values()), 4)} for p in phases]
    rows.append({"phase": "whole epoch (with profiler)", "profiler_seconds": "",
                 "profiler_share": "", "wall_clock_seconds": round(epoch_seconds, 2),
                 "wall_clock_share": ""})
    write_csv("profile_breakdown.csv", rows)
    (RESULTS / "profile_top_ops.txt").write_text(
        prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=15), encoding="utf-8")


# ---------------------------------------------------------- 3. workers
def data_only_seconds(workers):
    """Time to start the loader's workers and to read every batch once, with no model."""
    cfg = load_config(CONFIG, [f"data.num_workers={workers}"])
    train_loader, _, _, _ = build_loaders(cfg)
    start = time.perf_counter()
    it = iter(train_loader)
    next(it)
    first_batch = time.perf_counter() - start
    for _ in it:
        pass
    total = time.perf_counter() - start
    return first_batch, total - first_batch


def data_only_in_fresh_process(workers):
    out = subprocess.run([PYTHON, "-B", "-W", "ignore", __file__, "--data-only", str(workers)],
                         cwd=ROOT, capture_output=True, text=True)
    first, rest = map(float, out.stdout.strip().splitlines()[-1].split())
    return first, rest


def workers_study():
    rows = []
    for w in [0, 1, 2, 4, 8]:
        first, rest = data_only_in_fresh_process(w)
        history, test_acc = train_run(f"workers_{w}", f"data.num_workers={w}", "train.epochs=2")
        rows.append({"num_workers": w, "worker_start_s": round(first, 2),
                     "data_only_epoch_s": round(rest, 2),
                     "train_epoch1_s": float(history[0]["epoch_seconds"]),
                     "train_epoch2_s": float(history[1]["epoch_seconds"]),
                     "samples_per_second_epoch2": float(history[1]["samples_per_second"])})
        print(rows[-1], flush=True)
    write_csv("num_workers.csv", rows)


# ------------------------------------------------------- 4. batch size
def batch_study():
    rows = []
    for bs in [32, 64, 128, 256, 512]:
        history, test_acc = train_run(f"batch_{bs}", f"data.batch_size={bs}",
                                      "train.measure_memory=true")
        rows.append({"batch_size": bs,
                     "samples_per_second": round(sum(float(h["samples_per_second"]) for h in history)
                                                 / len(history), 1),
                     "rss_before_training_mb": float(history[0]["rss_before_training_mb"]),
                     "peak_rss_mb": max(float(h["peak_rss_mb"]) for h in history),
                     "val_accuracy": float(history[-1]["val_accuracy"]),
                     "test_accuracy": test_acc})
        rows[-1]["peak_increase_mb"] = round(rows[-1]["peak_rss_mb"] - rows[-1]["rss_before_training_mb"], 1)
        print(rows[-1], flush=True)
    write_csv("batch_size.csv", rows)


# ---------------------------------------- 5. schedules and optimizers
SCHEDULE_RUNS = [(opt, sched) for opt in ("sgd", "adam") for sched in ("step", "cosine", "plateau")]


def schedule_study():
    rows, curves = [], []
    for opt, sched in SCHEDULE_RUNS:
        lr = 0.05 if opt == "sgd" else 0.001
        name = f"{opt}_{sched}"
        history, test_acc = train_run(name, f"train.optimizer={opt}", f"train.lr={lr}",
                                      f"train.scheduler={sched}", "train.epochs=10")
        rows.append({"optimizer": "SGD + momentum 0.9" if opt == "sgd" else "Adam",
                     "start_lr": lr, "scheduler": sched,
                     "best_val_accuracy": max(float(h["val_accuracy"]) for h in history),
                     "final_val_accuracy": float(history[-1]["val_accuracy"]),
                     "final_lr": float(history[-1]["lr"]), "test_accuracy": test_acc})
        curves += [{"run": name, **h} for h in history]
        print(rows[-1], flush=True)
    write_csv("schedules.csv", rows)
    write_csv("schedule_curves.csv", curves)


# --------------------------------------------- 6. gradient accumulation
def accumulation_study():
    curves = []
    for name, overrides in (("accum_batch256", ["data.batch_size=256"]),
                            ("accum_64x4", ["data.batch_size=64", "train.accumulation_steps=4"])):
        history, test_acc = train_run(name, *overrides)
        curves += [{"run": name, "test_accuracy": test_acc, **h} for h in history]
    write_csv("grad_accumulation.csv", curves)


# ------------------------------------------------------ 7. interrupt/resume
def resume_study():
    common = ["train.scheduler=cosine", "train.epochs=6"]
    full, full_acc = train_run("resume_reference", *common)
    train_run("resume_interrupted", *common, stop_after=3)
    resumed, resumed_acc = train_run("resume_interrupted",
                                     resume=CHECKPOINTS / "resume_interrupted" / "last.pt")
    rows = []
    for a, b in zip(full, resumed):
        rows.append({"epoch": a["epoch"], "uninterrupted_train_loss": a["train_loss"],
                     "resumed_train_loss": b["train_loss"],
                     "uninterrupted_val_accuracy": a["val_accuracy"],
                     "resumed_val_accuracy": b["val_accuracy"],
                     "uninterrupted_lr": a["lr"], "resumed_lr": b["lr"],
                     "part": "before interruption" if int(a["epoch"]) <= 3 else "after resume"})
    rows.append({"epoch": "test", "uninterrupted_val_accuracy": full_acc, "resumed_val_accuracy": resumed_acc})
    write_csv("resume.csv", rows)


# ------------------------------------------------------------- MLflow
def export_mlflow():
    cfg = load_config(CONFIG)
    mlflow.set_tracking_uri((ROOT / cfg["mlflow"]["tracking_dir"]).as_uri())
    runs = mlflow.search_runs(experiment_names=[cfg["mlflow"]["experiment"]])
    keep = ["tags.mlflow.runName", "run_id", "params.data.batch_size", "params.data.num_workers",
            "params.train.optimizer", "params.train.lr", "params.train.scheduler",
            "params.train.epochs", "params.train.accumulation_steps", "metrics.val_accuracy",
            "metrics.test_accuracy", "metrics.samples_per_second"]
    runs[keep].sort_values("tags.mlflow.runName").to_csv(RESULTS / "mlflow_runs.csv", index=False)
    return len(runs)


# --------------------------------------------------------------- plots
def plots():
    w = pd.read_csv(RESULTS / "num_workers.csv")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(w.num_workers, w.data_only_epoch_s, "o-", label="data loading only (one pass, workers already up)")
    ax.plot(w.num_workers, w.train_epoch2_s, "o-", label="training epoch (2nd, workers already up)")
    ax.plot(w.num_workers, w.train_epoch1_s, "x--", label="training epoch (1st, incl. worker start-up)")
    ax.set_xlabel("num_workers")
    ax.set_ylabel("Seconds per epoch (55,000 images)")
    ax.set_title("DataLoader workers: time per epoch")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "num_workers.png", dpi=150)
    plt.close(fig)

    curves = pd.read_csv(RESULTS / "schedule_curves.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.5))
    for run, g in curves.groupby("run", sort=False):
        style = "-" if run.startswith("adam") else "--"
        a1.plot(g.epoch, g.val_accuracy * 100, style, marker="o", markersize=3, label=run)
        a2.plot(g.epoch, g.lr, style, marker="o", markersize=3, label=run)
    a1.set_xlabel("Epoch")
    a1.set_ylabel("Validation accuracy (%)")
    a1.set_title("10-epoch budget: optimizers and schedules")
    a2.set_yscale("log")
    a2.set_xlabel("Epoch")
    a2.set_ylabel("Learning rate used in the epoch")
    a2.set_title("Learning rate per epoch")
    for a in (a1, a2):
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "schedules.png", dpi=150)
    plt.close(fig)

    acc = pd.read_csv(RESULTS / "grad_accumulation.csv")
    res = pd.read_csv(RESULTS / "resume.csv")
    res = res[res.epoch != "test"].astype({"epoch": int})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.5))
    for run, g in acc.groupby("run", sort=False):
        a1.plot(g.epoch, g.train_loss, "o-" if run == "accum_batch256" else "x--",
                label="batch 256" if run == "accum_batch256" else "batch 64 x 4 accumulation")
    a1.set_xlabel("Epoch")
    a1.set_ylabel("Training loss")
    a1.set_title("Gradient accumulation")
    a2.plot(res.epoch, res.uninterrupted_train_loss, "o-", linewidth=3, alpha=0.5, label="uninterrupted")
    a2.plot(res.epoch, res.resumed_train_loss, "x--", color="black", label="stopped after epoch 3, resumed")
    a2.axvline(3.5, color="grey", linestyle=":")
    a2.set_xlabel("Epoch")
    a2.set_ylabel("Training loss")
    a2.set_title("Interrupt at epoch 3 and resume from checkpoint")
    for a in (a1, a2):
        a.grid(alpha=0.3)
        a.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "accumulation_and_resume.png", dpi=150)
    plt.close(fig)


def main():
    if "--data-only" in sys.argv:                      # helper mode, used by workers_study
        first, rest = data_only_seconds(int(sys.argv[sys.argv.index("--data-only") + 1]))
        print(f"{first} {rest}")
        return
    if "--fresh" in sys.argv:
        for path in (RESULTS, CHECKPOINTS, ROOT / "lab06" / "mlruns"):
            shutil.rmtree(path, ignore_errors=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    print(f"torch {torch.__version__}, {torch.get_num_threads()} threads", flush=True)
    steps = [("main_run.csv", lambda: write_csv("main_run.csv", train_run("main")[0])),
             ("profile_breakdown.csv", profile_one_epoch),
             ("num_workers.csv", workers_study),
             ("batch_size.csv", batch_study),
             ("schedules.csv", schedule_study),
             ("grad_accumulation.csv", accumulation_study),
             ("resume.csv", resume_study)]
    for csv_name, step in steps:
        if done(csv_name):
            print(f"skip {csv_name} (already done)", flush=True)
            continue
        print(f"== {csv_name}", flush=True)
        step()
    print(f"MLflow runs: {export_mlflow()}", flush=True)
    plots()
    print("done", flush=True)


if __name__ == "__main__":
    main()
