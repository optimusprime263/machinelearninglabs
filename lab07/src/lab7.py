"""Lab 7: pruning and knowledge distillation on Fashion-MNIST.

Run from the repository root: python lab07/src/lab7.py
Add --fresh to ignore the cache in data/lab07/ and train everything again.
Outputs go to lab07/results/.
"""
import copy
import csv
import gzip
import io
import json
import random
import statistics
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.flop_counter import FlopCounterMode
from torchvision import datasets, transforms

from models import baseline, count_parameters, remove_filters, student

SEED = 42
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
CACHE = DATA / "lab07"
RESULTS = Path(__file__).resolve().parents[1] / "results"
BATCH_SIZE = 128
EPOCHS = 5                     # baseline teacher and both students
FINETUNE_EPOCHS = 3
LR = 1e-3
FINETUNE_LR = 5e-4
SPARSITIES = [0, 0.25, 0.5, 0.75, 0.9, 0.95]
FILTER_FRACTIONS = [0.1, 0.25, 0.5]
T, ALPHA = 4.0, 0.5
EXTRA_STUDENT_SEEDS = [43, 44]
CPU_WATTS = 15                 # Ryzen 5 5600U TDP, used only for the energy estimate
PRUNABLE = ["conv1.weight", "conv2.weight", "fc1.weight", "fc2.weight"]


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


# ---------------------------------------------------------------- data
def load_data():
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
    train_full = datasets.FashionMNIST(DATA, train=True, download=True, transform=tf)
    test = datasets.FashionMNIST(DATA, train=False, download=True, transform=tf)
    train, val = random_split(train_full, [55000, 5000], generator=torch.Generator().manual_seed(SEED))
    order = torch.Generator()
    return (DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, generator=order),
            DataLoader(val, batch_size=512), DataLoader(test, batch_size=512))


# ------------------------------------------------------------ training
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            correct += (model(x).argmax(1) == y).sum().item()
            total += len(y)
    return correct / total


def kd_loss(student_logits, teacher_logits, labels):
    """alpha * CE + (1 - alpha) * T^2 * KL between the softened distributions.

    F.kl_div(log_softmax(student/T), softmax(teacher/T)) is the PyTorch form of
    KL(softmax(student/T) || softmax(teacher/T)) in the manual: the student's
    log-probabilities are the input, the teacher's probabilities the target.
    """
    ce = F.cross_entropy(student_logits, labels)
    kl = F.kl_div(F.log_softmax(student_logits / T, dim=1),
                  F.softmax(teacher_logits / T, dim=1), reduction="batchmean")
    return ALPHA * ce + (1 - ALPHA) * T * T * kl


def train(name, model, loaders, epochs, lr, log, masks=None, teacher=None, seed=SEED):
    """Adam training; masks (if given) are re-applied after every step so pruned weights stay 0."""
    train_loader, val_loader, _ = loaders
    train_loader.generator.manual_seed(seed)       # same batch order for runs with the same seed
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    params = dict(model.named_parameters())
    if teacher is not None:
        teacher.eval()
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        total, seen = 0.0, 0
        for x, y in train_loader:
            opt.zero_grad()
            out = model(x)
            if teacher is None:
                loss = F.cross_entropy(out, y)
            else:
                with torch.no_grad():
                    teacher_out = teacher(x)
                loss = kd_loss(out, teacher_out, y)
            loss.backward()
            opt.step()
            if masks:
                with torch.no_grad():
                    for key, mask in masks.items():
                        params[key].mul_(mask)
            total += loss.item() * len(y)
            seen += len(y)
        val_acc = evaluate(model, val_loader)
        log.append({"run": name, "epoch": epoch, "train_loss": round(total / seen, 4),
                    "val_accuracy": round(val_acc, 4)})
        print(f"  {name:<28} epoch {epoch}/{epochs}  loss {total / seen:.4f}  val {val_acc:.4f}",
              flush=True)
    return time.perf_counter() - start


# ------------------------------------------------------------- pruning
def magnitude_masks(model, sparsity, mode):
    """Masks that zero the `sparsity` share of weights with the smallest |w|,
    ranked over all layers together ('global') or inside each layer ('layer')."""
    weights = {k: v.detach() for k, v in model.named_parameters() if k in PRUNABLE}
    masks = {}
    if mode == "global":
        scores = torch.cat([w.abs().flatten() for w in weights.values()])
        drop = torch.zeros(scores.numel(), dtype=torch.bool)
        k = int(round(sparsity * scores.numel()))
        if k:
            drop[torch.topk(scores, k, largest=False).indices] = True
        start = 0
        for key, w in weights.items():
            masks[key] = (~drop[start:start + w.numel()]).view_as(w).float()
            start += w.numel()
    else:
        for key, w in weights.items():
            drop = torch.zeros(w.numel(), dtype=torch.bool)
            k = int(round(sparsity * w.numel()))
            if k:
                drop[torch.topk(w.abs().flatten(), k, largest=False).indices] = True
            masks[key] = (~drop).view_as(w).float()
    return masks


def apply_masks(model, masks):
    params = dict(model.named_parameters())
    with torch.no_grad():
        for key, mask in masks.items():
            params[key].mul_(mask)


def layer_density(model):
    params = dict(model.named_parameters())
    return {k: float((params[k] != 0).float().mean()) for k in PRUNABLE}


def weight_sparsity(model):
    params = dict(model.named_parameters())
    zeros = sum(int((params[k] == 0).sum()) for k in PRUNABLE)
    return zeros / sum(params[k].numel() for k in PRUNABLE)


# ------------------------------------------------------- cost measures
def dense_flops(model):
    with FlopCounterMode(display=False) as counter:
        model(torch.zeros(1, 1, 28, 28))
    return counter.get_total_flops()


def layer_flops(model):
    """Multiply-add FLOPs per layer for one image, from the Ch. 4 formulas."""
    c1, c2, h = model.conv1.out_channels, model.conv2.out_channels, model.fc1.out_features
    return {"conv1.weight": 2 * 9 * 1 * c1 * 28 * 28, "conv2.weight": 2 * 9 * c1 * c2 * 14 * 14,
            "fc1.weight": 2 * c2 * 49 * h, "fc2.weight": 2 * h * 10}


def effective_flops(model):
    """FLOPs that touch a non-zero weight; what an ideal sparse kernel would have to do."""
    density = layer_density(model)
    return int(sum(f * density[k] for k, f in layer_flops(model).items()))


def saved_bytes(obj):
    buffer = io.BytesIO()
    torch.save(obj, buffer)
    return buffer.getvalue()


def storage(model):
    """On-disk size of the state dict: dense, gzip of dense, and two sparse formats."""
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    dense = saved_bytes(state)
    coo = {k: (v.to_sparse() if k in PRUNABLE else v) for k, v in state.items()}
    csr = {k: (v.reshape(v.shape[0], -1).to_sparse_csr() if k in PRUNABLE else v)
           for k, v in state.items()}
    return {"dense_kb": len(dense) / 1024, "dense_gzip_kb": len(gzip.compress(dense)) / 1024,
            "sparse_coo_kb": len(saved_bytes(coo)) / 1024,
            "sparse_csr_kb": len(saved_bytes(csr)) / 1024}


def interleaved_latency_ms(models, batch, runs):
    """Median latency of each model; models are called in turn so CPU speed changes hit all equally."""
    x = torch.randn(batch, 1, 28, 28)
    times = {name: [] for name in models}
    with torch.inference_mode():
        for m in models.values():
            m.eval()
            m(x)                                     # warm-up
        for _ in range(runs):
            for name, m in models.items():
                start = time.perf_counter()
                m(x)
                times[name].append(time.perf_counter() - start)
    return {name: statistics.median(t) * 1000 for name, t in times.items()}


def sparse_kernel_benchmark(fc1_weight):
    """fc1 (320 x 3136) as a dense matrix vs a CSR sparse matrix, batch 128."""
    x = torch.randn(3136, 128)
    rows = []
    for sparsity in [0.0, 0.5, 0.75, 0.9, 0.95, 0.99]:
        w = fc1_weight.detach().clone()
        k = int(round(sparsity * w.numel()))
        if k:
            w.view(-1)[torch.topk(w.abs().flatten(), k, largest=False).indices] = 0
        w_csr = w.to_sparse_csr()
        times = {"dense": [], "csr": []}
        for _ in range(3):
            w @ x, torch.sparse.mm(w_csr, x)
        for _ in range(50):
            for name, fn in (("dense", lambda: w @ x), ("csr", lambda: torch.sparse.mm(w_csr, x))):
                start = time.perf_counter()
                fn()
                times[name].append(time.perf_counter() - start)
        dense_ms = statistics.median(times["dense"]) * 1000
        csr_ms = statistics.median(times["csr"]) * 1000
        rows.append({"sparsity": sparsity, "dense_matmul_ms": round(dense_ms, 4),
                     "csr_matmul_ms": round(csr_ms, 4), "csr_vs_dense": round(csr_ms / dense_ms, 2)})
    return rows


# --------------------------------------------------------------- caching
def load_cache(fresh):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / "cache.json"
    if fresh:
        for f in CACHE.glob("*"):
            f.unlink()
    return json.loads(path.read_text()) if path.exists() else {}


def save_cache(cache):
    (CACHE / "cache.json").write_text(json.dumps(cache, indent=1))


def cached_model(cache, key, build, run):
    """Train with run(model) once; later runs load the saved weights and timing."""
    path = CACHE / f"{key}.pt"
    model = build()
    if key in cache and path.exists():
        model.load_state_dict(torch.load(path))
        return model, cache[key]
    info = run(model)
    torch.save(model.state_dict(), path)
    cache[key] = info
    save_cache(cache)
    return model, info


def write_csv(name, rows):
    with open(RESULTS / name, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# ------------------------------------------------------------------ main
def main():
    RESULTS.mkdir(exist_ok=True)
    cache = load_cache("--fresh" in sys.argv)
    loaders = load_data()
    test_loader = loaders[2]
    log = cache.setdefault("log", [])
    print(f"torch {torch.__version__}, {torch.get_num_threads()} threads", flush=True)

    # 1. Oversized baseline (teacher)
    def run_baseline(model):
        set_seed()
        model.load_state_dict(baseline().state_dict())
        return {"train_seconds": train("baseline", model, loaders, EPOCHS, LR, log)}
    set_seed()
    base, base_info = cached_model(cache, "baseline", baseline, run_baseline)
    base_acc = evaluate(base, test_loader)
    print(f"baseline: {count_parameters(base)} parameters, test accuracy {base_acc:.4f}", flush=True)
    models_for_latency = {"baseline": base}

    # Control: the same 3 extra epochs without pruning. Pruned models get 3 more epochs of
    # training than the baseline, so this is the fair "0 % sparsity after fine-tuning" point.
    def run_control(model):
        model.load_state_dict(base.state_dict())
        set_seed()
        return {"finetune_seconds": train("dense_finetune", model, loaders, FINETUNE_EPOCHS,
                                          FINETUNE_LR, log)}
    control, _ = cached_model(cache, "dense_finetune", baseline, run_control)
    control_acc = evaluate(control, test_loader)
    print(f"dense control after 3 more epochs: {control_acc:.4f}", flush=True)

    # 2. Unstructured magnitude pruning, global and per layer, fine-tuned with fixed masks
    unstructured = []
    for mode in ("global", "layer"):
        for s in SPARSITIES:
            if s == 0:
                unstructured.append({"mode": mode, "target_sparsity": 0, "actual_sparsity": 0.0,
                                     "acc_before_finetune": base_acc, "acc_after_finetune": control_acc,
                                     "effective_flops": effective_flops(base)})
                continue
            key = f"{mode}_{s}"
            pruned = copy.deepcopy(base)
            masks = magnitude_masks(pruned, s, mode)
            apply_masks(pruned, masks)
            acc_before = evaluate(pruned, test_loader)

            def run_finetune(model, masks=masks, src=pruned, key=key):
                model.load_state_dict(src.state_dict())
                set_seed()
                seconds = train(key, model, loaders, FINETUNE_EPOCHS, FINETUNE_LR, log, masks=masks)
                return {"finetune_seconds": seconds}
            model, info = cached_model(cache, key, baseline, run_finetune)
            unstructured.append({"mode": mode, "target_sparsity": s,
                                 "actual_sparsity": round(weight_sparsity(model), 4),
                                 "acc_before_finetune": acc_before,
                                 "acc_after_finetune": evaluate(model, test_loader),
                                 "effective_flops": effective_flops(model), **storage(model)})
            if mode == "global" and s in (0.5, 0.9, 0.95):
                models_for_latency[f"global {int(s * 100)}%"] = model
            if mode == "layer" and s == 0.9:
                models_for_latency["layer-wise 90%"] = model
            print(f"{key}: before {acc_before:.4f}, after {unstructured[-1]['acc_after_finetune']:.4f}",
                  flush=True)
    for row in unstructured:
        for k in ("dense_kb", "dense_gzip_kb", "sparse_coo_kb", "sparse_csr_kb"):
            row.setdefault(k, storage(base)[k] if row["target_sparsity"] == 0 else None)
    write_csv("unstructured_pruning.csv", unstructured)

    # 2b. Structured pruning: remove whole conv filters by L2 norm
    structured = [{"filters_removed": 0, "conv1_filters": 32, "conv2_filters": 64,
                   "parameters": count_parameters(base), "acc_before_finetune": base_acc,
                   "acc_after_finetune": control_acc, "flops": dense_flops(base), **storage(base)}]
    for f in FILTER_FRACTIONS:
        key = f"structured_{f}"
        small = remove_filters(base, f)
        acc_before = evaluate(small, test_loader)

        def run_struct(model, src=small, key=key):
            model.load_state_dict(src.state_dict())
            set_seed()
            return {"finetune_seconds": train(key, model, loaders, FINETUNE_EPOCHS, FINETUNE_LR, log)}
        model, info = cached_model(cache, key, lambda s=small: copy.deepcopy(s), run_struct)
        structured.append({"filters_removed": f, "conv1_filters": model.conv1.out_channels,
                           "conv2_filters": model.conv2.out_channels,
                           "parameters": count_parameters(model), "acc_before_finetune": acc_before,
                           "acc_after_finetune": evaluate(model, test_loader),
                           "flops": dense_flops(model), **storage(model)})
        models_for_latency[f"structured {int(f * 100)}%"] = model
        print(f"{key}: before {acc_before:.4f}, after {structured[-1]['acc_after_finetune']:.4f}",
              flush=True)
    write_csv("structured_pruning.csv", structured)

    # 4. Distillation: the same student trained without and with the teacher
    set_seed()
    student_init = student().state_dict()

    def run_student(model, teacher=None, name="student_plain"):
        model.load_state_dict(student_init)
        set_seed()
        return {"train_seconds": train(name, model, loaders, EPOCHS, LR, log, teacher=teacher)}
    plain, plain_info = cached_model(cache, "student_plain", student, run_student)
    kd, kd_info = cached_model(cache, "student_kd", student,
                               lambda m: run_student(m, teacher=base, name="student_kd"))
    models_for_latency["student (plain)"] = plain
    models_for_latency["student (distilled)"] = kd

    # 3. Latency (all models measured together) and the sparse-kernel benchmark
    print("measuring latency...", flush=True)
    lat1 = interleaved_latency_ms(models_for_latency, 1, runs=200)
    lat128 = interleaved_latency_ms(models_for_latency, 128, runs=30)
    latency = []
    for name, model in models_for_latency.items():
        density = layer_density(model)
        latency.append({"model": name, "parameters": count_parameters(model),
                        "nonzero_weights": int(sum(density[k] * dict(model.named_parameters())[k].numel()
                                                   for k in PRUNABLE)),
                        "dense_flops": dense_flops(model), "effective_flops": effective_flops(model),
                        "latency_b1_ms": round(lat1[name], 4), "latency_b128_ms": round(lat128[name], 3),
                        "energy_mj_per_image": round(lat128[name] / 128 * CPU_WATTS, 4),
                        **{k: round(v, 1) for k, v in storage(model).items()}})
    write_csv("latency_and_size.csv", latency)
    write_csv("sparse_kernel_benchmark.csv", sparse_kernel_benchmark(base.fc1.weight))

    distill = []
    for name, model, info in (("teacher (baseline)", base, base_info),
                              ("student without teacher", plain, plain_info),
                              ("student with distillation", kd, kd_info)):
        key = {"teacher (baseline)": "baseline", "student without teacher": "student (plain)",
               "student with distillation": "student (distilled)"}[name]
        distill.append({"model": name, "parameters": count_parameters(model),
                        "test_accuracy": round(evaluate(model, test_loader), 4),
                        "dense_kb": round(storage(model)["dense_kb"], 1),
                        "latency_b1_ms": round(lat1[key], 4), "latency_b128_ms": round(lat128[key], 3),
                        "train_seconds": round(info["train_seconds"], 1)})
    write_csv("distillation.csv", distill)

    # One run per student is a single sample, so repeat both students with two more seeds
    # (new initial weights and batch order); seed 42 is the pair above.
    seed_rows = [{"seed": SEED, "plain_accuracy": distill[1]["test_accuracy"],
                  "distilled_accuracy": distill[2]["test_accuracy"]}]
    for seed in EXTRA_STUDENT_SEEDS:
        torch.manual_seed(seed)
        init = student().state_dict()

        def run_seed(model, teacher=None, name="", seed=seed, init=init):
            model.load_state_dict(init)
            torch.manual_seed(seed)
            return {"train_seconds": train(name, model, loaders, EPOCHS, LR, log,
                                           teacher=teacher, seed=seed)}
        p_model, _ = cached_model(cache, f"student_plain_s{seed}", student,
                                  lambda m, s=seed: run_seed(m, name=f"student_plain_s{s}"))
        k_model, _ = cached_model(cache, f"student_kd_s{seed}", student,
                                  lambda m, s=seed: run_seed(m, teacher=base, name=f"student_kd_s{s}"))
        seed_rows.append({"seed": seed, "plain_accuracy": round(evaluate(p_model, test_loader), 4),
                          "distilled_accuracy": round(evaluate(k_model, test_loader), 4)})
    for row in seed_rows:
        row["gain"] = round(row["distilled_accuracy"] - row["plain_accuracy"], 4)
    write_csv("distillation_seeds.csv", seed_rows)
    write_csv("training_log.csv", log)
    save_cache(cache)

    # Plots: accuracy against sparsity
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    for mode, color in (("global", "tab:blue"), ("layer", "tab:orange")):
        rows = [r for r in unstructured if r["mode"] == mode]
        xs = [r["target_sparsity"] * 100 for r in rows]
        label = "global" if mode == "global" else "layer-wise"
        ax1.plot(xs, [r["acc_after_finetune"] * 100 for r in rows], "o-", color=color,
                 label=f"{label}, after 3-epoch fine-tune")
        ax1.plot(xs, [r["acc_before_finetune"] * 100 for r in rows], "x--", color=color,
                 label=f"{label}, right after pruning")
    ax1.set_xlabel("Weights set to zero (%)")
    ax1.set_ylabel("Test accuracy (%)")
    ax1.set_title("Unstructured magnitude pruning")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8)
    xs = [r["filters_removed"] * 100 for r in structured]
    ax2.plot(xs, [r["acc_after_finetune"] * 100 for r in structured], "o-", color="tab:green",
             label="after 3-epoch fine-tune")
    ax2.plot(xs, [r["acc_before_finetune"] * 100 for r in structured], "x--", color="tab:green",
             label="right after pruning")
    for r in structured:
        ax2.annotate(f"{r['parameters'] / 1e6:.2f}M params",
                     (r["filters_removed"] * 100, r["acc_after_finetune"] * 100),
                     textcoords="offset points", xytext=(4, 6), fontsize=8)
    ax2.set_xlabel("Conv filters removed (%)")
    ax2.set_title("Structured pruning (whole filters, L2 norm)")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "sparsity_curves.png", dpi=150)
    plt.close(fig)
    print("done", flush=True)


if __name__ == "__main__":
    main()
