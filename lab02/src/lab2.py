"""Lab 2: a two-hidden-layer neural network from scratch (NumPy only).

Architecture: 64 -> W -> ReLU -> W -> ReLU -> 10 -> softmax, W in {16, 64}.
Run from the repository root: python lab02/src/lab2.py
Outputs go to lab02/results/ (loss_curves.png, gradient_check.txt, accuracy.md,
timing.txt, training_log.txt).
"""
import random
import time
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

RESULTS = Path(__file__).resolve().parents[1] / "results"
SEED = 42
INIT_STD = 0.1             # N(0, 0.01) = N(mean, variance) -> std = sqrt(0.01)
GRADCHECK_N = 100          # examples used by the gradient check
WIDTHS = [16, 64]
LEARNING_RATES = [0.001, 0.01, 0.1]
BATCH_SIZE = 32
EPOCHS = 50


# ---------------------------------------------------------------- 1. Dataset
def load_data():
    digits = load_digits()
    X = digits.data / 16.0                       # pixels 0..16 -> 0..1
    y = digits.target
    Y = np.eye(10)[y]                            # one-hot, shape (n, 10)
    X_tr, X_te, Y_tr, Y_te, y_tr, y_te = train_test_split(
        X, Y, y, test_size=0.3, stratify=y, random_state=SEED)
    return X_tr, X_te, Y_tr, Y_te, y_tr, y_te


# ----------------------------------------------------- 2. Network parameters
def init_params(width, rng):
    return {
        "W1": rng.normal(0.0, INIT_STD, (64, width)),
        "b1": np.zeros(width),
        "W2": rng.normal(0.0, INIT_STD, (width, width)),
        "b2": np.zeros(width),
        "W3": rng.normal(0.0, INIT_STD, (width, 10)),
        "b3": np.zeros(10),
    }


# ------------------------------------------------------------ 3. Forward pass
def relu(z):
    return np.maximum(0.0, z)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)         # numerical stability
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def forward(params, X):
    z1 = X @ params["W1"] + params["b1"]         # (batch, W)
    a1 = relu(z1)
    z2 = a1 @ params["W2"] + params["b2"]        # (batch, W)
    a2 = relu(z2)
    z3 = a2 @ params["W3"] + params["b3"]        # (batch, 10) logits
    p = softmax(z3)                              # (batch, 10) probabilities
    cache = {"X": X, "z1": z1, "a1": a1, "z2": z2, "a2": a2, "z3": z3, "p": p}
    return p, cache


# ------------------------------------------------------ 4. Cross-entropy loss
def cross_entropy(z3, Y):
    """Mean of -sum(y * log softmax(z)) over the batch, via log-sum-exp."""
    z = z3 - z3.max(axis=1, keepdims=True)
    log_p = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    return -np.mean(np.sum(Y * log_p, axis=1))


def loss_fn(params, X, Y):
    _, cache = forward(params, X)
    return cross_entropy(cache["z3"], Y)


# ----------------------------------------------------------- 5. Backward pass
def backward(params, cache, Y):
    n = Y.shape[0]
    dz3 = (cache["p"] - Y) / n                   # dL/dz3 for softmax + CE
    dW3 = cache["a2"].T @ dz3
    db3 = dz3.sum(axis=0)
    da2 = dz3 @ params["W3"].T
    dz2 = da2 * (cache["z2"] > 0)                # ReLU'
    dW2 = cache["a1"].T @ dz2
    db2 = dz2.sum(axis=0)
    da1 = dz2 @ params["W2"].T
    dz1 = da1 * (cache["z1"] > 0)
    dW1 = cache["X"].T @ dz1
    db1 = dz1.sum(axis=0)
    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "W3": dW3, "b3": db3}


def gradient_check(params, X, Y, rng, n_checks=10, eps=1e-5, tol=1e-6):
    """Central differences on n_checks random weights, compared to backprop.

    Weights whose gradient is exactly 0 (a pixel that is always 0, or a dead
    ReLU unit on this batch) are redrawn: 0 == 0 would pass without testing
    anything. The same weight is never checked twice.
    """
    _, cache = forward(params, X)
    grads = backward(params, cache, Y)
    rows, seen = [], set()
    while len(rows) < n_checks:
        name = str(rng.choice(["W1", "W2", "W3"]))
        W = params[name]
        idx = tuple(int(rng.integers(s)) for s in W.shape)
        if grads[name][idx] == 0 or (name, idx) in seen:
            continue
        seen.add((name, idx))
        old = W[idx]
        W[idx] = old + eps
        loss_plus = loss_fn(params, X, Y)
        W[idx] = old - eps
        loss_minus = loss_fn(params, X, Y)
        W[idx] = old
        numeric = (loss_plus - loss_minus) / (2 * eps)
        analytic = grads[name][idx]
        denom = max(abs(analytic) + abs(numeric), 1e-12)
        rel_err = abs(analytic - numeric) / denom
        rows.append((name, idx, analytic, numeric, rel_err, rel_err < tol))
    return rows


# ---------------------------------------------------------- 6. Training loop
def accuracy(params, X, y):
    p, _ = forward(params, X)
    return float(np.mean(p.argmax(axis=1) == y))


def train(width, lr, data, log):
    X_tr, X_te, Y_tr, Y_te, y_tr, y_te = data
    rng = np.random.default_rng(SEED)            # same init for a given width
    params = init_params(width, rng)
    n = X_tr.shape[0]
    history = {"train_loss": [], "test_loss": [], "test_acc": []}
    for epoch in range(1, EPOCHS + 1):
        order = rng.permutation(n)               # shuffle every epoch
        for start in range(0, n, BATCH_SIZE):
            batch = order[start:start + BATCH_SIZE]
            _, cache = forward(params, X_tr[batch])
            grads = backward(params, cache, Y_tr[batch])
            for k in params:
                params[k] -= lr * grads[k]
        train_loss = loss_fn(params, X_tr, Y_tr)
        test_loss = loss_fn(params, X_te, Y_te)
        test_acc = accuracy(params, X_te, y_te)
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        log(f"W={width:<3} lr={lr:<6} epoch {epoch:>2}/{EPOCHS}  "
            f"train_loss={train_loss:.4f}  test_loss={test_loss:.4f}  "
            f"test_acc={test_acc:.4f}")
    return params, history


def plot_loss_curves(histories, path):
    fig, axes = plt.subplots(1, len(WIDTHS), figsize=(12, 4.5), sharey=True)
    epochs = np.arange(1, EPOCHS + 1)
    colors = {0.001: "tab:blue", 0.01: "tab:orange", 0.1: "tab:green"}
    for ax, width in zip(axes, WIDTHS):
        for lr in LEARNING_RATES:
            h = histories[(width, lr)]
            ax.plot(epochs, h["train_loss"], color=colors[lr],
                    label=f"lr={lr} train")
            ax.plot(epochs, h["test_loss"], color=colors[lr], linestyle="--",
                    label=f"lr={lr} test")
        ax.axhline(np.log(10), color="grey", linewidth=0.8, linestyle=":")
        ax.set_title(f"Hidden width W = {width}")
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Cross-entropy loss")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Loss curves, mini-batch SGD (batch 32, 50 epochs)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ 7. Timing
def median_times(functions, runs=100):
    """Time each function separately, alternating the measurement order."""
    for fn in functions:
        fn()                                    # one warm-up per function
    times = np.empty((runs, len(functions)))
    for run in range(runs):
        for offset in range(len(functions)):
            index = (run + offset) % len(functions)
            t0 = time.perf_counter()
            functions[index]()
            times[run, index] = time.perf_counter() - t0
    return tuple(float(value) for value in np.median(times, axis=0))


def matmul_flops(width, n):
    fwd = 2 * n * (64 * width + width * width + width * 10)
    # backward: dW for all 3 layers (same size as forward) + da for layers 3, 2
    bwd = fwd + 2 * n * (width * 10 + width * width)
    return fwd, bwd


def time_passes(width, X, Y):
    """Median time of the full passes and of their matrix products alone."""
    params = init_params(width, np.random.default_rng(SEED))
    _, c = forward(params, X)
    dz3 = (c["p"] - Y) / Y.shape[0]
    dz2 = (dz3 @ params["W3"].T) * (c["z2"] > 0)
    dz1 = (dz2 @ params["W2"].T) * (c["z1"] > 0)

    def forward_matmuls():                       # 3 products
        X @ params["W1"]
        c["a1"] @ params["W2"]
        c["a2"] @ params["W3"]

    def backward_matmuls():                      # 5 products
        c["a2"].T @ dz3
        dz3 @ params["W3"].T
        c["a1"].T @ dz2
        dz2 @ params["W2"].T
        X.T @ dz1

    return median_times((lambda: forward(params, X),
                         lambda: backward(params, c, Y),
                         forward_matmuls,
                         backward_matmuls))


# -------------------------------------------------------------------- main
def main():
    random.seed(SEED)
    np.random.seed(SEED)
    RESULTS.mkdir(exist_ok=True)
    log_lines = []

    def log(line=""):
        print(line)
        log_lines.append(line)

    data = load_data()
    X_tr, X_te, Y_tr, Y_te, y_tr, y_te = data
    log(f"[1] train {X_tr.shape}, test {X_te.shape}, X range "
        f"[{X_tr.min():.1f}, {X_tr.max():.1f}], Y one-hot {Y_tr.shape}")
    log("    class counts train: " + str(np.bincount(y_tr).tolist()))
    log("    class counts test:  " + str(np.bincount(y_te).tolist()))

    # 2 + 3 + 4: shapes, forward check, untrained loss
    for width in WIDTHS:
        params = init_params(width, np.random.default_rng(SEED))
        shapes = {k: v.shape for k, v in params.items()}
        n_params = sum(v.size for v in params.values())
        p, cache = forward(params, X_tr)
        row_err = np.abs(p.sum(axis=1) - 1).max()
        assert p.shape == (X_tr.shape[0], 10)
        assert row_err < 1e-6
        loss0 = cross_entropy(cache["z3"], Y_tr)
        log(f"[2] W={width}: shapes {shapes}, {n_params} parameters")
        log(f"[3] W={width}: output {p.shape}, max |row sum - 1| = {row_err:.2e}")
        log(f"[4] W={width}: untrained loss = {loss0:.4f} (ln 10 = {np.log(10):.4f})")

    # 5: gradient check at initialisation, on GRADCHECK_N training examples
    check_lines = []
    for width in WIDTHS:
        params = init_params(width, np.random.default_rng(SEED))
        rows = gradient_check(params, X_tr[:GRADCHECK_N], Y_tr[:GRADCHECK_N],
                              np.random.default_rng(SEED + width))
        passed = sum(r[5] for r in rows)
        check_lines.append(f"Width W={width}: {passed}/10 weights pass "
                           f"(relative error < 1e-6, need >= 9)")
        check_lines.append(f"{'param':<6}{'index':<10}{'analytic':>15}"
                           f"{'numerical':>15}{'rel. error':>12}  ok")
        for name, idx, a, num, err, ok in rows:
            check_lines.append(f"{name:<6}{str(idx):<10}{a:>15.6e}{num:>15.6e}"
                               f"{err:>12.2e}  {'yes' if ok else 'NO'}")
        check_lines.append("")
        assert passed >= 9, f"gradient check failed for W={width}"
    (RESULTS / "gradient_check.txt").write_text("\n".join(check_lines))
    log("[5] gradient check")
    for line in check_lines:
        log("    " + line)

    # 6: training
    histories, acc_rows = {}, []
    for width in WIDTHS:
        for lr in LEARNING_RATES:
            params, h = train(width, lr, data, log)
            histories[(width, lr)] = h
            acc_rows.append((width, lr, h["train_loss"][-1], h["test_loss"][-1],
                             h["test_acc"][-1]))
    plot_loss_curves(histories, RESULTS / "loss_curves.png")
    acc_md = ["| Width W | Learning rate | Final train loss | Final test loss "
              "| Final test accuracy |", "|---|---|---|---|---|"]
    for width, lr, trl, tel, acc in acc_rows:
        acc_md.append(f"| {width} | {lr} | {trl:.4f} | {tel:.4f} | {acc:.2%} |")
    (RESULTS / "accuracy.md").write_text("\n".join(acc_md) + "\n")
    log("[6] final results")
    for line in acc_md:
        log("    " + line)

    # 7: timing, median of 100 runs
    timing = ["| Width W | Batch | Forward (ms) | Backward (ms) | Backward / forward "
              "| Forward matmuls (ms, share) | Backward matmuls (ms, share) "
              "| Forward MFLOP | Backward MFLOP |",
              "|---|---|---|---|---|---|---|---|---|"]
    for width in WIDTHS:
        for name, X, Y in [("32", X_tr[:32], Y_tr[:32]),
                           (f"{X_tr.shape[0]} (full train)", X_tr, Y_tr)]:
            t_fwd, t_bwd, m_fwd, m_bwd = time_passes(width, X, Y)
            f_fwd, f_bwd = matmul_flops(width, X.shape[0])
            timing.append(f"| {width} | {name} | {t_fwd * 1e3:.4f} | "
                          f"{t_bwd * 1e3:.4f} | {t_bwd / t_fwd:.2f} | "
                          f"{m_fwd * 1e3:.4f} ({m_fwd / t_fwd:.0%}) | "
                          f"{m_bwd * 1e3:.4f} ({m_bwd / t_bwd:.0%}) | "
                          f"{f_fwd / 1e6:.2f} | {f_bwd / 1e6:.2f} |")
    (RESULTS / "timing.txt").write_text("\n".join(timing) + "\n")
    log("[7] timing (median of 100 runs)")
    for line in timing:
        log("    " + line)

    (RESULTS / "training_log.txt").write_text("\n".join(log_lines) + "\n")


if __name__ == "__main__":
    main()
