"""Lab 5: a scalar autograd engine checked against PyTorch, and PyTorch execution modes.

Run from the repository root: python lab05/src/lab5.py
Outputs go to lab05/results/.
"""
import csv
import random
import statistics
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from torch import nn
from torch._dynamo.utils import counters

from engine import TinyMLP, Value, value_sum

SEED = 42
RESULTS = Path(__file__).resolve().parents[1] / "results"
EPOCHS = 30
BATCH_SIZE = 16
LR = 1.0
TRAINING_RUNS = 5


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def median_seconds(fn, runs, warmup=1):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def median_seconds_interleaved(fns, runs, warmup=1):
    """Time several functions in turns (a, b, a, b, ...), so a change in CPU speed
    during the measurement affects all of them equally. Returns one median per function."""
    for fn in fns:
        for _ in range(warmup):
            fn()
    times = [[] for _ in fns]
    for _ in range(runs):
        for i, fn in enumerate(fns):
            start = time.perf_counter()
            fn()
            times[i].append(time.perf_counter() - start)
    return [statistics.median(t) for t in times]


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# ------------------------------------------------------- 4. Gradient check
def check_gradients():
    """((a*b + c)**2).relu().sum() on 10-element vectors: engine vs PyTorch vs numerical."""
    rng = np.random.default_rng(SEED)
    a0, b0, c0 = rng.normal(size=(3, 10))

    a = [Value(v) for v in a0]
    b = [Value(v) for v in b0]
    c = [Value(v) for v in c0]
    out = value_sum(((x * y + z) ** 2).relu() for x, y, z in zip(a, b, c))
    out.backward()

    ta, tb, tc = (torch.tensor(v, dtype=torch.float64, requires_grad=True) for v in (a0, b0, c0))
    t_out = ((ta * tb + tc) ** 2).relu().sum()
    t_out.backward()

    def f(a_, b_, c_):
        return float(np.maximum((a_ * b_ + c_) ** 2, 0).sum())

    eps = 1e-6
    rows = []
    for name, base, engine_vals, torch_grad in (("a", a0, a, ta.grad), ("b", b0, b, tb.grad),
                                                ("c", c0, c, tc.grad)):
        for i in range(len(base)):
            plus, minus = [a0.copy(), b0.copy(), c0.copy()], [a0.copy(), b0.copy(), c0.copy()]
            k = "abc".index(name)
            plus[k][i] += eps
            minus[k][i] -= eps
            numeric = (f(*plus) - f(*minus)) / (2 * eps)
            rows.append({"input": f"{name}[{i}]", "engine": engine_vals[i].grad,
                         "pytorch": float(torch_grad[i]), "numerical": numeric,
                         "diff_engine_pytorch": abs(engine_vals[i].grad - float(torch_grad[i])),
                         "diff_engine_numerical": abs(engine_vals[i].grad - numeric)})
    summary = {
        "output_engine": out.data, "output_pytorch": float(t_out),
        "max_diff_engine_pytorch": max(r["diff_engine_pytorch"] for r in rows),
        "max_diff_engine_numerical": max(r["diff_engine_numerical"] for r in rows),
        "graph_nodes": len(out.topological_order()),
    }
    return rows, summary


def check_accumulation():
    """A graph where the same nodes are used several times, so gradients must add up."""
    x, y = Value(1.5), Value(-2.0)
    d = x * y                        # d is used three times below, x four times
    e = d * d + d + x
    out = (e * x + d).relu()
    out.backward()

    tx = torch.tensor(1.5, dtype=torch.float64, requires_grad=True)
    ty = torch.tensor(-2.0, dtype=torch.float64, requires_grad=True)
    td = tx * ty
    te = td * td + td + tx
    t_out = (te * tx + td).relu()
    t_out.backward()
    return {"x_engine": x.grad, "x_pytorch": float(tx.grad),
            "y_engine": y.grad, "y_pytorch": float(ty.grad),
            "max_diff": max(abs(x.grad - float(tx.grad)), abs(y.grad - float(ty.grad)))}


# ---------------------------------------------------------- 5. make_moons
def moons_data():
    X, y = make_moons(n_samples=1000, noise=0.1, random_state=SEED)
    return train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)


def initial_weights():
    """torch.nn.Linear-style uniform init, drawn once and shared by both implementations."""
    rng = np.random.default_rng(SEED)
    k1, k2 = 1 / np.sqrt(2), 1 / np.sqrt(8)
    return (rng.uniform(-k1, k1, (8, 2)), rng.uniform(-k1, k1, 8),
            rng.uniform(-k2, k2, (1, 8)), rng.uniform(-k2, k2, 1))


def batches(n_rows):
    """The same shuffled batch order for both implementations."""
    rng = np.random.default_rng(SEED)
    for _ in range(EPOCHS):
        order = rng.permutation(n_rows)
        yield [order[i:i + BATCH_SIZE] for i in range(0, n_rows, BATCH_SIZE)]


def train_engine(X, y):
    model = TinyMLP(*initial_weights())
    params = model.parameters()
    epoch_losses = []
    for epoch_batches in batches(len(X)):
        total = 0.0
        for idx in epoch_batches:
            for p in params:
                p.grad = 0.0
            losses = []
            for i in idx:
                p_hat = model([Value(X[i, 0]), Value(X[i, 1])])
                losses.append(-(p_hat.log()) if y[i] == 1 else -((1 - p_hat).log()))
            loss = value_sum(losses) / len(idx)
            loss.backward()
            for p in params:
                p.data -= LR * p.grad
            total += loss.data * len(idx)
        epoch_losses.append(total / len(X))
    return model, epoch_losses


def torch_model(dtype=torch.float64):
    w1, b1, w2, b2 = initial_weights()
    model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1), nn.Sigmoid()).to(dtype)
    with torch.no_grad():
        for layer, w, b in ((model[0], w1, b1), (model[2], w2, b2)):
            layer.weight.copy_(torch.tensor(w))
            layer.bias.copy_(torch.tensor(b))
    return model


def train_torch(X, y):
    model = torch_model()
    Xt = torch.tensor(X, dtype=torch.float64)
    yt = torch.tensor(y, dtype=torch.float64).unsqueeze(1)
    opt = torch.optim.SGD(model.parameters(), lr=LR)
    loss_fn = nn.BCELoss()
    epoch_losses = []
    for epoch_batches in batches(len(X)):
        total = 0.0
        for idx in epoch_batches:
            idx = torch.as_tensor(idx)
            opt.zero_grad()
            loss = loss_fn(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        epoch_losses.append(total / len(X))
    return model, epoch_losses


def engine_predict(model, X):
    """Forward pass with plain numbers, using the engine's trained parameters."""
    w1 = np.array([[w.data for w in row] for row in model.hidden.weight])
    b1 = np.array([b.data for b in model.hidden.bias])
    w2 = np.array([[w.data for w in row] for row in model.out.weight])
    b2 = np.array([b.data for b in model.out.bias])
    h = np.maximum(X @ w1.T + b1, 0)
    return 1 / (1 + np.exp(-(h @ w2.T + b2)))[:, 0]


def compare_training():
    X_train, X_test, y_train, y_test = moons_data()
    engine_model, engine_losses = train_engine(X_train, y_train)
    t_model, t_losses = train_torch(X_train, y_train)

    engine_params = np.array([p.data for p in engine_model.parameters()])
    torch_params = np.concatenate([t_model[0].weight.detach().numpy().ravel(),
                                   t_model[0].bias.detach().numpy(),
                                   t_model[2].weight.detach().numpy().ravel(),
                                   t_model[2].bias.detach().numpy()])
    with torch.no_grad():
        t_prob = t_model(torch.tensor(X_test, dtype=torch.float64)).numpy()[:, 0]
    e_prob = engine_predict(engine_model, X_test)

    engine_time = median_seconds(lambda: train_engine(X_train, y_train), TRAINING_RUNS)
    torch_time = median_seconds(lambda: train_torch(X_train, y_train), TRAINING_RUNS)

    summary = {
        "train_rows": len(X_train), "test_rows": len(X_test),
        "engine_test_accuracy": float(((e_prob > 0.5) == y_test).mean()),
        "pytorch_test_accuracy": float(((t_prob > 0.5) == y_test).mean()),
        "engine_final_loss": engine_losses[-1], "pytorch_final_loss": t_losses[-1],
        "max_loss_diff": max(abs(a - b) for a, b in zip(engine_losses, t_losses)),
        "max_param_diff": float(np.abs(engine_params - torch_params).max()),
        "engine_train_seconds": engine_time, "pytorch_train_seconds": torch_time,
        "speed_ratio": engine_time / torch_time,
    }
    plot_training(engine_losses, t_losses, engine_model, X_test, y_test)
    rows = [{"epoch": i + 1, "engine_loss": a, "pytorch_loss": b}
            for i, (a, b) in enumerate(zip(engine_losses, t_losses))]
    return rows, summary, t_model, X_test


def plot_training(engine_losses, torch_losses, model, X_test, y_test):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    epochs = range(1, len(engine_losses) + 1)
    ax1.plot(epochs, torch_losses, color="tab:orange", linewidth=3, label="PyTorch")
    ax1.plot(epochs, engine_losses, "o", color="tab:blue", markersize=4, label="my engine")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training BCE loss")
    ax1.set_title("Same weights, same batches: the curves overlap")
    ax1.legend()
    ax1.grid(alpha=0.3)

    xx, yy = np.meshgrid(np.linspace(-1.5, 2.5, 200), np.linspace(-1.0, 1.5, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = engine_predict(model, grid).reshape(xx.shape)
    ax2.contourf(xx, yy, zz, levels=[0, 0.5, 1], colors=["#c6dbef", "#fdd0a2"], alpha=0.8)
    ax2.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap="coolwarm", s=12, edgecolors="k",
                linewidths=0.3)
    ax2.set_title("Decision boundary of the engine model (test points)")
    fig.tight_layout()
    fig.savefig(RESULTS / "moons_training.png", dpi=150)
    plt.close(fig)


# ----------------------------------------------- 6. PyTorch graph and modes
def describe_graph(model, X_test):
    lines = ["Autograd graph of BCE loss for the trained 2-8-1 model (4 test rows)", ""]
    x = torch.tensor(X_test[:4], dtype=torch.float64)
    loss = nn.BCELoss()(model(x), torch.ones(4, 1, dtype=torch.float64))

    def walk(fn, depth, seen):
        if fn is None:
            return
        name = type(fn).__name__
        if hasattr(fn, "variable"):          # a leaf: the parameter that receives the gradient
            name += f" -> parameter {tuple(fn.variable.shape)}"
        if id(fn) in seen:
            lines.append("  " * depth + name + "  (already shown)")
            return
        seen[id(fn)] = fn                    # keep a reference so the id is not reused
        lines.append("  " * depth + name)
        for next_fn, _ in fn.next_functions:
            walk(next_fn, depth + 1, seen)
    walk(loss.grad_fn, 0, {})

    lines += ["", "torch.fx graph of the model's forward pass:", ""]
    lines += str(torch.fx.symbolic_trace(model).graph).splitlines()
    lines += ["", "Parameters:"]
    for name, p in model.named_parameters():
        lines.append(f"  {name:<10} shape {tuple(p.shape)}, dtype {p.dtype}, "
                     f"stride {p.stride()}, contiguous {p.is_contiguous()}")
    return "\n".join(lines) + "\n"


def try_torch_compile(fn, x):
    """Returns the compiled function, or None and the reason it cannot be used here."""
    torch._dynamo.reset()
    counters.clear()
    try:
        compiled = torch.compile(fn, fullgraph=True)
        with torch.no_grad():
            compiled(x)
    except Exception as e:                               # inductor does not work on Windows here
        root = e
        while root.__cause__ is not None or root.__context__ is not None:
            root = root.__cause__ or root.__context__
        first = " ".join(str(e).split())[:60]
        return None, (f"{type(e).__name__} ({first}...); root cause: "
                      f"{type(root).__name__}: {' '.join(str(root).split())[:110]}")
    graphs = counters["stats"]["unique_graphs"]
    if graphs == 0:
        return None, "torch.compile returned, but captured no graph, so it ran eagerly"
    return compiled, f"torch.compile works ({graphs} graph captured)"


class SmallMLP(nn.Module):
    """The trained 2-8-1 network, wrapped in a module whose forward is defined here."""

    def __init__(self):
        super().__init__()
        self.net = torch_model(torch.float32)

    def forward(self, x):
        return self.net(x)


class BigMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(784, 1024), nn.ReLU(), nn.Linear(1024, 1024),
                                 nn.ReLU(), nn.Linear(1024, 10))

    def forward(self, x):
        return self.net(x)


def elementwise_chain(x):
    return torch.sigmoid(x) * x + torch.tanh(x) * 0.5 - x * x


def compare_modes():
    small = SmallMLP().eval()
    cases = [
        ("2-8-1 MLP, batch 1", small, torch.randn(1, 2), 2000),
        ("2-8-1 MLP, batch 1024", small, torch.randn(1024, 2), 2000),
        ("784-1024-1024-10 MLP, batch 256", BigMLP().eval(), torch.randn(256, 784), 100),
        ("7 element-wise ops, 1M values", elementwise_chain, torch.randn(1_000_000), 100),
    ]
    compiled, compile_note = try_torch_compile(small, cases[0][2])
    rows = []
    with torch.no_grad():
        for name, fn, x, runs in cases:
            traced = torch.jit.trace(fn, x)
            same = torch.allclose(fn(x), traced(x), atol=1e-6)
            # TorchScript optimises the graph during its first runs, so 3 warm-up calls
            # each; eager and traced calls alternate
            eager, trace = median_seconds_interleaved([lambda: fn(x), lambda: traced(x)],
                                                      runs, warmup=3)
            row = {"case": name, "runs": runs, "eager_us": eager * 1e6,
                   "traced_us": trace * 1e6, "speedup": eager / trace, "same_output": same}
            if compiled is not None:
                comp = torch.compile(fn)
                row["compiled_us"] = median_seconds(lambda: comp(x), runs, warmup=3) * 1e6
            rows.append(row)
    return rows, compile_note


def tensor_layout():
    x = torch.randn(4096, 4096)
    xt = x.t()
    xc = xt.contiguous()
    lines = []
    for name, t in (("x", x), ("x.t()", xt), ("x.t().contiguous()", xc)):
        lines.append(f"{name:<20} shape {tuple(t.shape)}, dtype {t.dtype}, stride {t.stride()}, "
                     f"contiguous {t.is_contiguous()}, shares memory with x: "
                     f"{t.data_ptr() == x.data_ptr()}")
    try:
        xt.view(-1)
    except RuntimeError as e:
        lines.append(f"x.t().view(-1) fails: {str(e).splitlines()[0][:110]}")

    y = torch.randn(4096, 4096)
    w = torch.randn(4096, 256)
    ops = [
        ("t * 2 (one input)", lambda t: t * 2),
        ("t + y (y contiguous)", lambda t: t + y),
        ("t.sum(dim=1)", lambda t: t.sum(dim=1)),
        ("t @ w (4096x256)", lambda t: t @ w),
        ("t.contiguous() (copy)", lambda t: t.contiguous()),
    ]
    rows = []
    for name, op in ops:
        non_contig, contig = median_seconds_interleaved([lambda: op(xt), lambda: op(xc)], 30)
        rows.append({"operation": name, "non_contiguous_ms": non_contig * 1e3,
                     "contiguous_ms": contig * 1e3, "ratio": non_contig / contig})
    return "\n".join(lines) + "\n", rows


def main():
    set_seed()
    RESULTS.mkdir(exist_ok=True)
    print(f"torch {torch.__version__}, {torch.get_num_threads()} threads")

    rows, summary = check_gradients()
    write_csv(RESULTS / "gradient_check.csv", rows)
    acc = check_accumulation()
    write_csv(RESULTS / "gradient_check_summary.csv",
              [{**summary, **{f"accumulation_{k}": v for k, v in acc.items()}}])
    print("gradient check:", summary)
    print("accumulation check:", acc)
    assert summary["max_diff_engine_pytorch"] < 1e-6 and summary["max_diff_engine_numerical"] < 1e-6
    assert acc["max_diff"] < 1e-6

    rows, summary, t_model, X_test = compare_training()
    write_csv(RESULTS / "moons_losses.csv", rows)
    write_csv(RESULTS / "moons_summary.csv", [summary])
    print("training:", summary)

    (RESULTS / "torch_graph.txt").write_text(describe_graph(t_model, X_test), encoding="utf-8")
    rows, compile_note = compare_modes()
    write_csv(RESULTS / "execution_modes.csv", rows)
    (RESULTS / "torch_compile_attempt.txt").write_text(compile_note + "\n", encoding="utf-8")
    print("torch.compile:", compile_note)
    for r in rows:
        print(r)

    text, rows = tensor_layout()
    (RESULTS / "tensor_layout.txt").write_text(text, encoding="utf-8")
    write_csv(RESULTS / "contiguity_timing.csv", rows)
    print(text)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
