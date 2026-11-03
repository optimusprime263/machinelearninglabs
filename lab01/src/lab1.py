"""Lab 1: two baseline models on Breast Cancer Wisconsin and what they cost to run.

Run from the repository root: python lab01/src/lab1.py
Outputs go to lab01/results/.
"""
import csv
import random
import threading
import time
import tracemalloc
import warnings
from pathlib import Path

import joblib
import numpy as np
import psutil
from sklearn.base import clone
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
RESULTS = Path(__file__).resolve().parents[1] / "results"
MB = 1024 * 1024

MODELS = {
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=SEED),
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=SEED),
}

# Budgets from the manual. Memory is a range of what devices in each class have.
BUDGETS = {
    "Cloud":  {"memory_mb": (1024, None), "latency_ms": 100, "size_kb": 500 * 1024},
    "Edge":   {"memory_mb": (256, 1024),  "latency_ms": 50,  "size_kb": 50 * 1024},
    "Mobile": {"memory_mb": (64, 256),    "latency_ms": 20,  "size_kb": 10 * 1024},
    "TinyML": {"memory_mb": (None, 0.25), "latency_ms": 10,  "size_kb": 100},
}


def median_ms(fn, repeats):
    fn()  # warm-up, not measured
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return float(np.median(times)) * 1000


def peak_rss_mb(fn, interval=0.0005):
    """Run fn while a background thread samples this process's RSS; return the peak in MB."""
    process = psutil.Process()
    peak = process.memory_info().rss
    done = threading.Event()

    def sample():
        nonlocal peak
        while not done.is_set():
            peak = max(peak, process.memory_info().rss)
            time.sleep(interval)

    sampler = threading.Thread(target=sample)
    sampler.start()
    try:
        fn()
    finally:
        done.set()
        sampler.join()
    peak = max(peak, process.memory_info().rss)
    return peak / MB


def allocated_peak_kb(fn):
    """Peak memory that fn itself allocates (Python and NumPy objects), in KB."""
    tracemalloc.start()
    try:
        fn()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak / 1024


def fits(budget, peak_mb, latency_ms, size_kb):
    low, high = budget["memory_mb"]
    memory_any = high is None or peak_mb <= high       # fits the largest device in the class
    memory_all = low is None or peak_mb <= low          # fits even the smallest one
    if low is None:                                     # TinyML: one upper limit
        memory_all = memory_any
    latency = latency_ms <= budget["latency_ms"]
    size = size_kb <= budget["size_kb"]
    return memory_all, memory_any, latency, size, memory_any and latency and size


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    RESULTS.mkdir(exist_ok=True)

    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=SEED)
    print(f"data {X.shape}, train {X_train.shape}, test {X_test.shape}")
    one_sample = X_test[:1]

    accuracy_rows, measurement_rows, fit_rows = [], [], []
    for name, base in MODELS.items():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model = clone(base).fit(X_train, y_train)
        converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
        acc = accuracy_score(y_test, model.predict(X_test))
        accuracy_rows.append({"model": name, "test_accuracy": f"{acc:.4f}",
                              "note": "" if converged else "ConvergenceWarning: hit max_iter=1000"})

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            train_ms = median_ms(lambda: clone(base).fit(X_train, y_train), repeats=5)
            idle_mb = psutil.Process().memory_info().rss / MB
            train_peak = peak_rss_mb(lambda: clone(base).fit(X_train, y_train))
        infer_ms = median_ms(lambda: model.predict(one_sample), repeats=100)
        infer_peak = peak_rss_mb(lambda: [model.predict(one_sample) for _ in range(100)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            train_alloc = allocated_peak_kb(lambda: clone(base).fit(X_train, y_train))
        infer_alloc = allocated_peak_kb(lambda: model.predict(one_sample))

        model_path = RESULTS / f"model_{name}.joblib"
        joblib.dump(model, model_path)
        size_bytes = model_path.stat().st_size
        size_kb = size_bytes / 1024

        measurement_rows.append({
            "model": name,
            "train_time_ms": f"{train_ms:.2f}",
            "inference_ms": f"{infer_ms:.4f}",
            "size_bytes": size_bytes,
            "size_kb": f"{size_kb:.1f}",
            "process_before_mb": f"{idle_mb:.1f}",
            "peak_train_mb": f"{train_peak:.1f}",
            "peak_inference_mb": f"{infer_peak:.1f}",
            "train_allocated_kb": f"{train_alloc:.1f}",
            "inference_allocated_kb": f"{infer_alloc:.1f}",
        })
        peak = max(train_peak, infer_peak)
        for tier, budget in BUDGETS.items():
            memory_all, memory_any, latency, size, ok = fits(budget, peak, infer_ms, size_kb)
            fit_rows.append({"model": name, "tier": tier,
                             "memory_fits_smallest_device": memory_all,
                             "memory_fits_largest_device": memory_any,
                             "latency_ok": latency, "size_ok": size, "fits": ok})
        print(f"{name}: accuracy {acc:.4f}, train {train_ms:.2f} ms, inference {infer_ms:.4f} ms, "
              f"{size_bytes} bytes, peak {peak:.1f} MB")

    # Reference only, not one of the two required baselines: the same logistic
    # regression after standard scaling, to show why the unscaled one does not converge.
    scaled = make_pipeline(StandardScaler(),
                           LogisticRegression(max_iter=1000, random_state=SEED)).fit(X_train, y_train)
    accuracy_rows.append({
        "model": "reference: StandardScaler + logistic_regression",
        "test_accuracy": f"{accuracy_score(y_test, scaled.predict(X_test)):.4f}",
        "note": f"converged in {scaled[-1].n_iter_[0]} iterations"})

    write_csv(RESULTS / "baseline_accuracy.csv", accuracy_rows)
    write_csv(RESULTS / "measurements.csv", measurement_rows)
    write_csv(RESULTS / "deployment_fit.csv", fit_rows)
    for row in accuracy_rows + fit_rows:
        print(row)


if __name__ == "__main__":
    main()
