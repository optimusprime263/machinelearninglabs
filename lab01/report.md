# Lab 1: environment and first system measurements

## Goal

Set up an isolated Python environment with the pinned course packages, train
two baseline models on the Breast Cancer Wisconsin data, measure what they
cost to train and run (time, latency, size, memory), and decide on which
kind of device each model could run: Cloud, Edge, Mobile or TinyML.

## Method

**Environment.**

- Python 3.11.8 in a virtual environment `.venv` at the repository root.
  It is isolated: its `sys.prefix` points inside `.venv`, not to the system
  Python.
- The package versions are pinned in `requirements.txt`, exactly as listed in
  the manual.
- The list cannot be installed in one step: `mlflow==2.14.1` requires
  `pyarrow<16`, but the list pins `pyarrow==16.1.0`, so pip stops with
  ResolutionImpossible. I installed every other pinned package first, then
  mlflow without its dependencies, then mlflow's remaining dependencies.
- `pip check` now reports only this one conflict
  (`results/environment_check.txt`).
- `src/print_versions.py` prints Python and every package from
  `requirements.txt` with the installed and the pinned version; the output is
  in `results/versions.txt`.
- The repository follows the layout from the manual: one folder per lab with
  `report.md`, `src/` and `results/`, and `data/` ignored by git.

**Data and models.** `load_breast_cancer(return_X_y=True)`: 569 tumour
samples, 30 numeric features, 62.7 % benign. Split 70/30, stratified,
`random_state=42`: 398 training and 171 test samples. Two baselines, with the
exact settings from the manual:

- `LogisticRegression(max_iter=1000, random_state=42)`
- `RandomForestClassifier(n_estimators=100, random_state=42)`

**Measurements** (`src/lab1.py`, laptop CPU: Ryzen 5 5600U):

- **Training time:** a fresh copy of the model is fitted 5 times after
  1 warm-up fit; the median is reported.
- **Single-sample inference:** `predict` on one test sample, 100 times after
  a warm-up; the median is reported.
- **Model size:** the trained model is saved with `joblib.dump`, and the file
  size is reported in bytes and KB. There are two models, so there are two
  files: `results/model_logistic_regression.joblib` and
  `results/model_random_forest.joblib`.
- **Peak memory:** a background thread reads
  `psutil.Process().memory_info().rss` every 0.5 ms while the model trains,
  and again while it makes 100 single-sample predictions. The highest value
  is the peak memory of the whole Python process. I also recorded, with
  `tracemalloc`, how much memory the training and the prediction calls
  allocate themselves.

**Budget rule.** A model fits a class of device if all three hold:

- its peak memory is not more than what the largest device in that class
  has;
- its single-sample latency is within the limit;
- its file size is within the limit.

The table also shows whether the memory would fit the smallest device in the
class (for example 64 MB for Mobile).

## Results

### Environment

Python 3.11.8, numpy 1.26.4, scikit-learn 1.5.0, psutil 5.9.8, joblib 1.4.2.
Every package is installed at its pinned version; the full list is in
`results/versions.txt` (thop reports `0.1.1.post2209072238` as
`0.1.1-2209072238`, the same release).

### Accuracy (`results/baseline_accuracy.csv`)

| Model | Test accuracy | Note |
|---|---|---|
| Logistic regression | 0.9415 | ConvergenceWarning: stopped at max_iter=1000 |
| Random forest | 0.9357 | |
| *Reference: StandardScaler + logistic regression* | *0.9883* | *converged in 19 iterations; not one of the required baselines* |

The logistic regression from the manual does not converge. The features have
very different scales (some below 1, some in the thousands), and lbfgs needs
more than 1000 steps. I kept the required settings. The reference row shows
that the same model with standard scaling converges in 19 iterations and is
4.7 points more accurate.

### Cost measurements (`results/measurements.csv`)

| Model | Training time (median of 5) | Single-sample latency (median of 100) | Model size | Peak RSS, training | Peak RSS, inference | Allocated by training | Allocated by one prediction |
|---|---|---|---|---|---|---|---|
| Logistic regression | 573.65 ms | 0.1161 ms | 1,055 bytes (1.0 KB) | 105.0 MB | 105.0 MB | 67.3 KB | 1.7 KB |
| Random forest | 263.68 ms | 4.8436 ms | 290,889 bytes (284.1 KB) | 106.3 MB | 106.3 MB | 224.4 KB | 12.9 KB |

- **Training time:** the logistic regression is slower to train than the
  forest only because it runs all 1000 iterations without converging.
- **Latency:** the forest is 42 times slower per prediction, because it has to
  walk 100 trees.
- **Memory:** almost all of the ~105 MB is the Python process itself (the
  interpreter, NumPy and scikit-learn), not the models. Training and
  prediction allocate only kilobytes on top.

### Resource budget table (`results/deployment_fit.csv`)

| Model | Tier | Memory: largest device | Memory: smallest device | Latency | Size | Fits |
|---|---|---|---|---|---|---|
| Logistic regression | Cloud (≥ 1 GB, ≤ 100 ms, ≤ 500 MB) | yes | yes | yes | yes | **yes** |
| Logistic regression | Edge (256-1024 MB, ≤ 50 ms, ≤ 50 MB) | yes | yes | yes | yes | **yes** |
| Logistic regression | Mobile (64-256 MB, ≤ 20 ms, ≤ 10 MB) | yes | no | yes | yes | **yes** |
| Logistic regression | TinyML (≤ 256 KB, ≤ 10 ms, ≤ 100 KB) | no | no | yes | yes | **no** |
| Random forest | Cloud | yes | yes | yes | yes | **yes** |
| Random forest | Edge | yes | yes | yes | yes | **yes** |
| Random forest | Mobile | yes | no | yes | yes | **yes** |
| Random forest | TinyML | no | no | yes | no | **no** |

### Where each model can run

**Logistic regression: Cloud, Edge and Mobile; TinyML only if exported.**

- *Cloud and Edge:* it fits with a large margin. 105 MB of memory is below
  even the smallest Edge device (256 MB), 0.12 ms is far below 50 ms, and 1 KB
  is far below 50 MB.
- *Mobile:* it fits devices with 128-256 MB of memory, but not a 64 MB one
  as a full Python process.
- *TinyML:* as measured it does not fit, because the Python process needs
  about 105 MB against a 256 KB limit. The model itself is tiny: 30 weights
  and 1 bias, which is 124 bytes as float32. One prediction allocates 1.7 KB
  and costs 30 multiply-adds. Exported to C code without Python, it would fit
  a microcontroller easily.

**Random forest: Cloud, Edge and Mobile, but not TinyML.**

- *Cloud, Edge and Mobile:* these pass for the same reasons, and its 4.8 ms
  latency is within the 20 ms Mobile limit.
- *TinyML:* it fails twice. The saved model is 284 KB, almost 3 times the
  100 KB limit, and the Python process is far above 256 KB. Even exported to
  C, 100 trees would be too large without shrinking the forest.

**Caveat.** The latencies were measured on a laptop CPU. A phone is several
times slower, and a microcontroller is slower by orders of magnitude. The
forest's 4.8 ms could become tens of milliseconds on a phone, close to the
20 ms Mobile limit. The logistic regression's 30 multiply-adds stay cheap
anywhere.

## Three conclusions

1. **On this data the simple model is enough.** Logistic regression reaches
   0.9415 against the forest's 0.9357 (0.9883 with scaling), is 276 times
   smaller and 42 times faster per prediction.
2. **The runtime costs more than the model.** Both models need about 105 MB
   only because they run inside Python with NumPy and scikit-learn. The
   models allocate kilobytes. For Mobile and TinyML, the deployment format
   matters as much as the model itself.
3. **Measuring correctly changes the conclusions.** Warm-up and medians give
   stable times. Without the ConvergenceWarning check, the slow training of
   the logistic regression would look like a property of the model, while it
   actually comes from unscaled features.
