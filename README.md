# Machine learning labs

## Lab 1: environment and first system measurements

Logistic regression and random forest on Breast Cancer Wisconsin: test
accuracy, training time, single-sample latency, model size and peak memory,
and which device class (Cloud, Edge, Mobile, TinyML) each model fits.

```powershell
python lab01/src/print_versions.py
python lab01/src/lab1.py
```

See `lab01/report.md`.

## Lab 2: neural network from scratch

A NumPy implementation of a classifier for the scikit-learn digits dataset.
The network has two ReLU hidden layers and a softmax output. The experiment
compares widths 16 and 64 and learning rates 0.001, 0.01 and 0.1.

## Environment and run

The course specifies Python 3.11.8. The complete course version list is in
`requirements.txt`. Its pins conflict: `mlflow==2.14.1` requires `pyarrow<16`,
while the list pins `pyarrow==16.1.0`. A normal installation of the complete
list fails dependency resolution. The existing environment runs Lab 2, but
`pip check` reports this conflict in `lab02/results/environment_check.txt`.

For running only Lab 2 in a new environment, install its three direct dependencies:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install numpy==1.26.4 scikit-learn==1.5.0 matplotlib==3.9.0
python lab02/src/print_versions.py
python lab02/src/lab2.py
```

The script writes gradient checks, per-epoch logs, accuracy, timing and a loss
plot to `lab02/results/`. Timing depends on the machine and NumPy configuration.
See `lab02/report.md` for the results and limitations.

`print_versions.py` lists every course package. In an environment with only the
three Lab 2 dependencies, other course packages can be reported as missing.

## Lab 3: architectures and their costs

MLP, CNN and a small patch-attention model trained on Fashion-MNIST in PyTorch,
compared by accuracy, parameters, FLOPs and inference latency. Fashion-MNIST
is downloaded to `data/` on the first run (the folder is not committed).

```powershell
python lab03/src/print_versions.py
python lab03/src/lab3.py
```

Results, plots and the FLOP comparison are written to `lab03/results/`.
See `lab03/report.md`.

## Lab 4: data pipeline and quality

A leakage-free pipeline for UCI Adult: ingestion, schema validation, cleaning
fitted on the training set only, derived features, label-quality audit and a
SHA256 of the processed data. Settings are in `lab04/config.yaml`. The raw and
processed data go to `data/` (not committed).

It needs the course environment from `requirements.txt` (pandas, scikit-learn,
joblib, and PyYAML, which comes with it as a dependency). One command runs the
whole pipeline:

```powershell
python lab04/src/pipeline.py
```

See `lab04/report.md` and `lab04/data_card.md`.

## Lab 5: frameworks and automatic differentiation

A scalar autograd engine (`lab05/src/engine.py`) checked against PyTorch
autograd and numerical gradients, a 2-8-1 network trained on make_moons with
both, and PyTorch graph inspection, eager vs traced timings and tensor
contiguity. `torch.compile` is not available on Windows with PyTorch 2.3.1, so
the comparison uses `torch.jit.trace`.

```powershell
python lab05/src/print_versions.py
python lab05/src/lab5.py
```

The full run takes about 5 minutes, mostly the engine training (one run for the
comparison, then one warm-up and 5 timed runs).
See `lab05/report.md`.
