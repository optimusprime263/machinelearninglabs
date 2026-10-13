# Machine learning labs

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
