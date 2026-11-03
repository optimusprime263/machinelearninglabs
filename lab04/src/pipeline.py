"""Lab 4: data pipeline for UCI Adult, from raw data to a hashed processed dataset.

One command, from the repository root:  python lab04/src/pipeline.py
Settings come from lab04/config.yaml. Raw and processed data go to data/
(not committed); reports, the fitted preprocessor and the hashes go to lab04/results/.
"""
import hashlib
import json
import logging
import platform
import random
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from cleaning import CATEGORICAL, build_preprocessor
from label_quality import audit_sample, duplicate_conflicts
from validation import run_validation, to_markdown

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "lab04" / "config.yaml"
log = logging.getLogger("pipeline")


def sha256_of_frame(df):
    data = df.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(data).hexdigest(), data


# ------------------------------------------------------------ 2. Ingestion
def ingest(cfg):
    """Read the raw table exactly as OpenML gives it. Nothing is changed here."""
    bunch = fetch_openml(name=cfg["data"]["openml_name"], version=cfg["data"]["openml_version"],
                         as_frame=True, data_home=str(ROOT / cfg["data"]["data_home"]))
    raw = bunch.frame
    rows, cols = raw.shape
    log.info("Ingestion: %d rows, %d columns (OpenML id %s)", rows, cols, bunch.details["id"])
    assert (rows, cols) == (cfg["data"]["expected_rows"], cfg["data"]["expected_columns"])
    return raw


def with_label_name(raw, cfg):
    """Work on a copy that calls the label 'income', as in the schema. The raw frame stays as is."""
    return raw.rename(columns={cfg["data"]["raw_label_column"]: cfg["data"]["label_column"]})


# ------------------------------------------------------------ 5. Evaluation
def evaluate(cfg, train, test, with_features):
    label, positive = cfg["data"]["label_column"], cfg["data"]["positive_label"]
    pre = build_preprocessor(cfg, with_features=with_features)
    X_train = pre.fit_transform(train.drop(columns=label))     # fit on train only
    X_test = pre.transform(test.drop(columns=label))
    y_train = (train[label].astype(object) == positive).astype(int)
    y_test = (test[label].astype(object) == positive).astype(int)
    model = LogisticRegression(max_iter=cfg["model"]["max_iter"], random_state=cfg["seed"])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return pre, X_train, X_test, y_train, y_test, {
        "features": "with derived features" if with_features else "without derived features",
        "n_columns": int(X_train.shape[1]),
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "f1": round(float(f1_score(y_test, pred)), 4),
    }


def fitted_statistics(pre):
    columns = pre.named_steps["columns"]
    num = columns.named_transformers_["num"]
    cat = columns.named_transformers_["cat"]
    names = list(num.feature_names_in_)
    return {
        "numeric_median": dict(zip(names, map(float, num.named_steps["impute"].statistics_))),
        "clip_low": dict(zip(names, map(float, num.named_steps["clip"].lower_))),
        "clip_high": dict(zip(names, map(float, num.named_steps["clip"].upper_))),
        "categorical_mode": dict(zip(CATEGORICAL, map(str, cat.named_steps["impute"].statistics_))),
    }


def main():
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    results = ROOT / cfg["outputs"]["results_dir"]
    processed_dir = ROOT / cfg["outputs"]["processed_dir"]
    results.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(results / "pipeline.log", mode="w", encoding="utf-8")])
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    # 2. Ingestion
    raw = ingest(cfg)
    raw_hash, _ = sha256_of_frame(raw)
    log.info("Raw data SHA256: %s", raw_hash)
    df = with_label_name(raw, cfg)
    label = cfg["data"]["label_column"]

    # 3. Validation (report only, no row is dropped)
    checks = run_validation(df, cfg)
    (results / "validation_report.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    (results / "validation_report.md").write_text(to_markdown(checks, len(df)), encoding="utf-8")
    for c in checks:
        if c["violations"]:
            log.info("Validation: %s / %s: %d (%s)", c["check"], c["column"], c["violations"], c["detail"])
    log.info("Validation: %d checks run, report in validation_report.md", len(checks))

    # 4. Split first, then clean the training set only
    train, test = train_test_split(df, test_size=cfg["split"]["test_size"],
                                   stratify=df[label], random_state=cfg["seed"])
    log.info("Split: train %d rows, test %d rows", len(train), len(test))
    test_before = test.copy()
    n_train = len(train)
    if cfg["cleaning"]["drop_train_duplicates"]:
        train = train.drop_duplicates()
    log.info("Duplicates: dropped %d exact duplicates from train (%d -> %d); test untouched",
             n_train - len(train), n_train, len(train))
    leakage = {
        "shared_row_ids": len(train.index.intersection(test.index)),
        "test_rows_identical_to_a_train_row": int(
            test.merge(train.drop_duplicates(), how="inner", on=list(df.columns)).shape[0]),
        "preprocessor_fitted_on_rows": len(train),
    }
    log.info("Leakage checks: %d shared row ids; %d test rows are exact copies of a train row "
             "(kept, the test set is not changed)",
             leakage["shared_row_ids"], leakage["test_rows_identical_to_a_train_row"])

    # 4-5. Fit the preprocessor on train, compare with and without derived features
    comparison = []
    for with_features in (False, True):
        pre, X_train, X_test, y_train, y_test, scores = evaluate(cfg, train, test, with_features)
        comparison.append(scores)
        log.info("LogisticRegression %s: accuracy %.4f, F1 %.4f",
                 scores["features"], scores["accuracy"], scores["f1"])
    pd.DataFrame(comparison).to_csv(results / "feature_comparison.csv", index=False)
    assert test.equals(test_before), "the test set must not change"
    leakage["test_unchanged"] = True

    # The loop ends with the version that has the derived features; that one is saved.
    joblib.dump(pre, results / "preprocessor.joblib")
    stats = fitted_statistics(pre)
    (results / "fitted_statistics.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    log.info("Saved fitted preprocessor (with derived features) to preprocessor.joblib")

    # 6. Label quality
    conflicts, n_groups = duplicate_conflicts(df, label)
    conflicts.to_csv(results / "duplicate_conflicts.csv", index_label="row_id")
    sample = audit_sample(df, cfg)
    sample.to_csv(results / "label_audit_sample.csv", index=False)
    n_flagged = int((sample["dictionary_issues"] != "").sum())
    log.info("Label quality: %d groups of identical features with conflicting labels (%d rows); "
             "%d of %d sampled rows break a dictionary rule",
             n_groups, len(conflicts), n_flagged, len(sample))

    # 7. Processed dataset and its hash
    processed_train = X_train.assign(**{label: y_train.to_numpy()})
    processed_test = X_test.assign(**{label: y_test.to_numpy()})
    train_hash, train_bytes = sha256_of_frame(processed_train)
    test_hash, test_bytes = sha256_of_frame(processed_test)
    (processed_dir / "adult_train_processed.csv").write_bytes(train_bytes)
    (processed_dir / "adult_test_processed.csv").write_bytes(test_bytes)
    dataset_hash = hashlib.sha256(train_bytes + test_bytes).hexdigest()
    log.info("Processed dataset: train %s, test %s, SHA256 %s",
             processed_train.shape, processed_test.shape, dataset_hash)

    record = {
        "config": cfg,
        "rows": {"raw": len(raw), "train_after_split": n_train, "train_after_dedup": len(train),
                 "test": len(test)},
        "hashes": {"raw_sha256": raw_hash, "processed_sha256": dataset_hash,
                   "processed_train_sha256": train_hash, "processed_test_sha256": test_hash},
        "processed_columns": int(processed_train.shape[1]),
        "leakage_checks": leakage,
        "label_quality": {"conflict_groups": n_groups, "conflict_rows": len(conflicts),
                          "sample_rows_with_issues": n_flagged},
        "baseline": comparison,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scikit-learn": sklearn.__version__},
    }
    with open(results / "run_record.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(record, f, sort_keys=False, allow_unicode=True)
    log.info("Run record with config and hashes: run_record.yaml")


if __name__ == "__main__":
    main()
