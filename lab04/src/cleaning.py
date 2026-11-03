"""Cleaning and feature steps. Every statistic is learned in fit(), on the training set only."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
CATEGORICAL = ["workclass", "education", "marital-status", "occupation",
               "relationship", "race", "sex", "native-country"]
DERIVED = ["capital_net", "hours_per_age"]


class MissingToNaN(BaseEstimator, TransformerMixin):
    """Turn the '?' marker into a real missing value (no statistics, nothing to learn)."""

    def __init__(self, columns, marker="?"):
        self.columns = columns
        self.marker = marker

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            X[col] = X[col].astype(object).replace(self.marker, np.nan)
        return X


class DerivedFeatures(BaseEstimator, TransformerMixin):
    """capital_net = capital-gain - capital-loss, hours_per_age = hours-per-week / (age - 16).

    Both use only the values of the same row, so nothing is learned from the data.
    The dataset contains people aged 17 and older, so age - 16 is at least 1.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["capital_net"] = X["capital-gain"] - X["capital-loss"]
        X["hours_per_age"] = X["hours-per-week"] / (X["age"] - 16)
        return X


class PercentileClipper(BaseEstimator, TransformerMixin):
    """Clip every column to percentiles learned in fit()."""

    def __init__(self, lower=1, upper=99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        values = np.asarray(X, dtype=float)
        self.n_features_in_ = values.shape[1]
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.lower_ = np.nanpercentile(values, self.lower, axis=0)
        self.upper_ = np.nanpercentile(values, self.upper, axis=0)
        return self

    def transform(self, X):
        clipped = np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(clipped, columns=X.columns, index=X.index)
        return clipped

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = getattr(self, "feature_names_in_",
                                     [f"x{i}" for i in range(self.n_features_in_)])
        return np.asarray(input_features, dtype=object)


def build_preprocessor(cfg, with_features=True):
    numeric = NUMERIC + (DERIVED if with_features else [])
    low, high = cfg["cleaning"]["clip_percentiles"]
    numeric_steps = Pipeline([
        ("impute", SimpleImputer(strategy=cfg["cleaning"]["numeric_imputation"])),
        ("clip", PercentileClipper(low, high)),
        ("scale", StandardScaler()),
    ])
    categorical_steps = Pipeline([
        ("impute", SimpleImputer(strategy=cfg["cleaning"]["categorical_imputation"])),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    columns = ColumnTransformer([
        ("num", numeric_steps, numeric),
        ("cat", categorical_steps, CATEGORICAL),
    ], verbose_feature_names_out=False).set_output(transform="pandas")
    steps = [("missing", MissingToNaN(CATEGORICAL, cfg["schema"]["missing_marker"]))]
    if with_features:
        steps.append(("features", DerivedFeatures()))
    steps.append(("columns", columns))
    return Pipeline(steps)
