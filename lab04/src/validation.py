"""Schema, range, missing-value, category, duplicate and data-dictionary checks.

The checks only count and report. They never drop or change a row.
"""
import pandas as pd

SPOUSE_STATUSES = {"Married-civ-spouse", "Married-AF-spouse", "Married-spouse-absent"}


def _type_ok(series, expected):
    if expected == "int":
        if pd.api.types.is_integer_dtype(series):
            return True
        values = series.dropna()
        return pd.api.types.is_float_dtype(series) and bool((values % 1 == 0).all())
    return isinstance(series.dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(series)


def _is_marker(series, marker):
    return series.astype(object) == marker


def row_issues(df, cfg):
    """One text per row listing every data-dictionary rule it breaks ('' if none)."""
    schema = cfg["schema"]
    marker = schema["missing_marker"]
    issues = pd.Series([[] for _ in range(len(df))], index=df.index)

    def flag(mask, text):
        for idx in df.index[mask.fillna(False).to_numpy(dtype=bool)]:
            issues[idx].append(text)

    for col, (low, high) in schema["ranges"].items():
        flag((df[col] < low) | (df[col] > high), f"{col} outside {low}-{high}")
    for col, allowed in schema["categories"].items():
        ok = set(allowed) | ({marker} if col != cfg["data"]["label_column"] else set())
        values = df[col].astype(object)
        flag(values.notna() & ~values.isin(ok), f"{col} not in dictionary")
    codes = df["education"].astype(object).map(schema["education_codes"])
    flag(codes.notna() & (codes != df["education-num"]), "education-num does not match education")
    sex = df["sex"].astype(object)
    rel = df["relationship"].astype(object)
    flag((rel == "Husband") & (sex == "Female"), "Husband but Female")
    flag((rel == "Wife") & (sex == "Male"), "Wife but Male")
    flag(rel.isin(["Husband", "Wife"]) & ~df["marital-status"].astype(object).isin(SPOUSE_STATUSES),
         "Husband/Wife but not married")
    return issues.map("; ".join)


def run_validation(df, cfg):
    schema = cfg["schema"]
    marker = schema["missing_marker"]
    label = cfg["data"]["label_column"]
    checks = []

    def add(check, column, violations, detail):
        checks.append({"check": check, "column": column,
                       "violations": int(violations), "detail": detail})

    expected = list(schema["types"])
    missing_cols = [c for c in expected if c not in df.columns]
    extra_cols = [c for c in df.columns if c not in expected]
    add("columns", "all", len(missing_cols) + len(extra_cols),
        f"{df.shape[1]} columns; missing {missing_cols or 'none'}; extra {extra_cols or 'none'}")

    for col, expected_type in schema["types"].items():
        if col in df.columns:
            ok = _type_ok(df[col], expected_type)
            add("type", col, 0 if ok else 1, f"expected {expected_type}, found {df[col].dtype}")

    for col in expected:
        if col not in df.columns:
            continue
        n_nan = int(df[col].isna().sum())
        n_marker = int(_is_marker(df[col], marker).sum())
        # Missing values are allowed in categorical features only ('?' policy).
        allowed = schema["types"][col] == "category" and col != label
        add("missing", col, 0 if allowed else n_nan + n_marker,
            f"{n_nan} empty + {n_marker} '{marker}'"
            + ("" if allowed or n_nan + n_marker == 0 else " (not allowed for this column)"))

    for col, (low, high) in schema["ranges"].items():
        below = int((df[col] < low).sum())
        above = int((df[col] > high).sum())
        add("range", col, below + above,
            f"allowed {low}-{high}, found {df[col].min()}-{df[col].max()}; "
            f"{below} below, {above} above")

    for col, allowed_values in schema["categories"].items():
        ok = set(allowed_values) | ({marker} if col != label else set())
        values = df[col].astype(object)
        unknown = values[values.notna() & ~values.isin(ok)]
        add("category", col, len(unknown),
            f"{len(allowed_values)} allowed values"
            + (f"; unknown: {sorted(unknown.unique())[:10]}" if len(unknown) else ""))

    n_dup = int(df.duplicated().sum())
    add("duplicates", "all", n_dup, f"{n_dup} rows repeat an earlier row exactly "
        "(reported here; removed later from the training set only)")

    issues = row_issues(df, cfg)
    for rule in ["education-num does not match education", "Husband but Female",
                 "Wife but Male", "Husband/Wife but not married"]:
        hits = issues.index[issues.str.contains(rule, regex=False)]
        add("dictionary", rule, len(hits),
            f"{len(hits)} rows" + (f": row ids {list(map(int, hits[:10]))}" if len(hits) else ""))

    counts = df[label].value_counts()
    add("label", label, 0, ", ".join(f"{k}: {v}" for k, v in counts.items()))
    return checks


def to_markdown(checks, n_rows):
    lines = ["# Validation report", "",
             f"Raw dataset: {n_rows} rows. Checks only report; no row was dropped "
             "or changed at this step.", "",
             "| Check | Column | Violations | Detail |", "|---|---|---|---|"]
    for c in checks:
        lines.append(f"| {c['check']} | {c['column']} | {c['violations']} | {c['detail']} |")
    total = sum(c["violations"] for c in checks if c["check"] != "duplicates")
    lines += ["", f"Violations outside duplicates: {total}. "
              f"Exact duplicate rows: {next(c['violations'] for c in checks if c['check'] == 'duplicates')}."]
    return "\n".join(lines) + "\n"
