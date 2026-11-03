"""Label-quality checks for Adult: duplicate-conflict audit and a 50-row dictionary re-check."""
from validation import row_issues


def duplicate_conflicts(df, label):
    """Rows whose 14 features are identical but whose income labels differ."""
    features = [c for c in df.columns if c != label]
    groups = df.groupby(features, dropna=False, observed=True)[label]
    n_labels = groups.transform("nunique")
    conflicts = df[n_labels > 1].sort_values(features)
    n_groups = conflicts.groupby(features, dropna=False, observed=True).ngroups
    return conflicts, n_groups


def audit_sample(df, cfg):
    sample = df.sample(n=cfg["label_quality"]["sample_size"], random_state=cfg["seed"]).copy()
    sample["dictionary_issues"] = row_issues(sample, cfg)
    sample.insert(0, "row_id", sample.index)
    return sample
