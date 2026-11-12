"""Shared helpers: the competition metric and Kaggle submission files."""
from pathlib import Path

import pandas as pd

# Target weights of the competition metric (see the Kaggle Evaluation page).
TARGET_WEIGHTS = pd.Series({
    "Dry_Green_g": 0.1,
    "Dry_Dead_g": 0.1,
    "Dry_Clover_g": 0.1,
    "GDM_g": 0.2,
    "Dry_Total_g": 0.5,
})


def kaggle_r2(actual: pd.DataFrame, predicted: pd.DataFrame) -> float:
    """Weighted R² computed over all (image, target) values together.

    Every value is weighted by its target's weight. The baseline in the
    denominator is the single weighted mean of all values, not a per-target mean.
    """
    predicted = predicted.loc[actual.index, actual.columns]
    weights = TARGET_WEIGHTS[actual.columns]

    weighted_mean = (actual * weights).sum().sum() / (weights.sum() * len(actual))
    residual = ((actual - predicted) ** 2 * weights).sum().sum()
    total = ((actual - weighted_mean) ** 2 * weights).sum().sum()
    return float(1 - residual / total)


def make_submission(test: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    """Turn wide predictions into the long format Kaggle expects.

    `test` is test.csv: one row per (image, target) pair with the columns
    sample_id, image_path and target_name.
    `predictions` has one row per image (index = image_path) and one column per target.
    Returns a table with the columns sample_id and target, in the order of test.csv.
    """
    long = predictions.rename_axis("image_path").reset_index().melt(
        id_vars="image_path", var_name="target_name", value_name="target")
    submission = test.merge(long, on=["image_path", "target_name"], how="left")

    if submission["target"].isna().any():
        missing = submission.loc[submission["target"].isna(), "sample_id"].tolist()
        raise ValueError(f"No prediction for {len(missing)} rows, e.g. {missing[:3]}")
    # A weight cannot be negative.
    submission["target"] = submission["target"].clip(lower=0)
    return submission[["sample_id", "target"]]


def save_submission(submission: pd.DataFrame, path="submission.csv") -> Path:
    """Write the submission without the index column and return its path."""
    path = Path(path)
    submission.to_csv(path, index=False)
    return path