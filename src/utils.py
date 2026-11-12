"""Helpers for writing Kaggle submission files."""
from pathlib import Path

import pandas as pd


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