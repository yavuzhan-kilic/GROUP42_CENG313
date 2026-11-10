"""Validation splits, the competition metric and the baseline models."""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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


def split_images(images: pd.DataFrame, validation_ratio: float = 0.2, seed: int = 42):
    """Randomly split the image table into a training and a validation part."""
    order = np.random.default_rng(seed).permutation(len(images))
    validation_size = int(round(len(images) * validation_ratio))
    validation = images.iloc[np.sort(order[:validation_size])]
    train = images.iloc[np.sort(order[validation_size:])]
    return train, validation


def group_folds(groups: pd.Series, n_splits: int = 5) -> pd.Series:
    """Assign a fold number to every image so that a group never spans two folds.

    Images taken on the same day look alike; keeping them in one fold stops the
    model from being validated on near-copies of its training images.
    """
    folds = pd.Series(-1, index=groups.index, name="fold")
    splitter = GroupKFold(n_splits=n_splits)
    for fold, (_, validation_rows) in enumerate(splitter.split(groups, groups=groups)):
        folds.iloc[validation_rows] = fold
    return folds


def mean_predictions(train: pd.DataFrame, index) -> pd.DataFrame:
    """Baseline 1: predict the training mean of each target for every image."""
    values = np.tile(train.mean().to_numpy(), (len(index), 1))
    return pd.DataFrame(values, index=index, columns=train.columns)


def rgb_ridge_predictions(rgb: pd.DataFrame, train: pd.DataFrame, index, alpha: float = 1.0):
    """Baseline 2: Ridge regression from the mean RGB values to the five targets.

    Scaling and coefficients are fitted on the training images only.
    Negative predictions are clipped to zero because a weight cannot be negative.
    """
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(rgb.loc[train.index], train)
    values = model.predict(rgb.loc[index]).clip(min=0)
    return pd.DataFrame(values, index=index, columns=train.columns)


def cross_validate(images: pd.DataFrame, folds: pd.Series, models: dict):
    """Evaluate each model on every fold.

    `models` maps a name to a function (train_table, validation_index) -> predictions.
    Returns (scores, predictions): the weighted R² per fold and model, and each
    model's out-of-fold predictions for all images.
    """
    scores = pd.DataFrame(index=sorted(folds.unique()), columns=list(models), dtype=float)
    scores.index.name = "Fold"
    predictions = {name: pd.DataFrame(index=images.index, columns=images.columns, dtype=float)
                   for name in models}

    for fold in scores.index:
        train = images[folds != fold]
        validation = images[folds == fold]
        for name, predict in models.items():
            predicted = predict(train, validation.index)
            predictions[name].loc[validation.index] = predicted
            scores.loc[fold, name] = kaggle_r2(validation, predicted)
    return scores, predictions
