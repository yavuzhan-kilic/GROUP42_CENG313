"""Data preparation helpers for the CSIRO Image2Biomass project."""
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

# Columns of train.csv that describe a single (image, target) pair.
# Every other column is metadata that belongs to the image itself.
PAIR_COLUMNS = ["sample_id", "target_name", "target"]


def image_table(data: pd.DataFrame) -> pd.DataFrame:
    """Reshape the raw long table into one row per image and one column per target.

    The raw CSV stores five different measurements in a single `target` column,
    so it is not tidy. `pivot` fails if an (image, target) pair appears twice,
    which doubles as a duplicate check.
    """
    return data.pivot(index="image_path", columns="target_name", values="target")


def metadata_table(data: pd.DataFrame) -> pd.DataFrame:
    """Return one row per image with the columns that are not targets (date, state, ...)."""
    metadata = data.drop(columns=PAIR_COLUMNS).groupby("image_path")

    # Each image must carry the same metadata in all of its rows.
    if (metadata.nunique(dropna=False) > 1).any().any():
        raise ValueError("Some images have conflicting metadata across their rows.")
    return metadata.first()


def biomass_differences(images: pd.DataFrame) -> pd.DataFrame:
    """Check the two sums the labels should satisfy and return the absolute gaps in grams.

    Green dry matter (GDM) = green + clover, and total = GDM + dead material.
    """
    return pd.DataFrame({
        "Green + clover difference (g)":
            (images["Dry_Green_g"] + images["Dry_Clover_g"] - images["GDM_g"]).abs(),
        "Total biomass difference (g)":
            (images["GDM_g"] + images["Dry_Dead_g"] - images["Dry_Total_g"]).abs(),
    })


def image_features(data_dir, image_paths):
    """Open every image and return its mean red, green and blue value.

    Returns (features, missing, unreadable): a DataFrame indexed by image path,
    plus the paths of files that do not exist or cannot be decoded.
    """
    rows, missing, unreadable = {}, [], []
    for path in image_paths:
        file = Path(data_dir) / path
        if not file.is_file():
            missing.append(path)
            continue
        try:
            with Image.open(file) as image:
                pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
        except Exception:
            unreadable.append(path)
            continue
        rows[path] = pixels.mean(axis=(0, 1))

    features = pd.DataFrame.from_dict(rows, orient="index", columns=["Red", "Green", "Blue"])
    features.index.name = "image_path"
    return features, missing, unreadable


def image_sizes(data_dir, image_paths) -> pd.DataFrame:
    """Return the width and height of each image (reads only the file header)."""
    sizes = {}
    for path in image_paths:
        with Image.open(Path(data_dir) / path) as image:
            sizes[path] = image.size
    return pd.DataFrame.from_dict(sizes, orient="index", columns=["Width", "Height"])


def load_image(data_dir, image_path, max_side: int = 600) -> Image.Image:
    """Load an image for display, shrunk so its longer side is at most `max_side` pixels."""
    image = Image.open(Path(data_dir) / image_path).convert("RGB")
    image.thumbnail((max_side, max_side))
    return image
