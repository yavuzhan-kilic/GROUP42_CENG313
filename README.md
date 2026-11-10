# GROUP42_CENG313 — CSIRO Image2Biomass Prediction

CENG313 Introduction to Data Science course project, based on the Kaggle competition
[CSIRO - Image2Biomass Prediction](https://www.kaggle.com/competitions/csiro-biomass).

## Team

| Name | Student ID | GitHub |
|------|-----------|--------|
| Yavuzhan Kılıç | 23118080021 | [@yavuzhan-kilic](https://github.com/yavuzhan-kilic) |
| Efe Arda Dakes | 23118080071 | [@efe-arda-dakes](https://github.com/efe-arda-dakes) |

## Competition and dataset

- **Task:** predict five dry biomass weights (grams) from a top-view pasture image:
  `Dry_Green_g`, `Dry_Dead_g`, `Dry_Clover_g`, `GDM_g` (green + clover) and `Dry_Total_g` (GDM + dead).
- **Metric:** weighted R² computed over all targets together (weights: total 0.5, GDM 0.2, others 0.1).
- **Training data:** 357 images. `train.csv` has 1,785 rows, one per image and target pair, with
  metadata for each image (sampling date, state, species, NDVI, average height).
- **Test data:** images only. The test table has `sample_id`, `image_path` and `target_name`;
  the metadata columns are not available at test time.

## Approach

### Assignment 1 — EDA and baselines

Notebook: [`assignment1_eda_baseline.ipynb`](assignment1_eda_baseline.ipynb)

- **Tidy data:** the long table is reshaped to one row per image and one column per target.
- **Data quality (trust):** no duplicates, no missing or negative targets, no missing or unreadable
  images. One image (`ID2014192906`) has a total that is 0.3088 g lower than the sum of its parts;
  it is reported and kept.
- **EDA:** distributions and skew, zero values (clover is 0 g in 135 images), extreme images,
  correlations between targets and with metadata, representation over states, species and dates.
- **Validation:** a random 80/20 split compared with 5-fold cross-validation grouped by sampling date.
- **Baselines:** (1) training mean, (2) mean RGB values + Ridge regression.

Main findings:

- The targets are right-skewed and clover is zero in 37.8% of the images.
- Green, GDM and total biomass are strongly correlated (0.83 to 0.90).
- Average height is the most informative metadata column, but metadata is not available at test time.
- Mean colour is a weak predictor: RGB + Ridge is only slightly better than the training mean,
  and its score varies a lot between folds.

### Assignment 2 — Proposed solution

TODO

## Results

| Model | Random split R² | Grouped CV R² (mean ± std) | Kaggle public | Kaggle private |
|-------|----------------|----------------------------|---------------|----------------|
| Training mean | 0.2096 | 0.2283 ± 0.0382 | – | – |
| RGB + Ridge | 0.2865 | 0.2620 ± 0.1138 | – | – |

## Repository structure

```
GROUP42_CENG313/
├── README.md
├── requirements.txt
├── assignment1_eda_baseline.ipynb   # Assignment 1: EDA and baselines
├── data/                            # local copy of the Kaggle data (not committed)
└── src/
    ├── data.py      # reshaping, label consistency checks, image features
    └── models.py    # validation splits, Kaggle weighted R², baselines, cross-validation
```

## How to run

1. Open the notebook in Colab:
   <https://colab.research.google.com/github/yavuzhan-kilic/GROUP42_CENG313/blob/main/assignment1_eda_baseline.ipynb>
2. Accept the competition rules on Kaggle. Add your Kaggle API token to Colab's **Secrets** panel
   as `KAGGLE_API_TOKEN` and enable notebook access.
3. Run all cells. The notebook clones this repository, downloads the data and imports the code
   in `src/`.

To run locally, install `requirements.txt`, extract the competition data into `data/` and start
Jupyter with `CSIRO_DATA_DIR=data`.

## Progress log

| Date | Member | Work |
|------|--------|------|
| 2026-11-10 | Yavuzhan Kılıç | Repository structure, data helpers and baselines |
| 2026-11-11 | Efe Arda Dakes | Ran the notebook and reviewed the EDA findings |

## Kaggle submission history

No submissions yet.
