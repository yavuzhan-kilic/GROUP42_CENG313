# GROUP42_CENG313 — CSIRO Image2Biomass Prediction

CENG313 Introduction to Data Science course project, based on the Kaggle competition
[CSIRO - Image2Biomass Prediction](https://www.kaggle.com/competitions/csiro-biomass).

**Group ID:** 42

**Current Kaggle score (late submission, RGB + Ridge baseline):** public 0.31822 · private 0.25833

## Team

| Name | Student ID | GitHub |
|------|-----------|--------|
| Yavuzhan Kılıç | 23118080021 | [@yavuzhan-kilic](https://github.com/yavuzhan-kilic) |
| Efe Arda Dakes | 23118080071 | [@Efe-Arda-Dakes](https://github.com/Efe-Arda-Dakes) |

## Competition and dataset

- **Task:** predict five dry biomass weights (grams) from a top-view pasture image:
  `Dry_Green_g`, `Dry_Dead_g`, `Dry_Clover_g`, `GDM_g` (green + clover) and `Dry_Total_g` (GDM + dead).
- **Metric:** globally weighted R² computed over all (image, target) pairs together, in grams with no log
  transform (weights: total 0.5, GDM 0.2, others 0.1).
- **Training data:** 357 images. `train.csv` has 1,785 rows, one per image and target pair, with
  metadata for each image (sampling date, state, species, NDVI, average height).
- **Test data:** images only. The test table has `sample_id`, `image_path` and `target_name`;
  the metadata columns are not available at test time.

## Approach

### Assignment 1 — EDA and baselines

Notebook: [`assignment1_eda_baseline.ipynb`](assignment1_eda_baseline.ipynb)

The notebook follows the required sections: problem definition, data overview, EDA, preprocessing,
baseline models, Kaggle submission and conclusions.

- **Tidy data:** the long table is reshaped to one row per image and one column per target.
- **Data quality (trust):** no duplicates, no missing or negative targets, no missing or unreadable
  images. One image (`ID2014192906`) has a total that is 0.3088 g lower than the sum of its parts;
  it is reported and kept.
- **EDA:** distributions and skew, zero values (clover is 0 g in 135 images), extreme images,
  correlations between targets and with metadata, representation over states, species and dates.
- **Preprocessing:** no rows removed and nothing imputed (zeros are real measurements); each image is
  reduced to its mean RGB values, which are standardised inside the Ridge pipeline on the training folds
  only; negative predictions are clipped to 0.
- **Validation:** a random 80/20 split compared with 5-fold cross-validation grouped by sampling date.
- **Baselines:** (1) training mean, (2) mean RGB values + Ridge regression.
- **Kaggle:** late submission of RGB + Ridge, public 0.31822 and private 0.25833 (see the submission history).

Main findings:

- The targets are right-skewed and clover is zero in 37.8% of the images.
- Green, GDM and total biomass are strongly correlated (0.83 to 0.90).
- Average height is the most informative metadata column, but metadata is not available at test time.
- Mean colour is a weak predictor: RGB + Ridge is only slightly better than the training mean,
  and its score varies a lot between folds.
- The Kaggle private score (0.2583) is close to the grouped cross-validation score (0.2620), while the
  random split over-estimated it (0.2865), so grouped cross-validation is a reliable estimate.

### Assignment 2 — Proposed solution

TODO

## Results

| Model | Random split R² | Grouped CV R² (mean ± std) | Kaggle public | Kaggle private |
|-------|----------------|----------------------------|---------------|----------------|
| Training mean | 0.2096 | 0.2283 ± 0.0382 | – | – |
| RGB + Ridge | 0.2865 | 0.2620 ± 0.1138 | 0.31822 | 0.25833 |

The private score (0.2583) is close to the grouped cross-validation score (0.2620), while the random split
over-estimated it (0.2865). The best private score on the final leaderboard is 0.67899.

## Repository structure

```
GROUP42_CENG313/
├── README.md                        # group, members, competition, how to run, scores
├── requirements.txt                 # Python packages used
├── .gitignore                       # excludes data/, zip files and submission.csv
├── assignment1_eda_baseline.ipynb   # Assignment 1: EDA and baselines
├── src/
│   ├── __init__.py                  # empty file, makes src importable
│   ├── data.py                      # loading, reshaping, label checks, image features
│   ├── models.py                    # validation splits, baselines, cross-validation
│   └── utils.py                     # Kaggle weighted R², submission format and saving
├── data/                            # local copy of the Kaggle data (not committed)
└── submissions/                     # Kaggle score screenshots
```

## Getting the data

1. Join the competition on Kaggle and accept its rules:
   <https://www.kaggle.com/competitions/csiro-biomass/rules>
2. The Colab notebook downloads the data itself with the Kaggle API (see "How to run").
   To download it manually instead, install the Kaggle CLI, add your API token and run:
   ```
   kaggle competitions download -c csiro-biomass
   ```
   Then extract `csiro-biomass.zip` into the `data/` folder. The data is not committed to this repository.

## How to run

1. Open the notebook in Colab:
   <https://colab.research.google.com/github/ceng313-2026/GROUP42_CENG313/blob/main/assignment1_eda_baseline.ipynb>
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
| 2026-11-12 | Efe Arda Dakes | Added submission helpers (src/utils.py) and made the first Kaggle late submission (RGB + Ridge) |
| 2026-11-13 | Efe Arda Dakes | Restructured the notebook into the seven required sections: added problem definition, data overview summary, preprocessing and Kaggle submission sections |

## Kaggle submission history

The competition closed on 28 January 2026, so we use late submissions. They receive public and private
scores but do not appear on the leaderboard.

| Date | Kaggle notebook | Model | Public | Private | Screenshot |
|------|-----------------|-------|--------|---------|------------|
| 2026-11-12 | group42-assignment1-baseline (version 2) | RGB + Ridge | 0.31822 | 0.25833 | [link](submissions/assignment1_late_submission.png) |
