[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/6KpniKiX)

# CS506 Midterm — Amazon Review Rating Prediction

Predict the star rating (1–5) of Amazon Movie Reviews. Metric: **RMSE** (lower is better).

---

## How to run

```bash
pip install -r requirements.txt
jupyter notebook notebooks/modeling.ipynb
```

Or with make:

```bash
make run    # modeling notebook
make eda    # EDA notebook
make all    # everything
```

---

## What I built

Started with a dummy baseline (predict mean 3.5 for everything) and worked up to an ensemble of Ridge + LightGBM.

| Model | Validation RMSE |
|---|---|
| Dummy (predict 3.5) | ~1.18 |
| Ridge + TF-IDF | 1.18 |
| + bias features | 0.87 |
| + VADER sentiment | 0.8423 |
| + separate Summary/Text TF-IDF | 0.8008 |
| **+ LightGBM ensemble** | **~0.787** |

Best submission: `submissions/submission_ensemble.csv`

---

## The model

Two models averaged together:

**Ridge (~70% weight)** — trained on sparse TF-IDF + LSA + bias + sentiment + numeric features. Ridge handles high-dimensional sparse text really well.

**LightGBM (~30% weight)** — trained on dense features only (LSA + bias + sentiment + numeric). Trees don't play well with 30k sparse TF-IDF columns so I kept it dense.

The blend weight was picked by sweeping Ridge 50%→95% on the validation set and taking the best.

---

## Features

**Text:**
- Separate TF-IDF for `Summary` (10k features) and `Text` (30k features) — summaries are short and punchy, body text is long and noisy, so treating them separately helps
- LSA/SVD (200 components) on the combined matrix for dense semantic features

**User & product bias** (most impactful):
- Smoothed mean score per user and per product (Bayesian shrinkage, k=10)
- Bias = deviation from global mean (~3.97)
- Std of scores — captures whether a user is consistent or all over the place
- For LightGBM training: used leave-one-out bias to avoid leaking the target into the features

**Sentiment (VADER):**
- Compound, positive, negative scores on both Summary and Text
- Rule-based so no fitting needed — safe to apply to both folds

**Numeric:**
- Helpfulness ratio, text/summary length, word count, exclamation/question marks, uppercase ratio, unique word ratio, avg word length, year, month

---

## Validation

80/20 stratified split on Score. All transformers fit on the train fold only. Final models retrained on all labeled data before submission.

---

## Stuff that went wrong

**Data leakage** — I was computing user/product bias from the full dataset before splitting. The validation set's own scores were baked into the bias features, so local RMSE looked great (0.579) but Kaggle said 0.979. Splitting first fixed it.

**LightGBM overfitting** — first attempt got Train RMSE 0.34 / Valid RMSE 1.10. The bias features were including each sample's own score in its user mean, so trees just memorized. Fixed with leave-one-out bias.

**LSA fingerprinting** — 200 dense LSA components are basically a unique fingerprint per review. LightGBM memorized the training set through them. Kept LSA for Ridge only.

**Dummy prediction bug** — early submissions predicted 3.5 for everything because the model was trained but never actually called for predictions.

---

## Project structure

```
CS506_midterm/
├── data/
│   ├── train.csv
│   └── test.csv
├── notebooks/
│   ├── modeling.ipynb
│   └── eda.ipynb
├── src/
│   ├── data.py        # loading data
│   ├── features.py    # all feature engineering
│   └── model.py       # Ridge, LightGBM, eval, submission
├── submissions/
│   ├── submission.csv             # Ridge only
│   └── submission_ensemble.csv   # best — Ridge + LightGBM
├── requirements.txt
└── Makefile
```
