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

Started with a dummy baseline (predict mean 3.5 for everything) and worked up to a two-stage ensemble of Ridge + LightGBM.

| Model | Validation RMSE |
|---|---|
| Dummy (predict 3.5) | ~1.18 |
| Ridge + TF-IDF | 1.18 |
| + bias features | 0.87 |
| + VADER sentiment | 0.8423 |
| + separate Summary/Text TF-IDF | 0.8008 |
| + character n-grams (3–5) | 0.7864 |
| + ExtraTrees ensemble | ~0.776 |
| **+ two-stage residual prediction** | **~0.759** |

Best submission: `submissions/submission_ensemble.csv`

---

## The model

### Two-stage residual prediction

Instead of throwing everything into one model, I split the prediction into two stages:

**Stage 1 — bias baseline:**
```
baseline = clip(smoothed_user_mean + smoothed_product_mean - global_mean, 1, 5)
```
This captures the fact that some users always give 5 stars and some products always get 1 star.

**Stage 2 — text model on residual:**
```
residual = score - baseline
```
Ridge and LightGBM are both trained to predict the residual — how much the text pushes the score above or below what the user/product bias predicts.

**Final prediction:**
```
prediction = clip(baseline + Ridge_weight * ridge_residual + LightGBM_weight * lgbm_residual, 1, 5)
```

This is better than the all-in-one approach because the text model can focus purely on the content signal without having to also learn the user/product patterns.

### Ridge (~75% weight)
Trained on sparse features: separate TF-IDF (Summary + Text) + character n-grams on Text + character n-grams on Summary + LSA + numeric + sentiment. No bias in the feature matrix — bias is already in the baseline.

### ExtraTreesRegressor (~25% weight)
Trained on dense features: LSA + numeric + sentiment + LOO bias (as residual correction signals). Uses leave-one-out bias to prevent target leakage during training. ExtraTrees uses random splits rather than optimized splits — it is a bagging method, not boosting.

The blend weight was picked by sweeping Ridge 50%→95% on the validation set.

---

## Features

**Text:**
- Separate TF-IDF for `Summary` (10k features, `min_df=2`) and `Text` (30k features, `min_df=2`) — tested concatenated vs separate, separate wins by ~0.02 RMSE
- Character n-grams (`char_wb`, 3–5) on `Text` (20k features) — catches morphological patterns like "terribl", "excellen", "amaz" across word forms
- Character n-grams (`char_wb`, 3–5) on `Summary` (10k features) — high-signal short phrases like "must see", "avoid"
- LSA/SVD (200 components) on the combined text matrix for dense semantic features

**User & product bias baseline (Stage 1):**
- Smoothed Bayesian mean per user and per product (shrinkage k=10)
- `baseline = clip(user_smoothed + product_smoothed - global_mean, 1, 5)`
- Unknown users/products fall back to global mean

**Bias features for LightGBM (residual correction signals):**
- Leave-one-out smoothed means — each sample's user/product mean computed without that sample's score
- `user_score_std`, `product_score_std` — rating consistency of user/product
- User and product review counts

**Sentiment (VADER):**
- Compound, positive, negative scores on both Summary and Text
- Rule-based, no fitting needed — no leakage risk

**Sentiment (VADER):**
- Compound, positive, negative scores on both Summary and Text
- `vader_ratio = pos / (neg + 0.01)` — continuous signal for strongly positive vs negative reviews
- Rule-based, no fitting needed — no leakage risk

**Numeric:**
- Helpfulness ratio, text/summary length, word count, exclamation/question marks, uppercase ratio, unique word ratio, avg word length, year, month, quarter
- `SentenceCount`, `AvgSentenceLength` — structural review features
- `HelpfulnessXLength = HelpfulnessRatio × log1p(TextLength)` — interaction: long, helpful reviews tend to be more informative

---

## What I tested and didn't use

| Experiment | Result | Decision |
|---|---|---|
| Porter stemming | 0.7895 (worse) | Skip — review text relies on word form nuance |
| WordNet lemmatization | 0.8064 (worse) | Skip |
| Concatenated Summary+Text TF-IDF | 0.8057 (worse) | Skip — separate is better |
| `min_df=5` | 0.8232 (worse) | Skip — too aggressive |
| `max_features=50k` text | 0.7948 (worse) | Skip — adds noise |
| ExtraTrees with raw TF-IDF | overfits badly | Skip — use Ridge for sparse text |
| ExtraTrees with LSA (no LOO bias) | high overfitting | Skip — LOO bias is essential |

---

## Validation

80/20 stratified split on Score (`random_state=42`). All transformers (TF-IDF, SVD, bias statistics) fit on the train fold only, applied to the validation fold. Final models retrained on all labeled data before submission.

---

## Stuff that went wrong

**Data leakage** — I was computing user/product bias from the full dataset before splitting. The validation set's own scores were baked into the bias features, so local RMSE looked great (0.579) but Kaggle said 0.979. Splitting first fixed it.

**LightGBM overfitting with regular bias** — first attempt got Train RMSE 0.34 / Valid RMSE 1.10. The bias features included each sample's own score in its user mean, so trees memorized. Fixed with leave-one-out bias.

**LSA fingerprinting** — 200 dense LSA components are basically a unique fingerprint per review. LightGBM memorized training samples through them. Kept LSA for Ridge only; LightGBM uses it only with LOO bias as a regularizing signal.

**ExtraTrees overfitting** — early ExtraTrees runs without `min_samples_leaf` overfit badly on training data. Fixed by setting `min_samples_leaf=20` to force sufficient data at each leaf.

**Dummy prediction bug** — early submissions predicted 3.5 for everything because the model was trained but never actually called for predictions.

---

## Project structure

```
CS506_midterm/
├── data/
│   ├── train.csv
│   └── test.csv
├── notebooks/
│   ├── modeling.ipynb      # full pipeline
│   └── eda.ipynb
├── src/
│   ├── data.py             # loading data
│   ├── features.py         # TF-IDF, char n-grams, bias, baseline, sentiment, numeric
│   └── model.py            # Ridge, LightGBM, eval, submission
├── submissions/
│   ├── submission.csv             # Ridge two-stage only
│   └── submission_ensemble.csv   # best — Ridge + ExtraTrees ensemble
├── assets/
│   └── predicted_score_dist.png
├── requirements.txt
└── Makefile
```
