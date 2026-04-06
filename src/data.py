import pandas as pd


def load_train_data(path="../data/train.csv"):
    """Load train.csv. Contains both labeled rows (Score present) and unlabeled test rows (Score NaN)."""
    return pd.read_csv(path)


def load_test_data(path="../data/test.csv"):
    """Load test.csv (Id column only) — used to identify which rows in train.csv are the test set."""
    return pd.read_csv(path)