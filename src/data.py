import pandas as pd


def load_train_data(path="../data/train.csv"):
    return pd.read_csv(path)


def load_test_data(path="../data/test.csv"):
    return pd.read_csv(path)