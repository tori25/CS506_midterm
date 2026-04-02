import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


def train_ridge_regression(X_train, y_train, alpha=1.0):
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    return model


def evaluate_regression_model(model, X_train, X_valid, y_train, y_valid, title="Model"):
    y_train_pred = model.predict(X_train)
    y_valid_pred = model.predict(X_valid)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    valid_rmse = np.sqrt(mean_squared_error(y_valid, y_valid_pred))

    print(f"\n{title}")
    print("Training RMSE =", round(train_rmse, 4))
    print("Validation RMSE =", round(valid_rmse, 4))

    return valid_rmse


def save_model(model, filename):
    with open(filename, "wb") as f:
        pickle.dump(model, f)


def create_submission(
    model,
    test_df,
    X_test_kaggle,
    output_path="../submissions/submission.csv",
    clip_range=(1, 5),
    round_predictions=False
):
    predictions = model.predict(X_test_kaggle)

    if clip_range is not None:
        predictions = np.clip(predictions, clip_range[0], clip_range[1])

    if round_predictions:
        predictions = np.rint(predictions).astype(int)

    submission = pd.DataFrame({
        "Id": test_df["Id"],
        "Score": predictions
    })

    submission.to_csv(output_path, index=False)
    return submission