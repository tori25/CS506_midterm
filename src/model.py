import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.svm import LinearSVR
from sklearn.metrics import mean_squared_error


def train_ridge(X_train, y_train, alpha=10.0):
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    return model


def train_extra_trees(X_train, y_train,
                      n_estimators=300, min_samples_leaf=20,
                      max_features=0.3, random_state=42):
    """
    ExtraTreesRegressor on dense features (LSA + numeric + sentiment + LOO bias).
    Extremely Randomized Trees: uses random splits (not best splits) so it is
    faster and more regularized than Random Forest. Not a boosting method.
    """
    model = ExtraTreesRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_linear_svr(X_train, y_train, C=0.1, max_iter=2000):
    model = LinearSVR(C=C, max_iter=max_iter, dual=True)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_train, X_valid, y_train, y_valid, title="Model"):
    y_train_pred = np.clip(model.predict(X_train), 1, 5)
    y_valid_pred = np.clip(model.predict(X_valid), 1, 5)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    valid_rmse = np.sqrt(mean_squared_error(y_valid, y_valid_pred))

    print(f"\n{title}")
    print(f"  Train RMSE : {train_rmse:.4f}")
    print(f"  Valid RMSE : {valid_rmse:.4f}")

    return valid_rmse


def save_model(model, filename):
    with open(filename, "wb") as f:
        pickle.dump(model, f)


def load_model(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def create_submission(
    model,
    test_df,
    X_test,
    output_path="../submissions/submission.csv",
    clip_range=(1, 5),
    round_predictions=False
):
    predictions = model.predict(X_test)

    if clip_range is not None:
        predictions = np.clip(predictions, clip_range[0], clip_range[1])

    if round_predictions:
        predictions = np.rint(predictions).astype(int)

    submission = pd.DataFrame({
        "Id": test_df["Id"],
        "Score": predictions
    })

    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path} ({len(submission)} rows)")
    return submission
