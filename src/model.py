import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report


def train_decision_tree(X_train, y_train):
    model = DecisionTreeClassifier(
        random_state=0,
        class_weight="balanced",
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(
        random_state=0,
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_train, X_test, y_train, y_test, title="Model"):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    print(f"\n{title}")
    print("Accuracy on training set =", accuracy_score(y_train, y_train_pred))
    print("Accuracy on testing set  =", accuracy_score(y_test, y_test_pred))
    print("Weighted F1 on training set =", f1_score(y_train, y_train_pred, average='weighted'))
    print("Weighted F1 on testing set  =", f1_score(y_test, y_test_pred, average='weighted'))

    print("\nClassification Report (test):")
    print(classification_report(y_test, y_test_pred))

    cm = confusion_matrix(y_test, y_test_pred)

    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title(f'Confusion Matrix of {title}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    return f1_score(y_test, y_test_pred, average='weighted')


def save_model(model, filename):
    with open(filename, "wb") as f:
        pickle.dump(model, f)


def create_submission(model, test_df, feature_columns, output_path="../submissions/submission.csv"):
    X_test_kaggle = test_df.reindex(columns=feature_columns, fill_value=0)
    X_test_kaggle = X_test_kaggle.fillna(0)

    test_predictions = model.predict(X_test_kaggle)

    submission = pd.DataFrame({
        "Id": test_df["Id"],
        "Score": test_predictions
    })

    submission.to_csv(output_path, index=False)
    return submission