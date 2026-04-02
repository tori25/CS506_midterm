import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


def prepare_training_data(training_df, text_column="Text", max_features=10000, ngram_range=(1, 2)):
    dataset = training_df.dropna(subset=["Score"]).copy()

    # target
    y = dataset["Score"]

    # numeric features
    X_num = dataset.drop(columns=["Score"]).select_dtypes(include=["number"]).copy()
    X_num = X_num.fillna(0)

    # text feature
    if text_column not in dataset.columns:
        raise ValueError(f"Column '{text_column}' not found in training data.")

    text_data = dataset[text_column].fillna("").astype(str)

    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words="english"
    )
    X_text = tfidf.fit_transform(text_data)

    # combine numeric + text
    X_num_sparse = csr_matrix(X_num.values)
    X = hstack([X_num_sparse, X_text])

    return X, y, tfidf, X_num.columns.tolist()


def prepare_test_data(test_df, tfidf, numeric_columns, text_column="Text"):
    # numeric features
    X_num = test_df.reindex(columns=numeric_columns, fill_value=0).copy()
    X_num = X_num.fillna(0)

    # text feature
    if text_column not in test_df.columns:
        raise ValueError(f"Column '{text_column}' not found in test data.")

    text_data = test_df[text_column].fillna("").astype(str)
    X_text = tfidf.transform(text_data)

    # combine numeric + text
    X_num_sparse = csr_matrix(X_num.values)
    X_test = hstack([X_num_sparse, X_text])

    return X_test