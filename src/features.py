from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from scipy.sparse import hstack, csr_matrix
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_sia = SentimentIntensityAnalyzer()


def build_sentiment_features(df):
    """VADER sentiment scores on Summary and Text columns."""
    summaries = df["Summary"].fillna("").astype(str)
    texts = df["Text"].fillna("").astype(str)

    summary_scores = summaries.apply(_sia.polarity_scores)
    text_scores = texts.apply(_sia.polarity_scores)

    result = pd.DataFrame(index=df.index)
    result["summary_vader_compound"] = summary_scores.apply(lambda s: s["compound"])
    result["summary_vader_pos"]      = summary_scores.apply(lambda s: s["pos"])
    result["summary_vader_neg"]      = summary_scores.apply(lambda s: s["neg"])
    result["text_vader_compound"]    = text_scores.apply(lambda s: s["compound"])
    result["text_vader_pos"]         = text_scores.apply(lambda s: s["pos"])
    result["text_vader_neg"]         = text_scores.apply(lambda s: s["neg"])
    return result


def build_bias_features(train_labeled, all_data, shrinkage=10):
    """
    Compute smoothed (Bayesian) user and product bias features.
    shrinkage=k: rare users/products are pulled toward global mean.
    Formula: smoothed_mean = (n * raw_mean + k * global_mean) / (n + k)
    """
    global_mean = train_labeled["Score"].mean()

    user_stats = (
        train_labeled.groupby("UserId")["Score"]
        .agg(user_raw_mean="mean", user_review_count="count")
        .reset_index()
    )
    product_stats = (
        train_labeled.groupby("ProductId")["Score"]
        .agg(product_raw_mean="mean", product_review_count="count")
        .reset_index()
    )

    # Smoothed means
    user_stats["user_mean_score"] = (
        (user_stats["user_review_count"] * user_stats["user_raw_mean"] + shrinkage * global_mean)
        / (user_stats["user_review_count"] + shrinkage)
    )
    product_stats["product_mean_score"] = (
        (product_stats["product_review_count"] * product_stats["product_raw_mean"] + shrinkage * global_mean)
        / (product_stats["product_review_count"] + shrinkage)
    )

    df = all_data[["UserId", "ProductId"]].copy()
    df = df.merge(user_stats[["UserId", "user_mean_score", "user_review_count"]], on="UserId", how="left")
    df = df.merge(product_stats[["ProductId", "product_mean_score", "product_review_count"]], on="ProductId", how="left")

    df["user_mean_score"] = df["user_mean_score"].fillna(global_mean)
    df["user_review_count"] = df["user_review_count"].fillna(0)
    df["product_mean_score"] = df["product_mean_score"].fillna(global_mean)
    df["product_review_count"] = df["product_review_count"].fillna(0)

    df["user_bias"] = df["user_mean_score"] - global_mean
    df["product_bias"] = df["product_mean_score"] - global_mean

    # Combined bias signal
    df["user_product_bias"] = df["user_bias"] + df["product_bias"]

    return df[["user_mean_score", "user_review_count", "product_mean_score",
               "product_review_count", "user_bias", "product_bias",
               "user_product_bias"]].values


def build_numeric_features(df):
    X = pd.DataFrame(index=df.index)

    X["HelpfulnessRatio"] = (
        df["HelpfulnessNumerator"] / (df["HelpfulnessDenominator"] + 1)
    )
    X["HelpfulnessDenominator"] = df["HelpfulnessDenominator"].fillna(0)
    X["HelpfulnessNumerator"] = df["HelpfulnessNumerator"].fillna(0)

    text_col = df["Text"].fillna("")
    summary_col = df["Summary"].fillna("")

    X["TextLength"] = text_col.apply(len)
    X["SummaryLength"] = summary_col.apply(len)
    X["TextWordCount"] = text_col.apply(lambda x: len(x.split()))
    X["SummaryWordCount"] = summary_col.apply(lambda x: len(x.split()))

    X["NumExclamation"] = text_col.apply(lambda x: x.count("!"))
    X["NumQuestion"] = text_col.apply(lambda x: x.count("?"))
    X["UppercaseRatio"] = text_col.apply(
        lambda x: sum(1 for c in x if c.isupper()) / (len(x) + 1)
    )
    X["UniqueWordRatio"] = text_col.apply(
        lambda x: len(set(x.lower().split())) / (len(x.split()) + 1)
    )
    X["AvgWordLength"] = text_col.apply(
        lambda x: np.mean([len(w) for w in x.split()]) if x.split() else 0
    )

    dt = pd.to_datetime(df["Time"], unit="s", errors="coerce")
    X["Year"] = dt.dt.year.fillna(0)
    X["Month"] = dt.dt.month.fillna(0)

    return X.fillna(0)


def prepare_training_data(training_df, text_column="Text", max_features=30000, ngram_range=(1, 2)):
    # Labeled = has Score; unlabeled = test set
    labeled = training_df.dropna(subset=["Score"]).copy()

    y = labeled["Score"].values

    # Bias features (fit only on labeled, applied to labeled)
    bias = build_bias_features(labeled, labeled)

    # Numeric features
    num_feats = build_numeric_features(labeled)

    # Sentiment features
    sentiment = build_sentiment_features(labeled)
    num_feats = pd.concat([num_feats, sentiment], axis=1)

    numeric_columns = num_feats.columns.tolist()

    # Custom stopwords
    movie_stop = {
        "movie", "film", "watch", "watched", "one", "really",
        "also", "even", "get", "got", "make", "made"
    }
    stop_words = list(text.ENGLISH_STOP_WORDS.union(movie_stop))

    # TF-IDF on combined Summary + Text
    combined_text = (
        labeled["Summary"].fillna("").astype(str) + " " +
        labeled["Text"].fillna("").astype(str)
    )

    tfidf = TfidfVectorizer(
        lowercase=True,
        stop_words=stop_words,
        ngram_range=ngram_range,
        min_df=3,
        max_df=0.9,
        sublinear_tf=True,
        norm="l2",
        smooth_idf=True,
        use_idf=True,
        max_features=max_features
    )

    X_text = tfidf.fit_transform(combined_text)
    print("TF-IDF matrix shape:", X_text.shape)

    # LSA: reduce TF-IDF to 200 dense semantic components
    svd = TruncatedSVD(n_components=200, random_state=42)
    X_lsa = svd.fit_transform(X_text)
    X_lsa = Normalizer(copy=False).fit_transform(X_lsa)
    print(f"LSA explained variance: {svd.explained_variance_ratio_.sum():.3f}")

    X_num_sparse = csr_matrix(num_feats.values)
    X_bias_sparse = csr_matrix(bias)
    X_lsa_sparse = csr_matrix(X_lsa)
    X = hstack([X_num_sparse, X_bias_sparse, X_text, X_lsa_sparse])

    return X, y, tfidf, svd, numeric_columns, labeled


def prepare_test_data(test_df, tfidf, svd, numeric_columns, labeled_df, text_column="Text"):
    # Bias features: compute from labeled data, apply to test
    bias = build_bias_features(labeled_df, test_df)

    # Numeric features
    num_feats = build_numeric_features(test_df)
    sentiment = build_sentiment_features(test_df)
    num_feats = pd.concat([num_feats, sentiment], axis=1)
    num_feats = num_feats.reindex(columns=numeric_columns, fill_value=0).fillna(0)

    combined_text = (
        test_df["Summary"].fillna("").astype(str) + " " +
        test_df["Text"].fillna("").astype(str)
    )
    X_text = tfidf.transform(combined_text)

    # Apply fitted SVD
    X_lsa = svd.transform(X_text)
    X_lsa = Normalizer(copy=False).fit_transform(X_lsa)

    X_num_sparse = csr_matrix(num_feats.values)
    X_bias_sparse = csr_matrix(bias)
    X_lsa_sparse = csr_matrix(X_lsa)
    X_test = hstack([X_num_sparse, X_bias_sparse, X_text, X_lsa_sparse])

    return X_test
