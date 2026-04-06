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
    # Ratio: positive signal relative to negative signal
    result["summary_vader_ratio"]    = result["summary_vader_pos"] / (result["summary_vader_neg"] + 0.01)
    result["text_vader_ratio"]       = result["text_vader_pos"] / (result["text_vader_neg"] + 0.01)
    # Sentiment gap: summary and body disagree → signals sarcasm or mixed reviews
    result["sentiment_gap"]          = (result["summary_vader_compound"] - result["text_vader_compound"]).abs()
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
        .agg(user_raw_mean="mean", user_review_count="count", user_score_std="std")
        .reset_index()
    )
    product_stats = (
        train_labeled.groupby("ProductId")["Score"]
        .agg(product_raw_mean="mean", product_review_count="count", product_score_std="std")
        .reset_index()
    )

    # std is NaN for single-review users/products — fill with global std
    global_std = train_labeled["Score"].std()
    user_stats["user_score_std"] = user_stats["user_score_std"].fillna(global_std)
    product_stats["product_score_std"] = product_stats["product_score_std"].fillna(global_std)

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
    df = df.merge(
        user_stats[["UserId", "user_mean_score", "user_review_count", "user_score_std"]],
        on="UserId", how="left"
    )
    df = df.merge(
        product_stats[["ProductId", "product_mean_score", "product_review_count", "product_score_std"]],
        on="ProductId", how="left"
    )

    df["user_mean_score"]      = df["user_mean_score"].fillna(global_mean)
    df["user_review_count"]    = df["user_review_count"].fillna(0)
    df["user_score_std"]       = df["user_score_std"].fillna(global_std)
    df["product_mean_score"]   = df["product_mean_score"].fillna(global_mean)
    df["product_review_count"] = df["product_review_count"].fillna(0)
    df["product_score_std"]    = df["product_score_std"].fillna(global_std)

    df["user_bias"]         = df["user_mean_score"] - global_mean
    df["product_bias"]      = df["product_mean_score"] - global_mean
    df["user_product_bias"] = df["user_bias"] + df["product_bias"]

    return df[["user_mean_score", "user_review_count", "user_score_std",
               "product_mean_score", "product_review_count", "product_score_std",
               "user_bias", "product_bias", "user_product_bias"]].values


def build_bias_features_loo(train_labeled, shrinkage=10):
    """
    Leave-one-out bias features for training data only.
    Each sample's user/product mean is computed WITHOUT that sample's score,
    eliminating target leakage when training tree models.
    """
    global_mean = train_labeled["Score"].mean()
    global_std  = train_labeled["Score"].std()

    # Per-user aggregates across the whole training fold
    user_agg = train_labeled.groupby("UserId")["Score"].agg(["sum", "count", "std"]).rename(
        columns={"sum": "u_sum", "count": "u_count", "std": "u_std"}
    )
    product_agg = train_labeled.groupby("ProductId")["Score"].agg(["sum", "count", "std"]).rename(
        columns={"sum": "p_sum", "count": "p_count", "std": "p_std"}
    )

    df = train_labeled[["UserId", "ProductId", "Score"]].copy()
    df = df.join(user_agg,    on="UserId",    how="left")
    df = df.join(product_agg, on="ProductId", how="left")

    # LOO sum and count (exclude current sample)
    df["u_sum_loo"]   = df["u_sum"]   - df["Score"]
    df["u_count_loo"] = df["u_count"] - 1
    df["p_sum_loo"]   = df["p_sum"]   - df["Score"]
    df["p_count_loo"] = df["p_count"] - 1

    # LOO raw mean (fall back to global_mean when user/product has only 1 review)
    df["u_raw_loo"] = np.where(
        df["u_count_loo"] > 0, df["u_sum_loo"] / df["u_count_loo"], global_mean
    )
    df["p_raw_loo"] = np.where(
        df["p_count_loo"] > 0, df["p_sum_loo"] / df["p_count_loo"], global_mean
    )

    # Smoothed LOO means
    df["user_mean_score"] = (
        (df["u_count_loo"] * df["u_raw_loo"] + shrinkage * global_mean)
        / (df["u_count_loo"] + shrinkage)
    )
    df["product_mean_score"] = (
        (df["p_count_loo"] * df["p_raw_loo"] + shrinkage * global_mean)
        / (df["p_count_loo"] + shrinkage)
    )

    df["user_review_count"]    = df["u_count_loo"].clip(lower=0)
    df["product_review_count"] = df["p_count_loo"].clip(lower=0)
    df["user_score_std"]       = df["u_std"].fillna(global_std)
    df["product_score_std"]    = df["p_std"].fillna(global_std)
    df["user_bias"]            = df["user_mean_score"] - global_mean
    df["product_bias"]         = df["product_mean_score"] - global_mean
    df["user_product_bias"]    = df["user_bias"] + df["product_bias"]

    return df[["user_mean_score", "user_review_count", "user_score_std",
               "product_mean_score", "product_review_count", "product_score_std",
               "user_bias", "product_bias", "user_product_bias"]].values


def build_baseline(train_labeled, all_data, shrinkage=10):
    """
    Bias baseline for two-stage residual prediction.
    baseline = clip(user_smoothed_mean + product_smoothed_mean - global_mean, 1, 5)
    """
    global_mean = train_labeled["Score"].mean()

    user_stats = (
        train_labeled.groupby("UserId")["Score"]
        .agg(u_mean="mean", u_count="count")
        .reset_index()
    )
    product_stats = (
        train_labeled.groupby("ProductId")["Score"]
        .agg(p_mean="mean", p_count="count")
        .reset_index()
    )

    user_stats["user_smoothed"] = (
        (user_stats["u_count"] * user_stats["u_mean"] + shrinkage * global_mean)
        / (user_stats["u_count"] + shrinkage)
    )
    product_stats["product_smoothed"] = (
        (product_stats["p_count"] * product_stats["p_mean"] + shrinkage * global_mean)
        / (product_stats["p_count"] + shrinkage)
    )

    df = all_data[["UserId", "ProductId"]].copy()
    df = df.merge(user_stats[["UserId", "user_smoothed"]], on="UserId", how="left")
    df = df.merge(product_stats[["ProductId", "product_smoothed"]], on="ProductId", how="left")

    df["user_smoothed"]    = df["user_smoothed"].fillna(global_mean)
    df["product_smoothed"] = df["product_smoothed"].fillna(global_mean)

    baseline = df["user_smoothed"].values + df["product_smoothed"].values - global_mean
    return np.clip(baseline, 1, 5)


_NEGATION_WORDS = frozenset([
    "not", "no", "never", "neither", "nor", "nobody", "nothing", "nowhere",
    "don't", "dont", "didn't", "didnt", "won't", "wont", "wouldn't", "wouldnt",
    "couldn't", "couldnt", "can't", "cant", "isn't", "isnt", "aren't", "arent",
    "wasn't", "wasnt", "weren't", "werent", "doesn't", "doesnt", "haven't", "havent",
])

_EXTREME_POS_WORDS = frozenset([
    "amazing", "excellent", "perfect", "love", "best", "fantastic", "wonderful",
    "awesome", "outstanding", "superb", "brilliant", "incredible", "delicious",
    "favorite", "favourite", "exceptional", "phenomenal",
])

_EXTREME_NEG_WORDS = frozenset([
    "terrible", "awful", "horrible", "worst", "hate", "disgusting", "disappointing",
    "useless", "garbage", "trash", "pathetic", "dreadful", "atrocious", "abysmal",
    "revolting", "nauseating", "inedible", "defective", "broken", "rotten",
])


def build_numeric_features(df):
    """Extract hand-crafted numeric features: helpfulness ratios, text lengths, punctuation, and time."""
    X = pd.DataFrame(index=df.index)

    helpfulness_denom = df["HelpfulnessDenominator"].fillna(0)
    helpfulness_num   = df["HelpfulnessNumerator"].fillna(0)

    X["HelpfulnessRatio"]       = helpfulness_num / (helpfulness_denom + 1)
    X["HelpfulnessDenominator"] = helpfulness_denom
    X["HelpfulnessNumerator"]   = helpfulness_num
    X["LogHelpfulnessDenom"]    = np.log1p(helpfulness_denom)  # less skewed than raw count

    text_col    = df["Text"].fillna("")
    summary_col = df["Summary"].fillna("")

    X["TextLength"]      = text_col.apply(len)
    X["SummaryLength"]   = summary_col.apply(len)
    X["TextWordCount"]   = text_col.apply(lambda x: len(x.split()))
    X["SummaryWordCount"] = summary_col.apply(lambda x: len(x.split()))

    # Text/summary length ratio — very short summary + long rant can signal low ratings
    X["TextSummaryLengthRatio"] = X["TextLength"] / (X["SummaryLength"] + 1)

    X["NumExclamation"] = text_col.apply(lambda x: x.count("!"))
    X["NumQuestion"]    = text_col.apply(lambda x: x.count("?"))

    # Normalized punctuation density (raw counts are correlated with review length)
    X["ExclPerWord"]     = X["NumExclamation"] / (X["TextWordCount"] + 1)
    X["QuestionPerWord"] = X["NumQuestion"]    / (X["TextWordCount"] + 1)

    # Repeated punctuation (!! or ?? or more) — signals strong emotion
    X["RepeatedExcl"] = text_col.apply(lambda x: sum(1 for i in range(len(x) - 1) if x[i] == "!" and x[i+1] == "!"))
    X["RepeatedQues"] = text_col.apply(lambda x: sum(1 for i in range(len(x) - 1) if x[i] == "?" and x[i+1] == "?"))

    X["UppercaseRatio"] = text_col.apply(
        lambda x: sum(1 for c in x if c.isupper()) / (len(x) + 1)
    )

    # ALL-CAPS words (len >= 2) — shouting indicates strong feeling
    X["AllCapsWordCount"] = text_col.apply(
        lambda x: sum(1 for w in x.split() if len(w) >= 2 and w.isupper())
    )

    X["UniqueWordRatio"] = text_col.apply(
        lambda x: len(set(x.lower().split())) / (len(x.split()) + 1)
    )
    X["AvgWordLength"] = text_col.apply(
        lambda x: np.mean([len(w) for w in x.split()]) if x.split() else 0
    )

    # Negation and extreme-sentiment word counts
    X["NegationCount"]  = text_col.apply(
        lambda x: sum(1 for w in x.lower().split() if w.rstrip(".,!?;:") in _NEGATION_WORDS)
    )
    X["ExtremePosCount"] = text_col.apply(
        lambda x: sum(1 for w in x.lower().split() if w.rstrip(".,!?;:") in _EXTREME_POS_WORDS)
    )
    X["ExtremeNegCount"] = text_col.apply(
        lambda x: sum(1 for w in x.lower().split() if w.rstrip(".,!?;:") in _EXTREME_NEG_WORDS)
    )
    # Net extreme sentiment balance (positive - negative extreme words)
    X["ExtremeBalance"] = X["ExtremePosCount"] - X["ExtremeNegCount"]

    # Sentence-level features
    X["SentenceCount"] = text_col.apply(
        lambda x: max(1, x.count(".") + x.count("!") + x.count("?"))
    )
    X["AvgSentenceLength"] = X["TextWordCount"] / X["SentenceCount"]

    # Helpfulness × text length interaction (engaged, helpful long reviews)
    X["HelpfulnessXLength"] = X["HelpfulnessRatio"] * np.log1p(X["TextLength"])

    dt = pd.to_datetime(df["Time"], unit="s", errors="coerce")
    X["Year"]    = dt.dt.year.fillna(0)
    X["Month"]   = dt.dt.month.fillna(0)
    X["Quarter"] = dt.dt.quarter.fillna(0)

    return X.fillna(0)


def prepare_training_data(training_df, text_column="Text", max_features=30000, ngram_range=(1, 2)):
    """
    Fit all transformers on the labeled rows and return a 10-tuple:
    (X_sparse, y_residual, baseline, tfidf_summary, tfidf_text,
     tfidf_char, tfidf_char_summary, svd, numeric_columns, labeled_df)

    y_residual = score - bias_baseline so that Ridge/ExtraTrees learn
    only the part the baseline cannot explain.
    """
    # Labeled = has Score; unlabeled = test set
    labeled = training_df.dropna(subset=["Score"]).copy()

    y = labeled["Score"].values

    # Two-stage: compute bias baseline, train text model on residuals
    baseline = build_baseline(labeled, labeled)
    y_residual = y - baseline

    # Numeric + sentiment features (bias is in the baseline, not in X)
    num_feats = build_numeric_features(labeled)
    sentiment = build_sentiment_features(labeled)
    num_feats = pd.concat([num_feats, sentiment], axis=1)

    numeric_columns = num_feats.columns.tolist()

    # Custom stopwords
    movie_stop = {
        "movie", "film", "watch", "watched", "one", "really",
        "also", "even", "get", "got", "make", "made"
    }
    stop_words = list(text.ENGLISH_STOP_WORDS.union(movie_stop))

    base_tfidf_kwargs = dict(
        lowercase=True, stop_words=stop_words, ngram_range=ngram_range,
        min_df=2, max_df=0.9, sublinear_tf=True, norm="l2",
        smooth_idf=True, use_idf=True,
    )

    # Separate TF-IDF for Summary (high-signal, short) and Text (longer, noisier)
    tfidf_summary = TfidfVectorizer(max_features=10000, **base_tfidf_kwargs)
    tfidf_text    = TfidfVectorizer(max_features=max_features, **base_tfidf_kwargs)

    # Character n-grams on Text (3–5): catch morphological patterns like "terribl", "excellen"
    tfidf_char = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5),
        min_df=5, max_df=0.9, sublinear_tf=True, norm="l2",
        max_features=20000, lowercase=True,
    )

    # Character n-grams on Summary (3–5): short high-signal phrases like "must see", "avoid"
    tfidf_char_summary = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5),
        min_df=5, max_df=0.9, sublinear_tf=True, norm="l2",
        max_features=10000, lowercase=True,
    )

    X_summary      = tfidf_summary.fit_transform(labeled["Summary"].fillna("").astype(str))
    X_text_body    = tfidf_text.fit_transform(labeled["Text"].fillna("").astype(str))
    X_char         = tfidf_char.fit_transform(labeled["Text"].fillna("").astype(str))
    X_char_summary = tfidf_char_summary.fit_transform(labeled["Summary"].fillna("").astype(str))
    X_text = hstack([X_summary, X_text_body, X_char, X_char_summary])
    print("Summary TF-IDF:", X_summary.shape,
          " Text TF-IDF:", X_text_body.shape,
          " Char n-grams (Text):", X_char.shape,
          " Char n-grams (Summary):", X_char_summary.shape)

    # LSA on the combined sparse matrix
    svd = TruncatedSVD(n_components=200, random_state=42)
    X_lsa = svd.fit_transform(X_text)
    X_lsa = Normalizer(copy=False).fit_transform(X_lsa)
    print(f"LSA explained variance: {svd.explained_variance_ratio_.sum():.3f}")

    X_num_sparse = csr_matrix(num_feats.values)
    X_lsa_sparse = csr_matrix(X_lsa)
    X = hstack([X_num_sparse, X_text, X_lsa_sparse])

    return (X, y_residual, baseline,
            tfidf_summary, tfidf_text, tfidf_char, tfidf_char_summary,
            svd, numeric_columns, labeled)


def prepare_test_data(test_df, tfidf_summary, tfidf_text, tfidf_char, tfidf_char_summary,
                      svd, numeric_columns, labeled_df):
    """
    Apply fitted transformers from prepare_training_data() to the test set.
    Returns (X_test, baseline_test); caller reconstructs final score as
    clip(baseline_test + model.predict(X_test), 1, 5).
    """
    # Bias baseline for final prediction reconstruction
    baseline_test = build_baseline(labeled_df, test_df)

    # Numeric + sentiment features (no bias in X — bias is in baseline)
    num_feats = build_numeric_features(test_df)
    sentiment = build_sentiment_features(test_df)
    num_feats = pd.concat([num_feats, sentiment], axis=1)
    num_feats = num_feats.reindex(columns=numeric_columns, fill_value=0).fillna(0)

    X_summary      = tfidf_summary.transform(test_df["Summary"].fillna("").astype(str))
    X_text_body    = tfidf_text.transform(test_df["Text"].fillna("").astype(str))
    X_char         = tfidf_char.transform(test_df["Text"].fillna("").astype(str))
    X_char_summary = tfidf_char_summary.transform(test_df["Summary"].fillna("").astype(str))
    X_text = hstack([X_summary, X_text_body, X_char, X_char_summary])

    X_lsa = svd.transform(X_text)
    # Normalizer has no fitted state (row-wise L2 norm), so fit_transform == transform here
    X_lsa = Normalizer(copy=False).fit_transform(X_lsa)

    X_num_sparse = csr_matrix(num_feats.values)
    X_lsa_sparse = csr_matrix(X_lsa)
    X_test = hstack([X_num_sparse, X_text, X_lsa_sparse])

    return X_test, baseline_test
