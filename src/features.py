def prepare_training_data(training_df):
    dataset = training_df.dropna(subset=['Score']).copy()

    X = dataset.drop(columns=['Score'])
    y = dataset['Score']

    X = X.select_dtypes(include=['number']).copy()
    X = X.fillna(0)

    return X, y


def prepare_test_data(test_df, feature_columns):
    X_test = test_df.reindex(columns=feature_columns, fill_value=0)
    X_test = X_test.fillna(0)
    return X_test
