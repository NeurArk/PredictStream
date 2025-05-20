import pandas as pd
from sklearn.datasets import make_classification

from utils import model


def sample_df():
    X, y = make_classification(n_samples=50, n_features=4, n_classes=2, random_state=0)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])
    df["target"] = y
    return df


def test_train_test_split():
    df = sample_df()
    X_train, X_test, y_train, y_test = model.train_test_split_data(
        df, "target", test_size=0.2, random_state=42
    )
    assert len(X_train) + len(X_test) == len(df)
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)


def test_logistic_regression_training():
    df = sample_df()
    X_train, X_test, y_train, y_test = model.train_test_split_data(df, "target")
    clf = model.train_logistic_regression(X_train, y_train, max_iter=100)
    preds = clf.predict(X_test)
    assert len(preds) == len(y_test)


def test_random_forest_training():
    df = sample_df()
    X_train, X_test, y_train, y_test = model.train_test_split_data(df, "target")
    clf = model.train_random_forest_classifier(X_train, y_train, n_estimators=10)
    preds = clf.predict(X_test)
    assert len(preds) == len(y_test)


def test_cross_validation():
    df = sample_df()
    X = df.drop(columns=["target"])
    y = df["target"]
    clf = model.train_logistic_regression(X, y, max_iter=100)
    scores = model.cross_validate_model(clf, X, y, cv=3)
    assert len(scores) == 3


def test_model_caching():
    df = sample_df()
    X_train, _, y_train, _ = model.train_test_split_data(df, "target")

    @model.cache_model
    def custom_train(x, y):
        return model.train_logistic_regression(x, y, max_iter=50)

    first = custom_train(X_train, y_train)
    second = custom_train(X_train, y_train)
    assert first is second
