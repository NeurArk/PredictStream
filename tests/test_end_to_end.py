import pandas as pd
from sklearn.datasets import make_classification, make_regression
from utils import config, data, eda, model, eval as evaluation


def test_classification_workflow():
    X, y = make_classification(n_samples=50, n_features=4, random_state=0)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])
    df["target"] = y
    summary = eda.summary_statistics(df)
    assert "f0" in summary.columns
    X_train, X_test, y_train, y_test = model.train_test_split_data(df, "target")
    clf = model.train_logistic_regression(X_train, y_train, max_iter=50)
    preds = clf.predict(X_test)
    metrics = evaluation.performance_metrics(y_test, preds, problem_type="classification")
    assert "accuracy" in metrics


def test_regression_workflow():
    X, y = make_regression(n_samples=50, n_features=4, noise=0.1, random_state=0)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])
    df["target"] = y
    X_train, X_test, y_train, y_test = model.train_test_split_data(df, "target")
    reg = model.train_linear_regression(X_train, y_train)
    preds = reg.predict(X_test)
    metrics = evaluation.performance_metrics(y_test, preds, problem_type="regression")
    assert "r2" in metrics


def test_xgboost_workflows():
    Xc, yc = make_classification(n_samples=30, n_features=4, random_state=0)
    dfc = pd.DataFrame(Xc, columns=[f"f{i}" for i in range(4)])
    dfc["target"] = yc
    X_train, X_test, y_train, y_test = model.train_test_split_data(dfc, "target")
    clf = model.train_xgboost_classifier(X_train, y_train, n_estimators=10)
    preds = clf.predict(X_test)
    metrics = evaluation.performance_metrics(y_test, preds, problem_type="classification")
    assert "accuracy" in metrics

    Xr, yr = make_regression(n_samples=30, n_features=4, noise=0.1, random_state=0)
    dfr = pd.DataFrame(Xr, columns=[f"f{i}" for i in range(4)])
    dfr["target"] = yr
    X_train, X_test, y_train, y_test = model.train_test_split_data(dfr, "target")
    reg = model.train_xgboost_regressor(X_train, y_train, n_estimators=10)
    preds = reg.predict(X_test)
    metrics = evaluation.performance_metrics(y_test, preds, problem_type="regression")
    assert "r2" in metrics
