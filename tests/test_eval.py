import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestRegressor

from utils import eval as evaluation
from utils import viz


def sample_classification():
    X, y = make_classification(n_samples=50, n_features=4, random_state=0)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(4)]), pd.Series(y)


def sample_regression():
    X, y = make_regression(n_samples=50, n_features=4, noise=0.1, random_state=0)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(4)]), pd.Series(y)


def test_performance_metrics_classification():
    X, y = sample_classification()
    clf = LogisticRegression(max_iter=100).fit(X, y)
    preds = clf.predict(X)
    metrics = evaluation.performance_metrics(y, preds, problem_type="classification")
    assert set(metrics) == {"accuracy", "precision", "recall", "f1"}


def test_performance_metrics_regression():
    X, y = sample_regression()
    reg = LinearRegression().fit(X, y)
    preds = reg.predict(X)
    metrics = evaluation.performance_metrics(y, preds, problem_type="regression")
    assert set(metrics) == {"mae", "mse", "rmse", "r2"}


def test_confusion_matrix_and_curves():
    X, y = sample_classification()
    clf = LogisticRegression(max_iter=50).fit(X, y)
    preds = clf.predict(X)
    prob = clf.predict_proba(X)[:, 1]
    cm = evaluation.confusion_matrix(y, preds)
    assert cm.shape[0] == cm.shape[1]
    fig = viz.confusion_matrix_plot(y, preds)
    assert fig.data
    roc = viz.roc_curve_plot(y, prob)
    pr = viz.precision_recall_curve_plot(y, prob)
    assert roc.data and pr.data


def test_regression_plots_and_importance():
    X, y = sample_regression()
    reg = RandomForestRegressor(n_estimators=10, random_state=0).fit(X, y)
    preds = reg.predict(X)
    avp = viz.actual_vs_predicted_plot(y, preds)
    residual = viz.residual_plot(y, preds)
    imp = viz.feature_importance_plot(reg, list(X.columns))
    assert avp.data and residual.data and imp.data


def test_shap_summary_plot():
    X, y = sample_regression()
    reg = RandomForestRegressor(n_estimators=5, random_state=0).fit(X, y)
    fig = viz.shap_summary_plot(reg, X.head())
    assert hasattr(fig, "axes")
