import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LogisticRegression, LinearRegression

from utils import predict


def test_predict_single_and_batch_classification():
    X, y = make_classification(
        n_samples=30,
        n_features=3,
        n_informative=2,
        n_redundant=0,
        random_state=0,
    )
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(3)])
    clf = LogisticRegression(max_iter=100).fit(df, y)

    single = predict.predict_single(clf, df.iloc[0])
    batch = predict.predict_batch(clf, df)

    assert single in {0, 1}
    assert len(batch) == len(df)


def test_prediction_plot_and_export(tmp_path):
    X, y = make_regression(n_samples=30, n_features=3, noise=0.1, random_state=0)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(3)])
    reg = LinearRegression().fit(df, y)
    preds = predict.predict_batch(reg, df)

    fig = predict.prediction_plot(pd.Series(y), preds)
    assert fig

    csv_path = tmp_path / "pred.csv"
    xls_path = tmp_path / "pred.xlsx"
    predict.export_predictions(preds, csv_path)
    predict.export_predictions(preds, xls_path)
    assert csv_path.exists() and xls_path.exists()


def test_model_and_project_export(tmp_path):
    X, y = make_classification(n_samples=20, n_features=4, random_state=0)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])
    clf = LogisticRegression(max_iter=50).fit(df, y)

    model_path = tmp_path / "model.joblib"
    predict.export_model(clf, model_path)
    loaded = predict.load_model(model_path)
    preds = predict.predict_batch(loaded, df)

    project_path = tmp_path / "project.joblib"
    predict.save_project(project_path, model=loaded, data=df, predictions=preds)
    project = predict.load_project(project_path)

    assert isinstance(project["model"], LogisticRegression)
    assert len(project["data"]) == len(df)
    assert len(project["predictions"]) == len(preds)
