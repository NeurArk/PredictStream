from importlib import import_module

import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression

from utils import model, eval as evaluation, viz

PAGES = [
    "pages.data_explorer",
    "pages.modeling",
    "pages.prediction",
    "pages.report",
    "pages.time_series",
]

@pytest.mark.parametrize("mod_name", PAGES)
def test_page_importable(mod_name):
    mod = import_module(mod_name)
    assert hasattr(mod, "main")


def test_metrics_and_plots_generated():
    Xc, yc = make_classification(n_samples=30, n_features=4, random_state=0)
    dfc = pd.DataFrame(Xc, columns=[f"f{i}" for i in range(4)])
    dfc["target"] = yc
    X_train, X_test, y_train, y_test = model.train_test_split_data(dfc, "target")
    clf = model.train_logistic_regression(X_train, y_train, max_iter=50)
    preds = clf.predict(X_test)
    metrics_c = evaluation.performance_metrics(y_test, preds, problem_type="classification")
    assert "accuracy" in metrics_c
    cm = viz.confusion_matrix_plot(y_test, preds)
    roc = viz.roc_curve_plot(y_test, clf.predict_proba(X_test)[:, 1])
    pr = viz.precision_recall_curve_plot(y_test, clf.predict_proba(X_test)[:, 1])
    assert cm.data and roc.data and pr.data

    Xr, yr = make_regression(n_samples=30, n_features=4, noise=0.1, random_state=0)
    dfr = pd.DataFrame(Xr, columns=[f"f{i}" for i in range(4)])
    dfr["target"] = yr
    X_train, X_test, y_train, y_test = model.train_test_split_data(dfr, "target")
    reg = model.train_linear_regression(X_train, y_train)
    preds_r = reg.predict(X_test)
    metrics_r = evaluation.performance_metrics(y_test, preds_r, problem_type="regression")
    assert "r2" in metrics_r
    avp = viz.actual_vs_predicted_plot(y_test, preds_r)
    resid = viz.residual_plot(y_test, preds_r)
    assert avp.data and resid.data


def test_modeling_page_loads(monkeypatch):
    import streamlit as st
    from pages import modeling

    df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "target": [0, 1]})
    st.session_state.clear()
    st.session_state["data"] = df
    monkeypatch.setattr(modeling.ui, "apply_branding", lambda: None)
    modeling.main()


def test_modeling_page_widgets_exist():
    with open("pages/modeling.py", "r", encoding="utf-8") as f:
        content = f.read()
    assert "st.selectbox(\"Dataset\"" in content
    assert "st.multiselect(\"Models\"" in content
    assert "Train Models" in content
    assert "export_model" in content


def test_data_explorer_visualization_widgets_exist():
    with open("utils/components.py", "r", encoding="utf-8") as f:
        content = f.read()
    assert "Generate Histogram" in content
    assert "Generate Box Plot" in content
    assert "Generate Violin Plot" in content
    assert "Generate Heatmap" in content


def test_pair_plot_export(tmp_path):
    df = pd.DataFrame({
        "a": [1, 2, 3, 4],
        "b": [4, 3, 2, 1],
        "cat": ["x", "y", "x", "y"],
    })
    fig = viz.pair_plot(df, columns=["a", "b"], hue="cat")
    out_file = tmp_path / "pair.png"
    viz.export_figure(fig, out_file)
    assert out_file.exists() and out_file.stat().st_size > 0


def test_time_series_page_runs(monkeypatch):
    import streamlit as st
    from pages import time_series

    df = pd.DataFrame(
        {
            "date": pd.date_range("2021-01-01", periods=5, freq="D"),
            "value": range(5),
        }
    )
    st.session_state.clear()
    st.session_state["data"] = df
    st.session_state["datetime_cols"] = ["date"]
    monkeypatch.setattr(time_series.ui, "apply_branding", lambda: None)
    time_series.main()


def test_time_series_page_contents():
    with open("pages/time_series.py", "r", encoding="utf-8") as f:
        content = f.read()
    assert "time_series_plot" in content
    assert "decomposition_plot" in content


def test_time_series_page_forecast_widgets():
    with open("pages/time_series.py", "r", encoding="utf-8") as f:
        content = f.read()
    assert "Forecast Model" in content
    assert "Forecast Horizon" in content


def test_datetime_cols_persist_after_transforms():
    import streamlit as st
    from utils import transform, eda

    df = pd.DataFrame(
        {
            "date": pd.date_range("2021-01-01", periods=3, freq="D"),
            "value": [1.0, 2.0, 3.0],
        }
    )
    st.session_state.clear()
    st.session_state["data"] = df

    df_trans = transform.scale_features(df.copy(), ["value"], method="standard")
    st.session_state["data"] = df_trans
    st.session_state["datetime_cols"] = eda.detect_datetime_columns(df_trans)

    assert st.session_state["datetime_cols"] == ["date"]

