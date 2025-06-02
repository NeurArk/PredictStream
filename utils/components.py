"""Reusable Streamlit page components."""

from __future__ import annotations

from pathlib import Path
import tempfile
import logging

import pandas as pd
import streamlit as st

from . import eda, eval as evaluation, model, transform, viz, ui
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


def visualization_section(data: pd.DataFrame) -> None:
    """Render interactive visualization tools for the given data."""
    st.markdown("---")  # Add separator before visualization section
    st.subheader("Pair Plot")
    with st.expander("Create Pair Plot"):
        pair_cols = st.multiselect(
            "Columns",
            options=data.columns.tolist(),
            default=data.select_dtypes(include="number").columns.tolist(),
            key="pair_cols",
        )
        hue_col = st.selectbox(
            "Color By",
            options=["None"] + data.columns.tolist(),
            index=0,
            key="pair_hue",
        )
        export_fmt = st.selectbox(
            "Export Format",
            ["png", "jpg"],
            key="pair_fmt",
        )
        if st.button("Generate Pair Plot") and pair_cols:
            hue = None if hue_col == "None" else hue_col
            fig_pair = viz.pair_plot(data, columns=pair_cols, hue=hue)
            st.pyplot(fig_pair)
            with tempfile.NamedTemporaryFile(suffix=f".{export_fmt}") as tmp:
                viz.export_figure(fig_pair, Path(tmp.name))
                tmp.seek(0)
                st.download_button(
                    "Download Plot",
                    data=tmp.read(),
                    file_name=f"pair_plot.{export_fmt}",
                    mime=f"image/{export_fmt}",
                )

    st.markdown("---")
    st.subheader("Histogram")
    with st.expander("Create Histogram"):
        num_cols = data.select_dtypes(include="number").columns.tolist()
        hist_col = st.selectbox(
            "Column",
            options=num_cols,
            key="hist_col",
        )
        bins = st.slider(
            "Bins",
            5,
            100,
            20,
            step=5,
            key="hist_bins",
        )
        density = st.checkbox("Density", key="hist_density")
        export_fmt_h = st.selectbox(
            "Export Format",
            ["png", "jpg"],
            key="hist_fmt",
        )
        if st.button("Generate Histogram") and hist_col:
            fig_hist = viz.histogram(
                data,
                hist_col,
                bins=bins,
                density=density,
            )
            with st.container():
                ui.display_plotly_chart(fig_hist, use_container_width=True)
                st.markdown("&nbsp;")  # Add space after chart
            with tempfile.NamedTemporaryFile(suffix=f".{export_fmt_h}") as tmp:
                viz.export_figure(fig_hist, Path(tmp.name))
                tmp.seek(0)
                st.download_button(
                    "Download Histogram",
                    data=tmp.read(),
                    file_name=f"histogram.{export_fmt_h}",
                    mime=f"image/{export_fmt_h}",
                )

    st.markdown("---")
    st.subheader("Box Plot")
    with st.expander("Create Box Plot"):
        x_col = st.selectbox(
            "X Column",
            options=data.columns.tolist(),
            key="box_x",
        )
        num_cols = data.select_dtypes(include="number").columns.tolist()
        y_col = st.selectbox(
            "Y Column",
            options=num_cols,
            key="box_y",
        )
        export_fmt_b = st.selectbox(
            "Export Format",
            ["png", "jpg"],
            key="box_fmt",
        )
        if st.button("Generate Box Plot") and x_col and y_col:
            fig_box = viz.box_plot(data, x=x_col, y=y_col)
            with st.container():
                ui.display_plotly_chart(fig_box, use_container_width=True)
                st.markdown("&nbsp;")
            with tempfile.NamedTemporaryFile(suffix=f".{export_fmt_b}") as tmp:
                viz.export_figure(fig_box, Path(tmp.name))
                tmp.seek(0)
                st.download_button(
                    "Download Box Plot",
                    data=tmp.read(),
                    file_name=f"box_plot.{export_fmt_b}",
                    mime=f"image/{export_fmt_b}",
                )

    st.markdown("---")
    st.subheader("Violin Plot")
    with st.expander("Create Violin Plot"):
        x_col_v = st.selectbox(
            "X Column",
            options=data.columns.tolist(),
            key="violin_x",
        )
        num_cols = data.select_dtypes(include="number").columns.tolist()
        y_col_v = st.selectbox(
            "Y Column",
            options=num_cols,
            key="violin_y",
        )
        export_fmt_v = st.selectbox(
            "Export Format",
            ["png", "jpg"],
            key="violin_fmt",
        )
        if st.button("Generate Violin Plot") and x_col_v and y_col_v:
            fig_violin = viz.violin_plot(data, x=x_col_v, y=y_col_v)
            with st.container():
                ui.display_plotly_chart(fig_violin, use_container_width=True)
                st.markdown("&nbsp;")
            with tempfile.NamedTemporaryFile(suffix=f".{export_fmt_v}") as tmp:
                viz.export_figure(fig_violin, Path(tmp.name))
                tmp.seek(0)
                st.download_button(
                    "Download Violin Plot",
                    data=tmp.read(),
                    file_name=f"violin_plot.{export_fmt_v}",
                    mime=f"image/{export_fmt_v}",
                )

    st.markdown("---")
    st.subheader("Heatmap")
    with st.expander("Create Heatmap"):
        heat_cols = st.multiselect(
            "Columns",
            options=data.select_dtypes(include="number").columns.tolist(),
            default=data.select_dtypes(include="number").columns.tolist(),
            key="heat_cols",
        )
        export_fmt_hm = st.selectbox(
            "Export Format",
            ["png", "jpg"],
            key="heat_fmt",
        )
        if st.button("Generate Heatmap") and heat_cols:
            fig_heat = viz.heatmap(data[heat_cols])
            with st.container():
                ui.display_plotly_chart(fig_heat, use_container_width=True)
                st.markdown("&nbsp;")
            with tempfile.NamedTemporaryFile(suffix=f".{export_fmt_hm}") as tmp:
                viz.export_figure(fig_heat, Path(tmp.name))
                tmp.seek(0)
                st.download_button(
                    "Download Heatmap",
                    data=tmp.read(),
                    file_name=f"heatmap.{export_fmt_hm}",
                    mime=f"image/{export_fmt_hm}",
                )

    st.markdown("---")
    st.subheader("Insights")
    for insight in eda.data_insights_summary(data):
        st.write(f"- {insight}")


@st.cache_resource
def _cached_transformations(
    df: pd.DataFrame,
    missing_strategy: str,
    encode_cols: tuple[str, ...],
    encode_method: str,
    scale_cols: tuple[str, ...],
    scale_method: str,
) -> pd.DataFrame:
    """Return transformed DataFrame with caching."""
    df_trans = df.copy()
    if missing_strategy == "Drop rows":
        df_trans = transform.handle_missing_values(df_trans, strategy="drop")
    elif missing_strategy == "Fill Mean":
        df_trans = transform.handle_missing_values(df_trans, strategy="mean")
    elif missing_strategy == "Fill Median":
        df_trans = transform.handle_missing_values(df_trans, strategy="median")
    elif missing_strategy == "Fill Mode":
        df_trans = transform.handle_missing_values(df_trans, strategy="mode")
    if encode_cols:
        method = "onehot" if encode_method == "One-Hot" else "label"
        df_trans = transform.encode_features(df_trans, list(encode_cols), method=method)
    if scale_cols:
        method = "standard" if scale_method == "Standard" else "minmax"
        df_trans = transform.scale_features(df_trans, list(scale_cols), method=method)
    return df_trans


def transformation_section(data: pd.DataFrame) -> pd.DataFrame:
    """Provide UI to apply common data transformations."""
    st.markdown("---")
    st.subheader("Data Transformation")
    with st.expander("Transform Options"):
        missing_strategy = st.selectbox(
            "Missing Value Strategy",
            ["None", "Drop rows", "Fill Mean", "Fill Median", "Fill Mode"],
        )
        encode_cols = st.multiselect(
            "Columns to Encode",
            options=data.select_dtypes(exclude="number").columns.tolist(),
        )
        encode_method = st.selectbox(
            "Encoding Method",
            ["One-Hot", "Label"],
        )
        scale_cols = st.multiselect(
            "Columns to Scale",
            options=data.select_dtypes(include="number").columns.tolist(),
        )
        scale_method = st.selectbox(
            "Scaling Method",
            ["Standard", "Min-Max"],
        )
    if st.button("Apply Transformations"):
        try:
            df_trans = _cached_transformations(
                data,
                missing_strategy,
                tuple(encode_cols),
                encode_method,
                tuple(scale_cols),
                scale_method,
            )
        except (ValueError, KeyError, TypeError) as exc:
            st.error(f"Transformation error: {exc}")
        else:
            st.session_state["data"] = df_trans
            st.session_state["datetime_cols"] = eda.detect_datetime_columns(df_trans)
            data = df_trans
            st.success("Transformations applied!")
    return data


def classification_training_section(data: pd.DataFrame) -> None:
    """UI for training classification models and displaying metrics."""
    st.markdown("---")
    st.subheader("Model Training - Classification")
    target = st.selectbox(
        "Target Column",
        options=data.columns,
        help="Column you want to predict",
    )
    feature_cols = st.multiselect(
        "Feature Columns",
        options=[c for c in data.columns if c != target],
        default=[c for c in data.columns if c != target],
        help="Columns used as model inputs",
    )
    test_size = st.slider(
        "Test Size",
        0.1,
        0.5,
        0.2,
        step=0.05,
        help="Fraction of data used for testing",
    )
    model_name = st.selectbox(
        "Model",
        ["Logistic Regression", "Random Forest", "XGBoost"],
        help="Choose a classification algorithm",
    )

    hyperparams: dict[str, float] = {}
    if model_name == "Logistic Regression":
        hyperparams["C"] = st.number_input(
            "C",
            0.01,
            10.0,
            1.0,
            step=0.01,
            help="Inverse of regularization strength",
        )
    elif model_name == "Random Forest":
        hyperparams["n_estimators"] = st.slider(
            "n_estimators",
            10,
            200,
            100,
            step=10,
            help="Number of trees",
        )
    else:
        hyperparams["n_estimators"] = st.slider(
            "n_estimators",
            50,
            300,
            100,
            step=10,
            help="Number of boosting rounds",
        )
        hyperparams["learning_rate"] = st.number_input(
            "learning_rate",
            0.01,
            1.0,
            0.1,
            step=0.01,
            help="Learning rate",
        )
        hyperparams["max_depth"] = st.slider(
            "max_depth",
            1,
            10,
            3,
            step=1,
            help="Maximum tree depth",
        )

    if st.button("Train Model") and feature_cols:
        progress = st.progress(0)
        try:
            df_model = data[feature_cols + [target]]
            X_train, X_test, y_train, y_test = model.train_test_split_data(
                df_model,
                target,
                test_size=test_size,
                random_state=42,
            )
            progress.progress(25)
            if model_name == "Logistic Regression":
                clf = model.train_logistic_regression(
                    X_train,
                    y_train,
                    C=hyperparams.get("C", 1.0),
                    max_iter=200,
                )
            elif model_name == "Random Forest":
                clf = model.train_random_forest_classifier(
                    X_train,
                    y_train,
                    n_estimators=hyperparams.get("n_estimators", 100),
                    random_state=42,
                )
            else:
                clf = model.train_xgboost_classifier(
                    X_train,
                    y_train,
                    n_estimators=hyperparams.get("n_estimators", 100),
                    learning_rate=hyperparams.get("learning_rate", 0.1),
                    max_depth=hyperparams.get("max_depth", 3),
                    random_state=42,
                )
            progress.progress(75)
            scores = model.cross_validate_model(clf, X_train, y_train, cv=5)
            progress.progress(100)
            st.write("Cross-validation scores:", scores)
            st.write("Mean accuracy:", float(scores.mean()))

            y_pred = clf.predict(X_test)
            metrics = evaluation.performance_metrics(
                y_test, y_pred, problem_type="classification"
            )
            st.subheader("Test Metrics")
            st.json(metrics)

            fig_cm = viz.confusion_matrix_plot(y_test, y_pred)
            with st.container():
                ui.display_plotly_chart(fig_cm, use_container_width=True)
                st.markdown("&nbsp;")
            with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
                viz.export_figure(fig_cm, Path(tmp.name))
                tmp.seek(0)
                st.download_button(
                    "Download Confusion Matrix",
                    data=tmp.read(),
                    file_name="confusion_matrix.png",
                    mime="image/png",
                )

            if hasattr(clf, "predict_proba"):
                y_score = clf.predict_proba(X_test)[:, 1]
            else:
                y_score = clf.decision_function(X_test)
            fig_roc = viz.roc_curve_plot(y_test, y_score)
            with st.container():
                ui.display_plotly_chart(fig_roc, use_container_width=True)
                st.markdown("&nbsp;")
            with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
                viz.export_figure(fig_roc, Path(tmp.name))
                tmp.seek(0)
                st.download_button(
                    "Download ROC Curve",
                    data=tmp.read(),
                    file_name="roc_curve.png",
                    mime="image/png",
                )

            fig_pr = viz.precision_recall_curve_plot(y_test, y_score)
            with st.container():
                ui.display_plotly_chart(fig_pr, use_container_width=True)
                st.markdown("&nbsp;")
            with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
                viz.export_figure(fig_pr, Path(tmp.name))
                tmp.seek(0)
                st.download_button(
                    "Download PR Curve",
                    data=tmp.read(),
                    file_name="pr_curve.png",
                    mime="image/png",
                )

            if st.checkbox("Show Feature Importance"):
                fig_imp = viz.feature_importance_plot(clf, feature_cols)
                with st.container():
                    ui.display_plotly_chart(fig_imp, use_container_width=True)
                    st.markdown("&nbsp;")
                with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
                    viz.export_figure(fig_imp, Path(tmp.name))
                    tmp.seek(0)
                    st.download_button(
                        "Download Feature Importance",
                        data=tmp.read(),
                        file_name="feature_importance.png",
                        mime="image/png",
                    )

            if st.checkbox("Show SHAP Summary"):
                shap_fig = viz.shap_summary_plot(clf, X_train)
                st.pyplot(shap_fig)
                with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
                    viz.export_figure(shap_fig, Path(tmp.name))
                    tmp.seek(0)
                    st.download_button(
                        "Download SHAP Summary",
                        data=tmp.read(),
                        file_name="shap_summary.png",
                        mime="image/png",
                    )
        except Exception as exc:  # pragma: no cover - training can fail for many reasons
            progress.progress(0)
            logging.getLogger(__name__).exception(
                "Classification training failed: %s", exc
            )
            st.error(f"Classification training failed: {exc}")

    st.subheader("Detected Problem Type")
    st.write(model.detect_problem_type(data[target]))


def regression_training_section(data: pd.DataFrame) -> None:
    """UI for training regression models and displaying metrics."""
    st.markdown("---")
    st.subheader("Model Training - Regression")
    target_r = st.selectbox(
        "Target Column (regression)",
        options=data.columns,
        key="target_reg",
        help="Numeric column to predict",
    )
    feature_cols_r = st.multiselect(
        "Feature Columns (reg)",
        options=[c for c in data.columns if c != target_r],
        default=[c for c in data.columns if c != target_r],
        key="features_reg",
        help="Columns used as regression inputs",
    )
    test_size_r = st.slider(
        "Test Size (reg)",
        0.1,
        0.5,
        0.2,
        step=0.05,
        key="ts_reg",
        help="Fraction of data reserved for testing",
    )
    model_name_r = st.selectbox(
        "Model (reg)",
        ["Linear Regression", "Decision Tree", "Random Forest", "XGBoost"],
        key="model_reg",
        help="Choose a regression algorithm",
    )

    hyperparams_r: dict[str, float] = {}
    if model_name_r == "Decision Tree":
        hyperparams_r["max_depth"] = st.slider(
            "max_depth",
            1,
            20,
            3,
            key="max_depth",
            help="Maximum tree depth",
        )
    elif model_name_r == "Random Forest":
        hyperparams_r["n_estimators"] = st.slider(
            "n_estimators",
            10,
            200,
            100,
            10,
            key="n_est",
            help="Number of trees",
        )
        hyperparams_r["max_depth"] = st.slider(
            "max_depth_rf",
            1,
            20,
            3,
            key="max_depth_rf",
            help="Maximum tree depth",
        )
    elif model_name_r == "XGBoost":
        hyperparams_r["n_estimators"] = st.slider(
            "n_estimators",
            50,
            300,
            100,
            10,
            key="xgb_n_est",
            help="Number of boosting rounds",
        )
        hyperparams_r["learning_rate"] = st.number_input(
            "learning_rate",
            0.01,
            1.0,
            0.1,
            step=0.01,
            key="xgb_lr",
            help="Learning rate",
        )
        hyperparams_r["max_depth"] = st.slider(
            "max_depth_xgb",
            1,
            10,
            3,
            step=1,
            key="xgb_depth",
            help="Maximum tree depth",
        )

    if st.button("Train Regression Model") and feature_cols_r:
        progress_r = st.progress(0)
        try:
            df_r = data[feature_cols_r + [target_r]]
            X_train, X_test, y_train, y_test = model.train_test_split_data(
                df_r,
                target_r,
                test_size=test_size_r,
                random_state=42,
            )
            progress_r.progress(25)
            if model_name_r == "Linear Regression":
                reg = model.train_linear_regression(X_train, y_train)
            elif model_name_r == "Decision Tree":
                reg = model.train_decision_tree_regressor(
                    X_train,
                    y_train,
                    max_depth=hyperparams_r.get("max_depth"),
                    random_state=42,
                )
            elif model_name_r == "Random Forest":
                reg = model.train_random_forest_regressor(
                    X_train,
                    y_train,
                    n_estimators=hyperparams_r.get("n_estimators", 100),
                    max_depth=hyperparams_r.get("max_depth_rf"),
                    random_state=42,
                )
            else:
                reg = model.train_xgboost_regressor(
                    X_train,
                    y_train,
                    n_estimators=hyperparams_r.get("n_estimators", 100),
                    learning_rate=hyperparams_r.get("learning_rate", 0.1),
                    max_depth=hyperparams_r.get("max_depth", 3),
                    random_state=42,
                )
            progress_r.progress(75)
            scores = model.cross_validate_model(reg, X_train, y_train, cv=5)
            progress_r.progress(100)
            st.write("CV scores:", scores)
            st.write("Mean R2:", float(scores.mean()))

            preds = reg.predict(X_test)
            metrics = evaluation.performance_metrics(
                y_test, preds, problem_type="regression"
            )
            st.subheader("Test Metrics")
            st.json(metrics)

            fig_avp = viz.actual_vs_predicted_plot(y_test, preds)
            with st.container():
                ui.display_plotly_chart(fig_avp, use_container_width=True)
                st.markdown("&nbsp;")
            with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
                viz.export_figure(fig_avp, Path(tmp.name))
                tmp.seek(0)
                st.download_button(
                    "Download Actual vs Predicted",
                    data=tmp.read(),
                    file_name="actual_vs_predicted.png",
                    mime="image/png",
                )

            fig_resid = viz.residual_plot(y_test, preds)
            with st.container():
                ui.display_plotly_chart(fig_resid, use_container_width=True)
                st.markdown("&nbsp;")
            with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
                viz.export_figure(fig_resid, Path(tmp.name))
                tmp.seek(0)
                st.download_button(
                    "Download Residual Plot",
                    data=tmp.read(),
                    file_name="residuals.png",
                    mime="image/png",
                )

            if st.checkbox("Show Feature Importance", key="fi_reg"):
                fig_imp_r = viz.feature_importance_plot(reg, feature_cols_r)
                with st.container():
                    ui.display_plotly_chart(fig_imp_r, use_container_width=True)
                    st.markdown("&nbsp;")
                with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
                    viz.export_figure(fig_imp_r, Path(tmp.name))
                    tmp.seek(0)
                    st.download_button(
                        "Download Feature Importance",
                        data=tmp.read(),
                        file_name="feature_importance_reg.png",
                        mime="image/png",
                    )

            if st.checkbox("Show SHAP Summary", key="shap_reg"):
                shap_fig_r = viz.shap_summary_plot(reg, X_train)
                st.pyplot(shap_fig_r)
                with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
                    viz.export_figure(shap_fig_r, Path(tmp.name))
                    tmp.seek(0)
                    st.download_button(
                        "Download SHAP Summary",
                        data=tmp.read(),
                        file_name="shap_summary_reg.png",
                        mime="image/png",
                    )
        except Exception as exc:  # pragma: no cover - training can fail for many reasons
            progress_r.progress(0)
            logging.getLogger(__name__).exception(
                "Regression training failed: %s", exc
            )
            st.error(f"Regression training failed: {exc}")

    if st.button("Compare Regression Models") and feature_cols_r:
        df_r = data[feature_cols_r + [target_r]]
        X = df_r.drop(columns=[target_r])
        y = df_r[target_r]
        models_dict = {
            "Linear": LinearRegression(),
            "DecisionTree": DecisionTreeRegressor(),
            "RandomForest": RandomForestRegressor(random_state=42),
        }
        results = model.compare_models(models_dict, X, y, cv=5, scoring="r2")
        st.write(results)

