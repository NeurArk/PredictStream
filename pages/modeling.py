"""Interactive modeling page for training and comparing models."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Dict, List

import pandas as pd
import streamlit as st

from utils import model, predict, ui
from utils.logging import configure_logging

configure_logging()

st.set_page_config(page_title="Modeling", layout="wide")


def _available_datasets() -> List[str]:
    """Return keys of DataFrames stored in session state."""
    return [k for k, v in st.session_state.items() if isinstance(v, pd.DataFrame)]


def _get_dataset(name: str) -> pd.DataFrame | None:
    """Return DataFrame stored under the given key."""
    val = st.session_state.get(name)
    return val if isinstance(val, pd.DataFrame) else None


def main() -> None:
    """Render the modeling page."""
    ui.apply_branding()
    st.title("Modeling")

    data_keys = _available_datasets()
    if not data_keys:
        st.info("No dataset available. Load data on the Data Explorer page.")
        return

    with st.sidebar:
        dataset_key = st.selectbox("Dataset", data_keys, key="model_dataset")
    df = _get_dataset(dataset_key)
    if df is None or df.empty:
        st.info("Dataset is empty.")
        return

    with st.sidebar:
        target = st.selectbox("Target Column", df.columns, key="model_target")
        features = st.multiselect(
            "Feature Columns",
            [c for c in df.columns if c != target],
            default=[c for c in df.columns if c != target],
            key="model_features",
        )
        problem = st.radio(
            "Problem Type", ["classification", "regression"], key="model_problem"
        )
        cv = st.slider("CV Folds", 2, 10, 5, key="model_cv")

    options = (
        ["Logistic Regression", "Random Forest", "XGBoost"]
        if problem == "classification"
        else ["Linear Regression", "Decision Tree", "Random Forest", "XGBoost"]
    )
    selected = st.multiselect("Models", options, key="model_selection")

    params: Dict[str, Dict] = {}
    if "Logistic Regression" in selected:
        with st.expander("Logistic Regression"):
            params["Logistic Regression"] = {
                "C": st.number_input("C", 0.01, 10.0, 1.0, 0.01, key="lr_C"),
                "max_iter": int(
                    st.number_input("max_iter", 100, 5000, 1000, step=100, key="lr_iter")
                ),
            }
    if "Random Forest" in selected:
        with st.expander("Random Forest"):
            params["Random Forest"] = {
                "n_estimators": int(
                    st.slider("n_estimators", 10, 300, 100, 10, key="rf_n")
                ),
                "max_depth": st.slider("max_depth", 1, 20, 5, key="rf_depth"),
            }
    if "Decision Tree" in selected:
        with st.expander("Decision Tree"):
            params["Decision Tree"] = {
                "max_depth": st.slider("max_depth_dt", 1, 20, 5, key="dt_depth"),
            }
    if "XGBoost" in selected:
        with st.expander("XGBoost"):
            params["XGBoost"] = {
                "n_estimators": int(
                    st.slider("xgb_n_estimators", 10, 300, 100, 10, key="xgb_n")
                ),
                "learning_rate": st.number_input(
                    "learning_rate", 0.01, 1.0, 0.1, 0.01, key="xgb_lr"
                ),
                "max_depth": st.slider("xgb_max_depth", 1, 10, 3, key="xgb_depth"),
            }

    if st.button("Train Models") and selected and features:
        X = df[features]
        y = df[target]
        trained: Dict[str, object] = {}
        results: Dict[str, float] = {}
        for name in selected:
            if problem == "classification":
                if name == "Logistic Regression":
                    mdl = model.train_logistic_regression(X, y, **params.get(name, {}))
                elif name == "Random Forest":
                    mdl = model.train_random_forest_classifier(
                        X, y, random_state=42, **params.get(name, {})
                    )
                elif name == "XGBoost":
                    mdl = model.train_xgboost_classifier(
                        X, y, random_state=42, **params.get(name, {})
                    )
            else:
                if name == "Linear Regression":
                    mdl = model.train_linear_regression(X, y)
                elif name == "Decision Tree":
                    mdl = model.train_decision_tree_regressor(
                        X, y, random_state=42, **params.get(name, {})
                    )
                elif name == "Random Forest":
                    mdl = model.train_random_forest_regressor(
                        X, y, random_state=42, **params.get(name, {})
                    )
                elif name == "XGBoost":
                    mdl = model.train_xgboost_regressor(
                        X, y, random_state=42, **params.get(name, {})
                    )
            scores = model.cross_validate_model(mdl, X, y, cv=cv)
            results[name] = float(scores.mean())
            trained[name] = mdl
        st.session_state["trained_models"] = trained
        st.session_state["cv_results"] = results

    results = st.session_state.get("cv_results")
    if results:
        st.subheader("Cross-Validation Results")
        st.table(pd.DataFrame.from_dict(results, orient="index", columns=["score"]))

    trained_models = st.session_state.get("trained_models")
    if trained_models:
        export_choice = st.selectbox(
            "Select model to export", list(trained_models), key="export_choice"
        )
        if st.button("Export Model"):
            with tempfile.NamedTemporaryFile(suffix=".joblib") as tmp:
                predict.export_model(trained_models[export_choice], Path(tmp.name))
                tmp.seek(0)
                st.download_button(
                    "Download Model",
                    data=tmp.read(),
                    file_name=f"{export_choice.replace(' ', '_').lower()}.joblib",
                    mime="application/octet-stream",
                )


if __name__ == "__main__":
    main()
