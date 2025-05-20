"""Main entry point for PredictStream."""

import streamlit as st
from utils import config
from utils import data as data_utils
from utils import eda
from utils import model
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="PredictStream", layout="wide")


def main() -> None:
    """Render the main page."""
    st.title("PredictStream")

    with st.sidebar:
        st.header("Data Options")
        uploaded_file = st.file_uploader(
            "Upload CSV or Excel file",
            type=["csv", "xlsx", "xls"],
            key="file_uploader",
        )

        st.subheader("Sample Datasets")
        for name, path in config.SAMPLE_DATASETS.items():
            if st.button(f"Load {name}"):
                st.session_state["data"] = data_utils.load_data(path)
                st.session_state["data"] = data_utils.convert_dtypes(
                    st.session_state["data"]
                )
                st.success(f"{name} loaded!")

    if uploaded_file is not None:
        try:
            df = data_utils.load_data(uploaded_file)
            df = data_utils.convert_dtypes(df)
            st.session_state["data"] = df
            st.success("File loaded successfully!")
        except ValueError as exc:
            st.error(str(exc))

    data = st.session_state.get("data")
    if data is not None:
        st.subheader("Data Preview")
        page_size = 100
        total_pages = max(1, (len(data) - 1) // page_size + 1)
        page = st.number_input(
            "Page",
            min_value=1,
            max_value=total_pages,
            value=1,
            step=1,
        )
        start = (page - 1) * page_size
        end = start + page_size
        st.dataframe(data.iloc[start:end])

        st.subheader("Summary Statistics")
        summary = eda.summary_statistics(data)
        st.dataframe(summary)

        st.subheader("Data Quality")
        quality = eda.data_quality_assessment(data)
        st.dataframe(quality)

        st.subheader("Correlation Matrix")
        corr = eda.correlation_matrix(data)
        st.dataframe(corr)

        st.subheader("Insights")
        for insight in eda.data_insights_summary(data):
            st.write(f"- {insight}")

        st.subheader("Model Training - Classification")
        target = st.selectbox("Target Column", options=data.columns)
        feature_cols = st.multiselect(
            "Feature Columns",
            options=[c for c in data.columns if c != target],
            default=[c for c in data.columns if c != target],
        )
        test_size = st.slider("Test Size", 0.1, 0.5, 0.2, step=0.05)
        model_name = st.selectbox("Model", ["Logistic Regression", "Random Forest"])

        hyperparams = {}
        if model_name == "Logistic Regression":
            hyperparams["C"] = st.number_input("C", 0.01, 10.0, 1.0, step=0.01)
        else:
            hyperparams["n_estimators"] = st.slider("n_estimators", 10, 200, 100, step=10)

        if st.button("Train Model") and feature_cols:
            progress = st.progress(0)
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
            else:
                clf = model.train_random_forest_classifier(
                    X_train,
                    y_train,
                    n_estimators=hyperparams.get("n_estimators", 100),
                    random_state=42,
                )
            progress.progress(75)
            scores = model.cross_validate_model(clf, X_train, y_train, cv=5)
            progress.progress(100)
            st.write("Cross-validation scores:", scores)
            st.write("Mean accuracy:", float(scores.mean()))

        st.subheader("Detected Problem Type")
        st.write(model.detect_problem_type(data[target]))

        st.subheader("Model Training - Regression")
        target_r = st.selectbox("Target Column (regression)", options=data.columns, key="target_reg")
        feature_cols_r = st.multiselect(
            "Feature Columns (reg)",
            options=[c for c in data.columns if c != target_r],
            default=[c for c in data.columns if c != target_r],
            key="features_reg",
        )
        test_size_r = st.slider("Test Size (reg)", 0.1, 0.5, 0.2, step=0.05, key="ts_reg")
        model_name_r = st.selectbox(
            "Model (reg)",
            ["Linear Regression", "Decision Tree", "Random Forest"],
            key="model_reg",
        )

        hyperparams_r = {}
        if model_name_r == "Decision Tree":
            hyperparams_r["max_depth"] = st.slider("max_depth", 1, 20, 3, key="max_depth")
        elif model_name_r == "Random Forest":
            hyperparams_r["n_estimators"] = st.slider("n_estimators", 10, 200, 100, 10, key="n_est")
            hyperparams_r["max_depth"] = st.slider("max_depth_rf", 1, 20, 3, key="max_depth_rf")

        if st.button("Train Regression Model") and feature_cols_r:
            df_r = data[feature_cols_r + [target_r]]
            X_train, X_test, y_train, y_test = model.train_test_split_data(
                df_r,
                target_r,
                test_size=test_size_r,
                random_state=42,
            )
            if model_name_r == "Linear Regression":
                reg = model.train_linear_regression(X_train, y_train)
            elif model_name_r == "Decision Tree":
                reg = model.train_decision_tree_regressor(
                    X_train,
                    y_train,
                    max_depth=hyperparams_r.get("max_depth"),
                    random_state=42,
                )
            else:
                reg = model.train_random_forest_regressor(
                    X_train,
                    y_train,
                    n_estimators=hyperparams_r.get("n_estimators", 100),
                    max_depth=hyperparams_r.get("max_depth_rf"),
                    random_state=42,
                )
            scores = model.cross_validate_model(reg, X_train, y_train, cv=5)
            st.write("CV scores:", scores)
            st.write("Mean R2:", float(scores.mean()))

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


if __name__ == "__main__":
    main()
