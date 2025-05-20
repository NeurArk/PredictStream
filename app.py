"""Main entry point for PredictStream."""

import streamlit as st
from utils import config
from utils import data as data_utils
from utils import eda
from utils import model

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


if __name__ == "__main__":
    main()
