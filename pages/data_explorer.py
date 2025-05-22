"""Data Explorer page for PredictStream."""

import streamlit as st
from utils import config
from utils import data as data_utils
from utils import eda
from utils import model
from utils import transform
from utils import ui
from utils import viz
from pathlib import Path
import tempfile
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Data Explorer", layout="wide")


def main() -> None:
    """Render the data exploration and modeling page."""
    ui.apply_branding()
    st.title("Data Explorer")
    theme = st.sidebar.selectbox(
        "Theme",
        ["Light", "Dark"],
        help="Toggle interface theme",
        key="theme",
    )
    st.markdown(ui.get_theme_css(theme), unsafe_allow_html=True)

    with st.expander("Getting Started"):
        st.markdown(ui.getting_started_markdown())

    with st.sidebar:
        st.header("Data Options")
        uploaded_file = st.file_uploader(
            "Upload CSV or Excel file",
            type=["csv", "xlsx", "xls"],
            key="file_uploader",
            help="Supported formats: CSV, XLSX",
        )

        st.subheader("Sample Datasets")
        for name, path in config.SAMPLE_DATASETS.items():
            if st.button(f"Load {name}"):
                st.session_state["data"] = data_utils.load_data(path)
                st.session_state["data"] = data_utils.convert_dtypes(
                    st.session_state["data"]
                )
                st.success(f"{name} loaded!")

        with st.expander("Help"):
            st.markdown(ui.help_markdown())

    if uploaded_file is not None:
        try:
            df = data_utils.load_data(uploaded_file)
            df = data_utils.convert_dtypes(df)
            st.session_state["data"] = df
            st.success("File loaded successfully!")
        except ValueError as exc:
            st.error(f"Failed to load file: {exc}")

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
        st.dataframe(data.iloc[start:end], use_container_width=True)

        st.subheader("Summary Statistics")

        @st.cache_data
        def _summary(df):
            return eda.summary_statistics(df)

        summary = _summary(data)
        st.dataframe(summary, use_container_width=True)

        st.subheader("Data Quality")

        @st.cache_data
        def _quality(df):
            return eda.data_quality_assessment(df)

        quality = _quality(data)
        st.dataframe(quality, use_container_width=True)

        st.subheader("Correlation Matrix")

        @st.cache_data
        def _corr(df):
            return eda.correlation_matrix(df)

        corr = _corr(data)
        st.dataframe(corr, use_container_width=True)

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

        st.subheader("Insights")
        for insight in eda.data_insights_summary(data):
            st.write(f"- {insight}")

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
            df_trans = data.copy()
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
                df_trans = transform.encode_features(df_trans, encode_cols, method=method)
            if scale_cols:
                method = "standard" if scale_method == "Standard" else "minmax"
                df_trans = transform.scale_features(df_trans, scale_cols, method=method)
            st.session_state["data"] = df_trans
            data = df_trans
            st.success("Transformations applied!")

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
            ["Logistic Regression", "Random Forest"],
            help="Choose a classification algorithm",
        )

        hyperparams = {}
        if model_name == "Logistic Regression":
            hyperparams["C"] = st.number_input(
                "C",
                0.01,
                10.0,
                1.0,
                step=0.01,
                help="Inverse of regularization strength",
            )
        else:
            hyperparams["n_estimators"] = st.slider(
                "n_estimators",
                10,
                200,
                100,
                step=10,
                help="Number of trees",
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
            except Exception as exc:
                progress.progress(0)
                st.error(f"Classification training failed: {exc}")

        st.subheader("Detected Problem Type")
        st.write(model.detect_problem_type(data[target]))

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
            ["Linear Regression", "Decision Tree", "Random Forest"],
            key="model_reg",
            help="Choose a regression algorithm",
        )

        hyperparams_r = {}
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
                else:
                    reg = model.train_random_forest_regressor(
                        X_train,
                        y_train,
                        n_estimators=hyperparams_r.get("n_estimators", 100),
                        max_depth=hyperparams_r.get("max_depth_rf"),
                        random_state=42,
                    )
                progress_r.progress(75)
                scores = model.cross_validate_model(reg, X_train, y_train, cv=5)
                progress_r.progress(100)
                st.write("CV scores:", scores)
                st.write("Mean R2:", float(scores.mean()))
            except Exception as exc:
                progress_r.progress(0)
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


if __name__ == "__main__":
    main()
