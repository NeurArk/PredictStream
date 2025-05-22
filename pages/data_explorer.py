"""Data Explorer page for PredictStream."""

import streamlit as st
from utils import config
from utils import data as data_utils
from utils import eda
from utils import ui
from utils import components
import logging
from utils.logging import configure_logging

configure_logging()



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
        data_utils.upload_data_to_session(
            "Upload CSV or Excel file",
            session_key="data",
            datetime_key="datetime_cols",
            uploader_key="file_uploader",
            help="Supported formats: CSV, XLSX",
        )

        st.subheader("Sample Datasets")
        for name, path in config.SAMPLE_DATASETS.items():
            if st.button(f"Load {name}"):
                try:
                    st.session_state["data"] = data_utils.load_data(path)
                    st.session_state["data"] = data_utils.convert_dtypes(
                        st.session_state["data"]
                    )
                    st.session_state["datetime_cols"] = eda.detect_datetime_columns(
                        st.session_state["data"]
                    )
                    st.success(f"{name} loaded!")
                except (ValueError, TypeError) as exc:
                    logging.getLogger(__name__).error(
                        "Failed to load sample data %s: %s", name, exc
                    )
                    st.error(f"Failed to load sample data: {exc}")

        with st.expander("Help"):
            st.markdown(ui.help_markdown())



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

        components.visualization_section(data)
        data = components.transformation_section(data)
        components.classification_training_section(data)
        components.regression_training_section(data)


if __name__ == "__main__":
    main()

