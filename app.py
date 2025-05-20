"""Main entry point for PredictStream."""

import streamlit as st

from utils import config
from utils import data as data_utils

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

        st.subheader("Data Summary")
        summary = data_utils.data_summary(data)
        st.dataframe(summary)


if __name__ == "__main__":
    main()
