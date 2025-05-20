"""Main entry point for PredictStream."""

import streamlit as st
from utils import config

st.set_page_config(page_title="PredictStream", layout="wide")


def main() -> None:
    """Render the main page."""
    st.title("PredictStream")
    st.write("Upload a dataset to get started or explore sample datasets.")

    uploaded_file = st.file_uploader(
        "Upload CSV or Excel file", type=["csv", "xlsx", "xls"], key="file_uploader"
    )

    if uploaded_file is not None:
        st.success("File uploaded successfully!")

    st.subheader("Sample Datasets")
    for name, path in config.SAMPLE_DATASETS.items():
        if st.button(f"Load {name}"):
            st.session_state["uploaded_file"] = path.read_bytes()
            st.success(f"{name} loaded!")


if __name__ == "__main__":
    main()
