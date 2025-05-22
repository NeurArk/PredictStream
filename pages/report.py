"""Report generation page."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from utils import data as data_utils
from utils import eda
from utils import ui

st.set_page_config(page_title="Report", layout="wide")


def main() -> None:
    """Render the report generation page."""
    ui.apply_branding()
    st.title("Report Generator")

    with st.sidebar:
        data_file = st.file_uploader(
            "Upload Data", type=["csv", "xlsx", "xls"], key="report_data"
        )

    if data_file is not None:
        try:
            df = data_utils.load_data(data_file)
            df = data_utils.convert_dtypes(df)
            st.session_state["report_data"] = df
        except (ValueError, TypeError) as exc:
            st.error(f"Failed to load data: {exc}")

    df = st.session_state.get("report_data")
    if df is not None:
        st.subheader("Data Preview")
        st.dataframe(df.head(), use_container_width=True)
        if st.button("Generate Report"):
            report = eda.profile_report(df)
            st.session_state["report"] = report
            st.success("Report generated!")

    report = st.session_state.get("report")
    if report:
        st.subheader("Summary Statistics")
        st.dataframe(report["summary"], use_container_width=True)
        st.subheader("Data Quality")
        st.dataframe(report["quality"], use_container_width=True)
        st.subheader("Correlation")
        st.dataframe(report["correlation"], use_container_width=True)

        with tempfile.NamedTemporaryFile(suffix=".xlsx") as tmp:
            eda.export_report(report, Path(tmp.name))
            tmp.seek(0)
            st.download_button(
                "Download Report",
                data=tmp.read(),
                file_name="report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )


if __name__ == "__main__":
    main()
