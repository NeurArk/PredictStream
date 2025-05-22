from __future__ import annotations

from pathlib import Path
import tempfile

import pandas as pd
import streamlit as st

from utils import ui, viz
from utils import time_series as ts
from utils.logging import configure_logging

configure_logging()

st.set_page_config(page_title="Time Series", layout="wide")


def main() -> None:
    """Render the time series analysis page."""
    ui.apply_branding()
    st.title("Time Series Analysis")
    ui.apply_theme()

    df = st.session_state.get("data")
    if df is None or df.empty:
        st.info("No dataset available. Load data on the Data Explorer page.")
        return

    datetime_cols = st.session_state.get("datetime_cols") or []
    if not datetime_cols:
        st.info("No datetime columns detected in the dataset.")
        return

    with st.sidebar:
        time_col = st.selectbox("Datetime Column", datetime_cols, key="ts_time")
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        value_col = st.selectbox("Value Column", numeric_cols, key="ts_value")
        export_fmt = st.selectbox("Export Format", ["png", "jpg"], key="ts_fmt")
        period = st.number_input(
            "Seasonal Period", min_value=2, value=2, step=1, key="ts_period"
        )
        model_choice = st.selectbox(
            "Forecast Model", ["Naive", "ARIMA"], key="ts_model"
        )
        horizon = st.number_input(
            "Forecast Horizon", min_value=1, value=5, step=1, key="ts_horizon"
        )

    if st.button("Generate Plots"):
        ts_fig = viz.time_series_plot(df, time_col, value_col, title="Time Series")
        st.plotly_chart(ts_fig, use_container_width=True)
        with tempfile.NamedTemporaryFile(suffix=f".{export_fmt}") as tmp:
            viz.export_figure(ts_fig, Path(tmp.name))
            tmp.seek(0)
            st.download_button(
                "Download Time Series",
                data=tmp.read(),
                file_name=f"time_series.{export_fmt}",
                mime=f"image/{export_fmt}",
            )

        series = df.set_index(time_col)[value_col]
        if model_choice == "ARIMA":
            try:
                forecast = ts.arima_forecast(series, steps=horizon)
            except ImportError:
                st.error("statsmodels is required for ARIMA forecasting.")
                forecast = None
        else:
            forecast = ts.naive_forecast(series, steps=horizon)

        if forecast is not None:
            st.subheader("Forecast")
            st.write(forecast.to_frame(name="forecast"))

        dec_fig = viz.decomposition_plot(
            df.set_index(time_col)[value_col], period=period, title="Decomposition"
        )
        st.pyplot(dec_fig)
        with tempfile.NamedTemporaryFile(suffix=f".{export_fmt}") as tmp:
            viz.export_figure(dec_fig, Path(tmp.name))
            tmp.seek(0)
            st.download_button(
                "Download Decomposition",
                data=tmp.read(),
                file_name=f"decomposition.{export_fmt}",
                mime=f"image/{export_fmt}",
            )


if __name__ == "__main__":
    main()
