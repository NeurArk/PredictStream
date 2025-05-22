"""Prediction page for single and batch predictions."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from utils import data as data_utils
from utils import predict
from utils import ui

st.set_page_config(page_title="Prediction", layout="wide")


def main() -> None:
    """Render the prediction page."""
    ui.apply_branding()
    st.title("Prediction")

    with st.sidebar:
        mode = st.radio("Mode", ["Single", "Batch"], key="pred_mode")
        model_file = st.file_uploader(
            "Upload Model (.joblib)", type=["joblib"], key="model_file"
        )
        data_utils.upload_data_to_session(
            "Upload Data",
            session_key="pred_data",
            uploader_key="pred_data",
        )

    model_obj = None
    if model_file is not None:
        with tempfile.NamedTemporaryFile(suffix=".joblib") as tmp:
            tmp.write(model_file.read())
            tmp.flush()
            model_obj = predict.load_model(Path(tmp.name))

    data = st.session_state.get("pred_data")

    if model_obj is not None and data is not None:
        st.subheader("Data Preview")
        st.dataframe(data.head(), use_container_width=True)

        if mode == "Single":
            idx = st.number_input("Row Index", min_value=0, max_value=len(data)-1, value=0, step=1)
            if st.button("Predict"):
                row = data.iloc[int(idx)]
                pred = predict.predict_single(model_obj, row)
                st.write("Prediction:", pred)
        else:
            if st.button("Run Batch Prediction"):
                preds = predict.predict_batch(model_obj, data)
                st.dataframe(preds.to_frame(), use_container_width=True)
                with tempfile.NamedTemporaryFile(suffix=".csv") as tmp:
                    predict.export_predictions(preds, Path(tmp.name))
                    tmp.seek(0)
                    st.download_button(
                        "Download Predictions",
                        data=tmp.read(),
                        file_name="predictions.csv",
                        mime="text/csv",
                    )


if __name__ == "__main__":
    main()
