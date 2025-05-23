"""Main entry point for PredictStream."""

import streamlit as st
from utils import ui
from utils.logging import configure_logging

configure_logging()

st.set_page_config(page_title="PredictStream", layout="wide")

if "theme" not in st.session_state:
    st.session_state["theme"] = "Light"

ui.apply_theme(st.session_state["theme"])


def main() -> None:
    """Render the home page with navigation links."""
    ui.apply_branding()
    st.title("PredictStream")
    st.write(
        "Use the sidebar to navigate to different sections of the application."
    )
    with st.sidebar:
        st.selectbox("Theme", ["Light", "Dark"], key="theme")
        st.page_link("app.py", label="Home", icon="🏠")
        st.page_link("pages/data_explorer.py", label="Data Explorer", icon="📊")
        st.page_link("pages/modeling.py", label="Modeling", icon="🧠")
        st.page_link("pages/prediction.py", label="Prediction", icon="🔮")
        st.page_link("pages/time_series.py", label="Time Series", icon="📈")
        st.page_link("pages/report.py", label="Report", icon="📄")


if __name__ == "__main__":
    main()
