"""Modeling page placeholder."""

import streamlit as st
from utils import ui

st.set_page_config(page_title="Modeling", layout="wide")


def main() -> None:
    """Render the modeling page."""
    ui.apply_branding()
    st.title("Modeling")
    st.write("This page will host advanced modeling features in future releases.")


if __name__ == "__main__":
    main()
