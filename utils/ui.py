"""UI helper utilities."""

from __future__ import annotations

from typing import Dict

from . import config
import streamlit as st

THEME_CSS: Dict[str, str] = {
    "Light": """
<style>
[data-testid='stSidebar'] {background-color: #f0f2f6;}
.plotly-chart .main-svg {background-color: #ffffff;}
</style>
""",
    "Dark": """
<style>
body {background-color: #0e1117; color: #f0f0f0;}
[data-testid='stSidebar'] {background-color: #262730;}
.plotly-chart .main-svg {background-color: #0e1117;}
</style>
""",
}

# Branding colors
BRAND_PRIMARY = "#4B8BBE"
BRAND_CSS = f"""
<style>
:root {{
    --brand-color: {BRAND_PRIMARY};
}}
h1, h2, h3 {{
    color: var(--brand-color);
}}
</style>
"""


def get_theme_css(theme: str) -> str:
    """Return CSS style for the given theme."""
    return THEME_CSS.get(theme, "")


def apply_theme(theme: str | None = None) -> None:
    """Apply the selected theme's CSS to the app."""
    if theme is None:
        theme = st.session_state.get("theme", "Light")
    st.markdown(get_theme_css(theme), unsafe_allow_html=True)


def getting_started_markdown() -> str:
    """Return markdown text for the getting started guide."""
    return (
        "## Getting Started\n"
        "1. Upload your CSV or Excel file using the sidebar.\n"
        "2. Explore the automatic summary statistics and visualizations.\n"
        "3. Select target and feature columns to train models.\n"
        "4. Review metrics and export your results.\n"
        "\n"
        "### Sample Use Cases\n"
        "- Quick exploratory analysis of a new dataset.\n"
        "- Comparing several models on the same data.\n"
    )


def help_markdown() -> str:
    """Return markdown text for the in-app help section."""
    return (
        "### Help\n"
        "- Use the sidebar to upload data or load a sample dataset.\n"
        "- Configure models using the options provided.\n"
        "- Results and metrics appear below each section.\n"
        "- Export figures and predictions using the download buttons."
    )


def apply_branding() -> None:
    """Apply branding styles and logo to the sidebar."""
    st.markdown(BRAND_CSS, unsafe_allow_html=True)
    if config.LOGO_PATH.exists():
        st.sidebar.image(str(config.LOGO_PATH), use_column_width=True)
