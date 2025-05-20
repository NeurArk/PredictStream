"""UI helper utilities."""

from __future__ import annotations

from typing import Dict

THEME_CSS: Dict[str, str] = {
    "Light": "",
    "Dark": "body { background-color: #0e1117; color: #f0f0f0; }",
}


def get_theme_css(theme: str) -> str:
    """Return CSS style for the given theme."""
    return THEME_CSS.get(theme, "")


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
