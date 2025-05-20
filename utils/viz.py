"""Data visualization utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def histogram(
    df: pd.DataFrame,
    column: str,
    *,
    bins: int = 20,
    density: bool = False,
    title: Optional[str] = None,
) -> go.Figure:
    """Return a histogram or density plot for a column."""
    histnorm = "probability density" if density else None
    fig = px.histogram(df, x=column, nbins=bins, histnorm=histnorm, title=title)
    return fig


def scatter_plot(
    df: pd.DataFrame,
    x: str,
    y: str,
    *,
    color: Optional[str] = None,
    title: Optional[str] = None,
) -> go.Figure:
    """Return a scatter plot."""
    fig = px.scatter(df, x=x, y=y, color=color, title=title)
    return fig


def bar_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    *,
    title: Optional[str] = None,
) -> go.Figure:
    """Return a bar chart."""
    fig = px.bar(df, x=x, y=y, title=title)
    return fig


def pie_chart(
    df: pd.DataFrame,
    names: str,
    values: str,
    *,
    title: Optional[str] = None,
) -> go.Figure:
    """Return a pie chart."""
    fig = px.pie(df, names=names, values=values, title=title)
    return fig


def box_plot(
    df: pd.DataFrame,
    x: str,
    y: str,
    *,
    title: Optional[str] = None,
) -> go.Figure:
    """Return a box plot."""
    fig = px.box(df, x=x, y=y, title=title)
    return fig


def violin_plot(
    df: pd.DataFrame,
    x: str,
    y: str,
    *,
    title: Optional[str] = None,
) -> go.Figure:
    """Return a violin plot."""
    fig = px.violin(df, x=x, y=y, box=True, title=title)
    return fig


def heatmap(
    df: pd.DataFrame,
    *,
    columns: Optional[list[str]] = None,
    title: Optional[str] = None,
) -> go.Figure:
    """Return a correlation heatmap for the given columns."""
    cols = columns or df.select_dtypes(include="number").columns.tolist()
    corr = df[cols].corr()
    fig = px.imshow(corr, text_auto=True, title=title)
    return fig


def export_figure(fig: go.Figure, path: Path) -> None:
    """Export a figure to an HTML file."""
    fig.write_html(str(path))
