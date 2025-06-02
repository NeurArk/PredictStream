"""Data visualization utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import logging

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    confusion_matrix as sk_confusion_matrix,
)
import matplotlib.pyplot as plt
import shap
from .eda import naive_seasonal_decompose


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
    fig.update_layout(height=500, margin=dict(l=20, r=20, t=40, b=20))
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
    fig.update_layout(height=500, margin=dict(l=20, r=20, t=40, b=20))
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
    fig.update_layout(height=500, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def pair_plot(
    df: pd.DataFrame,
    *,
    columns: Optional[list[str]] = None,
    hue: Optional[str] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """Return a pair plot figure for the selected columns."""
    cols = columns or df.select_dtypes(include="number").columns.tolist()
    plot_df = df[cols].copy()
    if hue and hue in df.columns and hue not in cols:
        plot_df[hue] = df[hue]
    grid = sns.pairplot(plot_df, hue=hue)
    fig = grid.fig
    if title:
        fig.suptitle(title)
        fig.tight_layout()
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
    
    # Use px.imshow for simpler and more stable heatmap
    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale='RdBu',
        color_continuous_midpoint=0,
        title=title or "Correlation Heatmap"
    )
    
    # Minimal layout updates
    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig


def export_figure(fig: object, path: Path) -> None:
    """Export a figure to HTML or image formats."""
    ext = path.suffix.lower()
    if ext == ".html":
        if isinstance(fig, go.Figure):
            fig.write_html(str(path))
        else:
            logging.getLogger(__name__).error("HTML export requires a Plotly figure")
            raise ValueError("HTML export requires a Plotly figure")
    elif ext in {".png", ".jpg", ".jpeg"}:
        if isinstance(fig, go.Figure):
            try:
                # Create a copy to avoid modifying the original figure
                fig_copy = go.Figure(fig)
                # Apply white background for export
                fig_copy.update_layout(
                    template="plotly_white",
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                )
                fig_copy.write_image(str(path), width=1200, height=800, scale=2)
            except ValueError as exc:
                logging.getLogger(__name__).exception(
                    "Static image export failed: %s", exc
                )
                raise RuntimeError(
                    "Static image export requires the kaleido package"
                ) from exc
        elif isinstance(fig, plt.Figure):
            fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
        else:
            logging.getLogger(__name__).error("Unsupported figure type for image export")
            raise ValueError("Unsupported figure type for image export")
    else:
        logging.getLogger(__name__).error("Unsupported export format: %s", ext)
        raise ValueError(f"Unsupported export format: {ext}")


def confusion_matrix_plot(y_true, y_pred, *, title: Optional[str] = None) -> go.Figure:
    """Return a confusion matrix heatmap."""
    cm = sk_confusion_matrix(y_true, y_pred)
    fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues", title=title)
    fig.update_xaxes(title="Predicted")
    fig.update_yaxes(title="Actual")
    fig.update_layout(height=500, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def roc_curve_plot(y_true, y_score, *, title: Optional[str] = None) -> go.Figure:
    """Return ROC curve figure."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC"))
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash"))
    fig.update_layout(
        title=title or "ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
    )
    return fig


def precision_recall_curve_plot(y_true, y_score, *, title: Optional[str] = None) -> go.Figure:
    """Return precision-recall curve figure."""
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name="PR"))
    fig.update_layout(
        title=title or "Precision-Recall Curve",
        xaxis_title="Recall",
        yaxis_title="Precision",
    )
    return fig


def actual_vs_predicted_plot(y_true, y_pred, *, title: Optional[str] = None) -> go.Figure:
    """Return actual vs predicted scatter plot."""
    fig = px.scatter(x=y_true, y=y_pred, labels={"x": "Actual", "y": "Predicted"}, title=title)
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    fig.add_shape(type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val, line=dict(dash="dash"))
    return fig


def residual_plot(y_true, y_pred, *, title: Optional[str] = None) -> go.Figure:
    """Return residual plot."""
    residuals = np.array(y_true) - np.array(y_pred)
    fig = px.scatter(x=y_pred, y=residuals, labels={"x": "Predicted", "y": "Residual"}, title=title)
    fig.add_shape(type="line", x0=np.min(y_pred), y0=0, x1=np.max(y_pred), y1=0, line=dict(dash="dash"))
    return fig


def feature_importance_plot(model, feature_names: list[str], *, title: Optional[str] = None) -> go.Figure:
    """Return feature importance bar chart."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_)
        if importances.ndim > 1:
            importances = importances[0]
    else:
        raise ValueError("Model has no feature importances")
    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    df = df.sort_values("importance", ascending=False)
    fig = px.bar(df, x="feature", y="importance", title=title or "Feature Importance")
    return fig


def shap_summary_plot(model, X: pd.DataFrame, *, title: Optional[str] = None):
    """Return SHAP summary plot as a Matplotlib figure."""
    explainer = shap.Explainer(model, X)
    values = explainer(X)
    shap.plots.beeswarm(values, show=False)
    fig = plt.gcf()
    if title:
        fig.suptitle(title)
    return fig


def time_series_plot(
    df: pd.DataFrame,
    time_col: str,
    value_col: str,
    *,
    title: Optional[str] = None,
) -> go.Figure:
    """Return a line plot for a time series."""
    plot_df = df.sort_values(time_col)
    fig = px.line(plot_df, x=time_col, y=value_col, title=title)
    return fig


def decomposition_plot(
    series: pd.Series,
    *,
    period: int,
    title: Optional[str] = None,
) -> plt.Figure:
    """Return a basic additive decomposition plot."""
    parts = naive_seasonal_decompose(series, period)
    fig, axes = plt.subplots(4, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(series.index, series.values)
    axes[0].set_title("Observed")
    axes[1].plot(series.index, parts["trend"].values)
    axes[1].set_title("Trend")
    axes[2].plot(series.index, parts["seasonal"].values)
    axes[2].set_title("Seasonality")
    axes[3].plot(series.index, parts["resid"].values)
    axes[3].set_title("Residual")
    plt.tight_layout()
    if title:
        fig.suptitle(title)
        fig.tight_layout()
    return fig
