"""Prediction and export utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd
from joblib import dump, load
from sklearn.base import BaseEstimator

from .model import detect_problem_type
from . import viz


def predict_single(model: BaseEstimator, row: pd.Series | Dict[str, Any] | pd.DataFrame) -> Any:
    """Return prediction for a single data row."""
    if isinstance(row, dict):
        df = pd.DataFrame([row])
    elif isinstance(row, pd.Series):
        df = row.to_frame().T
    elif isinstance(row, pd.DataFrame):
        if len(row) != 1:
            raise ValueError("DataFrame must contain exactly one row")
        df = row
    else:
        raise TypeError("row must be dict, Series, or single-row DataFrame")
    return model.predict(df)[0]


def predict_batch(model: BaseEstimator, df: pd.DataFrame) -> pd.Series:
    """Return predictions for a DataFrame."""
    preds = model.predict(df)
    return pd.Series(preds, index=df.index, name="prediction")


def prediction_plot(y_true: pd.Series, y_pred: pd.Series) -> Any:
    """Return a plot comparing predictions to true values."""
    problem_type = detect_problem_type(y_true)
    if problem_type == "classification":
        return viz.confusion_matrix_plot(y_true, y_pred)
    return viz.actual_vs_predicted_plot(y_true, y_pred)


def export_predictions(preds: pd.Series, path: Path) -> None:
    """Export predictions to CSV or Excel based on file extension."""
    if path.suffix.lower() == ".csv":
        preds.to_csv(path, index=True)
    elif path.suffix.lower() in {".xls", ".xlsx"}:
        preds.to_excel(path, index=True)
    else:
        raise ValueError(f"Unsupported export format: {path.suffix}")


def export_model(model: BaseEstimator, path: Path) -> None:
    """Export a trained model to disk."""
    dump(model, path)


def load_model(path: Path) -> BaseEstimator:
    """Load a model from disk."""
    return load(path)


def save_project(path: Path, *, model: BaseEstimator, data: pd.DataFrame, predictions: pd.Series | None = None) -> None:
    """Save project components to a joblib file."""
    dump({"model": model, "data": data, "predictions": predictions}, path)


def load_project(path: Path) -> Dict[str, Any]:
    """Load project components from a joblib file."""
    return load(path)
