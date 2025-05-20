"""Model evaluation utilities."""

from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    confusion_matrix as sk_confusion_matrix,
    roc_curve,
    precision_recall_curve,
)


def performance_metrics(
    y_true: Iterable,
    y_pred: Iterable,
    *,
    problem_type: str,
) -> Dict[str, float]:
    """Return performance metrics based on problem type."""
    if problem_type == "classification":
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        }
    if problem_type == "regression":
        mse = mean_squared_error(y_true, y_pred)
        return {
            "mae": mean_absolute_error(y_true, y_pred),
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
            "r2": r2_score(y_true, y_pred),
        }
    raise ValueError(f"Unknown problem_type: {problem_type}")


def confusion_matrix(y_true: Iterable, y_pred: Iterable) -> pd.DataFrame:
    """Return confusion matrix as DataFrame."""
    cm = sk_confusion_matrix(y_true, y_pred)
    return pd.DataFrame(cm)


def roc_curve_data(y_true: Iterable, y_score: Iterable) -> pd.DataFrame:
    """Return false positive rate, true positive rate, and thresholds."""
    fpr, tpr, thresh = roc_curve(y_true, y_score)
    return pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresh})


def precision_recall_curve_data(y_true: Iterable, y_score: Iterable) -> pd.DataFrame:
    """Return precision-recall curve values."""
    precision, recall, thresh = precision_recall_curve(y_true, y_score)
    return pd.DataFrame({"precision": precision, "recall": recall, "threshold": np.append(thresh, np.nan)})

