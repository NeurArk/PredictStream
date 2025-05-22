"""Data loading and preprocessing utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def load_data(file: Any) -> pd.DataFrame:
    """Load a CSV or Excel file into a DataFrame."""
    if file is None:
        raise ValueError("No file provided")

    if hasattr(file, "name"):
        ext = Path(file.name).suffix.lower()
    else:
        path = Path(str(file))
        ext = path.suffix.lower()
        if not path.exists():
            raise ValueError("Invalid file path")

    try:
        if ext == ".csv":
            return pd.read_csv(file)
        if ext in {".xls", ".xlsx"}:
            return pd.read_excel(file)
    except Exception as exc:  # pragma: no cover - pass through
        raise ValueError(f"Failed to read file: {exc}") from exc
    raise ValueError(f"Unsupported file type: {ext}")


def convert_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Attempt to convert object columns to numeric or datetime."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    for column in df.columns:
        if df[column].dtype == object:
            df[column] = pd.to_numeric(df[column], errors="ignore")
            if df[column].dtype == object:
                df[column] = pd.to_datetime(df[column], errors="ignore")
    return df


def data_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return a statistical summary of the dataframe."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if df.empty:
        raise ValueError("DataFrame is empty")
    return df.describe(include="all")
