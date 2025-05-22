"""Data transformation utilities for preprocessing."""

from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler


MISSING_STRATEGIES = {"drop", "mean", "median", "mode"}
ENCODING_METHODS = {"onehot", "label"}
SCALING_METHODS = {"minmax", "standard"}


def handle_missing_values(df: pd.DataFrame, *, strategy: str = "drop") -> pd.DataFrame:
    """Handle missing values according to the specified strategy."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if strategy not in MISSING_STRATEGIES:
        raise ValueError(f"Invalid strategy: {strategy}")
    if strategy == "drop":
        return df.dropna()
    df = df.copy()
    if strategy == "mean":
        for col in df.select_dtypes(include="number"):
            df[col] = df[col].fillna(df[col].mean())
    elif strategy == "median":
        for col in df.select_dtypes(include="number"):
            df[col] = df[col].fillna(df[col].median())
    elif strategy == "mode":
        for col in df.columns:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].mode().iloc[0])
    return df


def encode_features(
    df: pd.DataFrame,
    columns: Sequence[str],
    *,
    method: str = "onehot",
) -> pd.DataFrame:
    """Encode categorical features using the given method."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    missing = set(columns) - set(df.columns)
    if missing:
        raise KeyError(f"Columns not found: {', '.join(missing)}")
    if method not in ENCODING_METHODS:
        raise ValueError(f"Invalid encoding method: {method}")
    df = df.copy()
    if method == "onehot":
        return pd.get_dummies(df, columns=list(columns), drop_first=False)
    for col in columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    return df


def scale_features(
    df: pd.DataFrame,
    columns: Sequence[str],
    *,
    method: str = "standard",
) -> pd.DataFrame:
    """Scale numeric features with the given method."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    missing = set(columns) - set(df.columns)
    if missing:
        raise KeyError(f"Columns not found: {', '.join(missing)}")
    if not all(pd.api.types.is_numeric_dtype(df[c]) for c in columns):
        raise TypeError("Scale features require numeric columns")
    if method not in SCALING_METHODS:
        raise ValueError(f"Invalid scaling method: {method}")
    df = df.copy()
    scaler = StandardScaler() if method == "standard" else MinMaxScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df
