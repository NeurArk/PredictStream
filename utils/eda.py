"""Exploratory data analysis utilities."""

from __future__ import annotations

from typing import Any, Dict, List
from functools import wraps

import numpy as np
import pandas as pd


def _hash_df(df: pd.DataFrame) -> int:
    """Return a hash for a DataFrame."""
    return int(pd.util.hash_pandas_object(df, index=True).sum())


def df_cache(func):
    """Cache DataFrame-returning functions based on input hash."""

    cache: Dict[tuple, Any] = {}

    @wraps(func)
    def wrapper(df: pd.DataFrame, *args, **kwargs):
        key = (_hash_df(df), func.__name__, args, tuple(sorted(kwargs.items())))
        if key not in cache:
            cache[key] = func(df.copy(), *args, **kwargs)
        return cache[key]

    return wrapper


@df_cache
def summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Return summary statistics for all columns."""
    return df.describe(include="all")


@df_cache
def data_quality_assessment(df: pd.DataFrame) -> pd.DataFrame:
    """Return data quality metrics for each column."""
    total = len(df)
    return pd.DataFrame({
        "dtype": df.dtypes,
        "missing": df.isna().sum(),
        "missing_percent": df.isna().mean() * 100,
        "unique": df.nunique(dropna=False),
    })


@df_cache
def correlation_matrix(df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    """Return the correlation matrix for numeric columns."""
    numeric_df = df.select_dtypes(include="number")
    return numeric_df.corr(method=method)


def numeric_distributions(df: pd.DataFrame, bins: int = 10) -> Dict[str, pd.Series]:
    """Return histogram counts for numeric columns."""
    histograms: Dict[str, pd.Series] = {}
    numeric_df = df.select_dtypes(include="number")
    for column in numeric_df.columns:
        histograms[column] = pd.cut(numeric_df[column], bins=bins).value_counts().sort_index()
    return histograms


def categorical_analysis(df: pd.DataFrame, top_n: int = 10) -> Dict[str, pd.Series]:
    """Return value counts for categorical columns."""
    counts: Dict[str, pd.Series] = {}
    categorical_df = df.select_dtypes(exclude="number")
    for column in categorical_df.columns:
        counts[column] = categorical_df[column].value_counts(dropna=False).head(top_n)
    return counts


def missing_value_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return a boolean matrix indicating missing values."""
    return df.isna()


def profile_report(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate a simple data profile report."""
    return {
        "summary": summary_statistics(df),
        "quality": data_quality_assessment(df),
        "correlation": correlation_matrix(df),
    }


def data_insights_summary(df: pd.DataFrame) -> List[str]:
    """Generate simple insights from the data."""
    insights: List[str] = []
    quality = data_quality_assessment(df)
    missing_cols = quality[quality["missing"] > 0].index.tolist()
    if missing_cols:
        insights.append("Columns with missing values: " + ", ".join(missing_cols))

    corr = correlation_matrix(df).abs()
    if not corr.empty:
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        strong = upper.stack().loc[lambda s: s > 0.8]
        if not strong.empty:
            pairs = [f"{i} & {j}" for i, j in strong.index]
            insights.append("Strong correlations detected: " + ", ".join(pairs))

    if not insights:
        insights.append("No notable data issues detected.")
    return insights
