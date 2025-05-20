"""Modeling utilities for classification tasks."""

from __future__ import annotations

from functools import wraps
from typing import Callable, Tuple

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator


def cache_model(func: Callable) -> Callable:
    """Simple caching decorator for model training functions."""

    cache = {}

    @wraps(func)
    def wrapper(*args, **kwargs):
        key_parts = [func.__name__]
        for arg in args:
            if isinstance(arg, pd.DataFrame):
                key_parts.append(pd.util.hash_pandas_object(arg, index=True).sum())
            elif isinstance(arg, pd.Series):
                key_parts.append(pd.util.hash_pandas_object(arg, index=True).sum())
            else:
                key_parts.append(repr(arg))
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}={v}")
        key = tuple(key_parts)
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]

    return wrapper


def train_test_split_data(
    df: pd.DataFrame,
    target: str,
    *,
    test_size: float = 0.2,
    random_state: int | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split dataframe into train and test sets."""
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


@cache_model
def train_logistic_regression(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    C: float = 1.0,
    max_iter: int = 1000,
) -> LogisticRegression:
    """Train a Logistic Regression classifier."""
    model = LogisticRegression(C=C, max_iter=max_iter, n_jobs=None)
    model.fit(X, y)
    return model


@cache_model
def train_random_forest_classifier(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_estimators: int = 100,
    max_depth: int | None = None,
    random_state: int | None = None,
) -> RandomForestClassifier:
    """Train a Random Forest classifier."""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
    )
    model.fit(X, y)
    return model


def cross_validate_model(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    cv: int = 5,
) -> np.ndarray:
    """Return cross-validation scores for the given model."""
    scores = cross_val_score(model, X, y, cv=cv)
    return scores
