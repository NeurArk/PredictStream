"""Modeling utilities for classification tasks."""

from __future__ import annotations

from functools import wraps
from pathlib import Path
from typing import Callable, Dict, Tuple

import pandas as pd
import numpy as np
import streamlit as st
from joblib import dump
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor
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
    problem = detect_problem_type(y)
    strat = y if problem == "classification" else None
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=strat,
    )


@st.cache_resource
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


@st.cache_resource
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


@st.cache_resource
@cache_model
def train_xgboost_classifier(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int = 3,
    random_state: int | None = None,
) -> XGBClassifier:
    """Train an XGBoost classifier."""
    model = XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state,
        eval_metric="logloss",
        use_label_encoder=False,
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


def detect_problem_type(y: pd.Series) -> str:
    """Return 'classification' or 'regression' based on target variable."""
    if pd.api.types.is_numeric_dtype(y):
        unique = y.nunique(dropna=False)
        threshold = max(20, int(0.05 * len(y)))
        return "classification" if unique <= threshold else "regression"
    return "classification"


@st.cache_resource
@cache_model
def train_linear_regression(
    X: pd.DataFrame,
    y: pd.Series,
) -> LinearRegression:
    """Train a Linear Regression model."""
    model = LinearRegression()
    model.fit(X, y)
    return model


@st.cache_resource
@cache_model
def train_decision_tree_regressor(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    max_depth: int | None = None,
    random_state: int | None = None,
) -> DecisionTreeRegressor:
    """Train a Decision Tree Regressor."""
    model = DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)
    model.fit(X, y)
    return model


@st.cache_resource
@cache_model
def train_random_forest_regressor(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_estimators: int = 100,
    max_depth: int | None = None,
    random_state: int | None = None,
) -> RandomForestRegressor:
    """Train a Random Forest Regressor."""
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
    )
    model.fit(X, y)
    return model


@st.cache_resource
@cache_model
def train_xgboost_regressor(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int = 3,
    random_state: int | None = None,
) -> XGBRegressor:
    """Train an XGBoost regressor."""
    model = XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state,
    )
    model.fit(X, y)
    return model


def compare_models(
    models: Dict[str, BaseEstimator],
    X: pd.DataFrame,
    y: pd.Series,
    *,
    cv: int = 5,
    scoring: str | None = None,
) -> Dict[str, float]:
    """Return mean cross-validation scores for multiple models."""
    results: Dict[str, float] = {}
    for name, mdl in models.items():
        scores = cross_val_score(mdl, X, y, cv=cv, scoring=scoring)
        results[name] = float(scores.mean())
    return results


def save_model(model: BaseEstimator, path: Path) -> None:
    """Serialize a trained model to disk."""
    dump(model, path)
