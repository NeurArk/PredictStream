"""Data loading and preprocessing utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable
import logging

MAX_UPLOAD_SIZE_MB = 100


def validate_file_size(file: Any, max_mb: int = MAX_UPLOAD_SIZE_MB) -> int:
    """Validate that file size does not exceed ``max_mb`` megabytes.

    Parameters
    ----------
    file:
        A file-like object or path.
    max_mb:
        Maximum size in megabytes allowed.

    Returns
    -------
    int
        Size of the file in bytes.

    Raises
    ------
    ValueError
        If the file exceeds ``max_mb`` megabytes.
    """
    limit = max_mb * 1024 * 1024
    size = getattr(file, "size", None)
    if size is None:
        try:
            if isinstance(file, (str, Path)):
                path = Path(file)
            else:
                path = Path(getattr(file, "name"))
            size = path.stat().st_size
        except OSError as exc:
            logging.getLogger(__name__).warning("Could not determine file size: %s", exc)
            size = None
    if size is not None and size > limit:
        raise ValueError(f"File size {size} exceeds limit of {limit} bytes")
    return size or 0


import streamlit as st

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
            logging.getLogger(__name__).error("Invalid file path: %s", path)
            raise ValueError("Invalid file path")

    try:
        if ext == ".csv":
            return pd.read_csv(file)
        if ext in {".xls", ".xlsx"}:
            return pd.read_excel(file)
    except Exception as exc:  # pragma: no cover - pass through
        logging.getLogger(__name__).exception("Failed to read file: %s", exc)
        raise ValueError(f"Failed to read file: {exc}") from exc
    logging.getLogger(__name__).error("Unsupported file type: %s", ext)
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


def validate_file_type(file: Any, allowed_types: Iterable[str]) -> str:
    """Return the lowercase file extension if allowed."""
    if file is None:
        raise ValueError("No file provided")

    if hasattr(file, "name"):
        ext = Path(file.name).suffix.lower()
    else:
        path = Path(str(file))
        ext = path.suffix.lower()
        if not path.exists():
            logging.getLogger(__name__).error("Invalid file path: %s", path)
            raise ValueError("Invalid file path")

    if ext not in {f".{t.lstrip('.').lower()}" for t in allowed_types}:
        logging.getLogger(__name__).error("Unsupported file type: %s", ext)
        raise ValueError(f"Unsupported file type: {ext}")
    return ext


def process_uploaded_file(
    uploaded_file: Any,
    *,
    session_key: str,
    detect_datetime: bool = False,
    datetime_key: str = "datetime_cols",
    max_size_mb: int = MAX_UPLOAD_SIZE_MB,
) -> pd.DataFrame | None:
    """Load an uploaded file and store the DataFrame in session state."""
    if uploaded_file is None:
        return None
    try:
        _ = validate_file_size(uploaded_file, max_size_mb)
        _ = validate_file_type(uploaded_file, ["csv", "xls", "xlsx"])
        df = load_data(uploaded_file)
        df = convert_dtypes(df)
    except (ValueError, TypeError) as exc:  # pragma: no cover - tested via wrapper
        logging.getLogger(__name__).error("Failed to load uploaded file: %s", exc)
        st.error(f"Failed to load file: {exc}")
        return None
    st.session_state[session_key] = df
    if detect_datetime:
        from . import eda

        st.session_state[datetime_key] = eda.detect_datetime_columns(df)
    return df


def upload_data_to_session(
    label: str,
    *,
    session_key: str,
    datetime_key: str | None = None,
    uploader_key: str | None = None,
    help: str | None = None,
    types: Iterable[str] = ("csv", "xlsx", "xls"),
    max_size_mb: int = MAX_UPLOAD_SIZE_MB,
) -> pd.DataFrame | None:
    """Upload a file and store the loaded DataFrame in session state."""
    file = st.file_uploader(
        label,
        type=list(types),
        key=uploader_key or f"{session_key}_uploader",
        help=help,
    )
    return process_uploaded_file(
        file,
        session_key=session_key,
        detect_datetime=datetime_key is not None,
        datetime_key=datetime_key or "datetime_cols",
        max_size_mb=max_size_mb,
    )
