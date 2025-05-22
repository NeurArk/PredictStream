"""Time series forecasting utilities."""

from __future__ import annotations

from typing import Tuple

import pandas as pd


def _extend_index(index: pd.Index, steps: int) -> pd.Index:
    """Return an extended index for forecast values."""
    if isinstance(index, pd.DatetimeIndex) and index.freq is not None:
        start = index[-1] + index.freq
        return pd.date_range(start, periods=steps, freq=index.freq)
    return pd.RangeIndex(index[-1] + 1, index[-1] + 1 + steps)


def naive_forecast(series: pd.Series, steps: int = 1) -> pd.Series:
    """Forecast future values using the last observed value."""
    last = series.iloc[-1]
    index = _extend_index(series.index, steps)
    return pd.Series([last] * steps, index=index, name="naive_forecast")


def arima_forecast(
    series: pd.Series,
    *,
    order: Tuple[int, int, int] = (1, 1, 0),
    steps: int = 1,
) -> pd.Series:
    """Forecast future values using an ARIMA model.

    Requires the ``statsmodels`` package. If it is not installed an
    ``ImportError`` is raised.
    """
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("statsmodels is required for ARIMA forecasting") from exc

    model = ARIMA(series, order=order)
    fitted = model.fit()
    forecast = fitted.forecast(steps=steps)
    index = _extend_index(series.index, steps)
    forecast.index = index
    return forecast
