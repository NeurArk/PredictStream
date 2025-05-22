import pandas as pd
import pytest

from utils import time_series


def sample_series():
    return pd.Series(
        range(5), index=pd.date_range("2021-01-01", periods=5, freq="D")
    )


def test_naive_forecast_extends_index():
    series = sample_series()
    forecast = time_series.naive_forecast(series, steps=3)
    assert len(forecast) == 3
    assert forecast.iloc[0] == series.iloc[-1]
    assert forecast.index[0] == series.index[-1] + series.index.freq


def test_arima_forecast_runs_or_errors():
    series = sample_series()
    try:
        fc = time_series.arima_forecast(series, steps=2)
        assert len(fc) == 2
    except ImportError:
        pytest.skip("statsmodels not available")


