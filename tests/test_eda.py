import pandas as pd
from utils import eda


def sample_df():
    return pd.DataFrame({
        'num1': [1, 2, 3, 4, 5],
        'num2': [1, 2, 3, 4, 5],
        'cat': ['a', 'b', 'a', None, 'b'],
    })


def test_summary_statistics():
    df = sample_df()
    summary = eda.summary_statistics(df)
    assert 'num1' in summary.columns
    assert 'cat' in summary.columns


def test_data_quality_assessment():
    df = sample_df()
    quality = eda.data_quality_assessment(df)
    assert quality.loc['cat', 'missing'] == 1
    assert quality.loc['num1', 'missing'] == 0


def test_correlation_matrix():
    df = sample_df()
    corr = eda.correlation_matrix(df)
    assert corr.loc['num1', 'num2'] == 1.0


def test_numeric_distributions():
    df = sample_df()
    hists = eda.numeric_distributions(df, bins=2)
    assert 'num1' in hists
    assert hists['num1'].sum() == len(df)


def test_categorical_analysis():
    df = sample_df()
    counts = eda.categorical_analysis(df)
    assert counts['cat']['a'] == 2


def test_missing_value_matrix():
    df = sample_df()
    matrix = eda.missing_value_matrix(df)
    assert matrix['cat'].sum() == 1


def test_profile_report():
    df = sample_df()
    report = eda.profile_report(df)
    assert 'summary' in report and 'quality' in report and 'correlation' in report


def test_data_insights_summary():
    df = sample_df()
    insights = eda.data_insights_summary(df)
    assert any('missing values' in text for text in insights)


def test_eda_caching():
    df = sample_df()
    first = eda.summary_statistics(df)
    second = eda.summary_statistics(df)
    assert first is second


def test_detect_datetime_columns_and_decompose():
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    df = pd.DataFrame({"date": dates, "val": range(10)})
    cols = eda.detect_datetime_columns(df)
    assert cols == ["date"]
    parts = eda.naive_seasonal_decompose(df["val"], period=2)
    assert set(parts) == {"trend", "seasonal", "resid"}
