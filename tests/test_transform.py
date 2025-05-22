import numpy as np
import pandas as pd
import pytest
from utils import transform


def sample_df():
    return pd.DataFrame({"num": [1.0, np.nan, 3.0], "cat": ["a", "b", None]})


def test_handle_missing_values_drop():
    df = sample_df()
    res = transform.handle_missing_values(df, strategy="drop")
    assert res.isna().sum().sum() == 0
    assert len(res) == 1


def test_handle_missing_values_fill_mean():
    df = sample_df()
    res = transform.handle_missing_values(df, strategy="mean")
    assert res["num"].isna().sum() == 0
    assert res.loc[1, "num"] == pytest.approx((1.0 + 3.0) / 2)


def test_encode_features_onehot():
    df = sample_df().fillna({"cat": "b"})
    res = transform.encode_features(df, ["cat"], method="onehot")
    assert "cat_a" in res.columns and "cat_b" in res.columns


def test_encode_features_label():
    df = sample_df().fillna({"cat": "b"})
    res = transform.encode_features(df, ["cat"], method="label")
    assert pd.api.types.is_integer_dtype(res["cat"])


def test_scale_features_standard():
    df = pd.DataFrame({"num": [1.0, 2.0, 3.0]})
    res = transform.scale_features(df, ["num"], method="standard")
    assert pytest.approx(res["num"].mean(), abs=1e-6) == 0
    assert pytest.approx(res["num"].std(ddof=0), abs=1e-6) == 1


def test_scale_features_minmax():
    df = pd.DataFrame({"num": [1.0, 2.0, 3.0]})
    res = transform.scale_features(df, ["num"], method="minmax")
    assert res["num"].min() == 0
    assert res["num"].max() == 1


def test_transformation_workflow():
    df = sample_df()
    df = transform.handle_missing_values(df, strategy="mean")
    df = transform.encode_features(df, ["cat"], method="label")
    df = transform.scale_features(df, ["num"], method="minmax")
    assert df["num"].min() == 0
    assert "cat" in df.columns
