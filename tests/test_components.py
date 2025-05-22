import pandas as pd
import streamlit as st
from importlib import import_module

from utils import components


def sample_df():
    return pd.DataFrame({"a": [1, 2], "b": [3, 4], "target": [0, 1]})


def test_components_functions_exist():
    mod = import_module("utils.components")
    for name in (
        "visualization_section",
        "transformation_section",
        "classification_training_section",
        "regression_training_section",
    ):
        assert hasattr(mod, name)


def test_components_functions_run():
    df = sample_df()
    st.session_state.clear()
    components.visualization_section(df)
    df2 = components.transformation_section(df)
    assert isinstance(df2, pd.DataFrame)
    components.classification_training_section(df)
    components.regression_training_section(df)
